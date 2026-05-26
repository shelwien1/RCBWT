// rcbwt2: rcbwt + two-pass backward recurrence in build_keys.
// Usage: rcbwt2 inputfile outputfile [dumb-BWT-file]
//
// Output format (both fast and reference paths):
//   [ 4 bytes LE primary index pi ][ n bytes BWT ]
//
// Difference from rcbwt: build_keys() no longer calls build_code at all.
// Only keys[n-1] is seeded (single 64-bit write — no full-array memset).
//
// Pass 1 (write-only) walks j = n-2..0 computing keys[j] from keys[j+1]
// in a strictly non-cyclic order. Pass 2 walks j = n-1..0 refining
// keys[n-1] (which still carries the seed) and the immediate neighbours
// affected by the seed; it exits the moment a key matches its previous
// value, since the recurrence is deterministic in the neighbour and any
// later key would then also be unchanged. The shared body is a template
// parameterized on whether to compare and exit early.
//
//      code[j] = mul_hi(~0, lo) + mul_hi(keys[(j+1)%n] << 16, r)
//
// On book1 this converges in ~n+6 inner steps (vs ~6-7n for rcbwt's
// per-position build_code, and 3n for the previous full-pass variant).
// BWT and pi remain byte-identical to rcbwt and the dumb-BWT oracle.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- constants & fixed-size tables -------------------------------------

// Alphabet (byte values).
static const int CNUM    = 256;
static const int CNUM1   = CNUM + 1;                   // cum-table entries per row

// MSD radix on the top K_TOP_BITS of the composite key.
static const int K_TOP_BITS = 16;
static const int N_BUCKETS  = 1 << K_TOP_BITS;         // 65536 = CNUM * CNUM

// Slab-kernel target size (bitonic-sortable in a real SIMD port).
static const size_t K_SLAB_TARGET   = 256;
static const size_t K_SLAB_BUF_SIZE = 2 * K_SLAB_TARGET;    // static slab buffer
static const size_t K_MAX_SLABS     = 4096;                 // pre-histogram cap

// All compile-time-sized tables live as static globals — no per-call malloc.
// The fast driver runs at most once per process, so these are not reset between
// calls; the only resets are explicit memset() / overwrite at the use sites.
//
// g_cum2 is a dense order-2 cumulative-frequency table: one CNUM1-entry row
// per (q,p) context pair, indexed directly by `ctx * CNUM1 + c`. This costs
// ~64 MB of BSS but lets cum_lookup be one array access with no fallback
// branch. Unobserved (q,p) contexts get a uniform row from normalize_row
// (its zero-count floor produces equal intervals for every symbol).
static uint32_t g_cum2     [N_BUCKETS * CNUM1];        // dense order-2 cum
static uint32_t g_bbase    [N_BUCKETS + 1];            // distribution prefix sum
static uint32_t g_cursor   [N_BUCKETS];                // scatter cursors

// ---- stats counters -----------------------------------------------------

// Per-run counters reset by stats_reset(). cmp_cyclic / memcmp_calls
// track both the fast and the reference paths since both now share
// cmp_cyclic on a single input buffer.
struct Stats {
    size_t cmp_cyclic_calls;       // cmp_cyclic invocations
    size_t memcmp_calls;           // raw memcmp calls (1..3 per cmp_cyclic)
    size_t sort_and_emit_calls;    // each slab or whole bucket = 1 call
    size_t buckets_nonempty;       // distribution-pass output buckets with m>0
    size_t slab_passes;            // slab-partition kernel invocations
    size_t bucket_fallback;        // buckets sorted via malloc'd big buffer
    size_t code_iters;             // total build_code inner-loop iterations
};
static Stats g_stats;

static void stats_reset() { memset(&g_stats, 0, sizeof(g_stats)); }

// ---- helpers ------------------------------------------------------------

// Return the high 64 bits of the 96-bit product (a * b). Used in
// build_code to scale a 64-bit interval span by a 32-bit fixed-point
// cumulative-probability value (already pre-scaled to 2^32 = 1.0).
static inline uint64_t mul_u64_u32_hi(uint64_t a, uint32_t b) {
    return (uint64_t)(((__uint128_t)a * b) >> 32);
}

// Compare cyclic rotations of inp starting at a and b via chained memcmps
// on the single input buffer. WLOG lo = min(a,b), hi = max(a,b):
//
//   memcmp #1:  inp[lo..lo+m1)        vs inp[hi..n)        (m1 = n - hi)
//                 both sides linear
//   memcmp #2:  inp[lo+m1..n)         vs inp[0..hi-lo)     (length hi-lo)
//                 hi-side has wrapped to 0; lo-side still linear
//   memcmp #3:  inp[0..lo)            vs inp[hi-lo..hi)    (length lo)
//                 both sides have wrapped
//
// Each subsequent memcmp runs only when the previous returned 0.
static int cmp_cyclic(const uint8_t* inp, size_t n,
                      uint32_t a, uint32_t b) {
    g_stats.cmp_cyclic_calls++;
    if (a == b) return 0;
    int sign = 1;
    uint32_t lo = a, hi = b;
    if (lo > hi) { uint32_t t = lo; lo = hi; hi = t; sign = -1; }
    size_t m1 = n - hi;
    g_stats.memcmp_calls++;
    int r = memcmp(inp + lo, inp + hi, m1);
    if (r != 0) return sign * r;
    size_t m2 = hi - lo;
    g_stats.memcmp_calls++;
    r = memcmp(inp + lo + m1, inp, m2);
    if (r != 0) return sign * r;
    g_stats.memcmp_calls++;
    r = memcmp(inp, inp + m2, lo);
    return sign * r;
}

// ---- I/O ----------------------------------------------------------------

// Read the whole file at `path` into a freshly-malloced buffer. On success
// stores the buffer in *buf and returns its byte length; on failure (open,
// size, read) prints to stderr and returns 0 with *buf left NULL.
static size_t read_file(const char* path, uint8_t** buf) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return 0; }
    *buf = (uint8_t*)malloc((size_t)sz);
    if (!*buf) { fclose(f); return 0; }
    if (fread(*buf, 1, (size_t)sz, f) != (size_t)sz) {
        perror("fread"); fclose(f); free(*buf); *buf = NULL; return 0;
    }
    fclose(f);
    return (size_t)sz;
}

// Write the BWT container at `path`: n bytes of BWT body, then a 4-byte
// little-endian primary index pi (same layout the rcbwt streaming path
// produces, and the same layout the standalone BWT.cpp / unBWT.cpp pair
// uses). Returns 0 on success, -1 on any I/O error.
static int write_file(const char* path, uint32_t pi,
                      const uint8_t* bwt, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    if (n && fwrite(bwt, 1, n, f) != n) {
        perror("fwrite"); fclose(f); return -1;
    }
    uint8_t tail[4];
    tail[0] = (uint8_t)(pi & 0xff);
    tail[1] = (uint8_t)((pi >> 8) & 0xff);
    tail[2] = (uint8_t)((pi >> 16) & 0xff);
    tail[3] = (uint8_t)((pi >> 24) & 0xff);
    if (fwrite(tail, 1, 4, f) != 4) { perror("fwrite"); fclose(f); return -1; }
    fclose(f);
    return 0;
}

// Read a BWT container produced by write_file / rcbwt: n bytes BWT body
// followed by a 4-byte LE pi. Caller supplies n; *bwt is malloc'd here.
// Returns 0 on success.
static int read_bwt_file(const char* path, size_t n,
                         uint8_t** bwt, uint32_t* pi) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    *bwt = (uint8_t*)malloc(n);
    if (n && fread(*bwt, 1, n, f) != n) {
        perror("fread bwt"); fclose(f); free(*bwt); *bwt = NULL; return -1;
    }
    uint8_t tail[4];
    if (fread(tail, 1, 4, f) != 4) {
        perror("fread pi"); fclose(f); free(*bwt); *bwt = NULL; return -1;
    }
    *pi = (uint32_t)tail[0]
        | ((uint32_t)tail[1] <<  8)
        | ((uint32_t)tail[2] << 16)
        | ((uint32_t)tail[3] << 24);
    fclose(f);
    return 0;
}

// ---- order-2 model -----------------------------------------------------

// Cumulative table convention: cum[0..CNUM] stored in CNUM1 uint32 entries.
// cum[0] = 0; cum[c+1] = sum of normalised counts up to c (out of 2^32);
// cum[CNUM] = 2^32 stored as 0 (32-bit wrap). r = cum[c+1] - cum[c] is then
// computed modularly and gives the right value even when c = CNUM-1.

// Convert one CNUM-entry row from raw counts to a cumulative-probability
// table (in-place, written to row[0..CNUM]). Zero counts get a 1-symbol
// floor so every symbol has a non-zero interval; the resulting row[c+1]
// values are scaled to the 2^32 fixed-point grid, with row[CNUM] = 2^32
// stored as 0 (wraps in uint32). An all-zero input row therefore yields a
// uniform distribution — that is the implicit fallback for (q,p) contexts
// that were never observed during the model build.
static void normalize_row(uint32_t* row) {
    uint32_t flo[CNUM];
    uint64_t total = 0;
    for (int c = 0; c < CNUM; c++) {
        flo[c] = row[c] ? row[c] : 1;
        total += flo[c];
    }
    const uint64_t Z = 1ULL << 32;
    uint64_t acc = 0;
    for (int c = 0; c < CNUM; c++) {
        acc += flo[c];
        row[c + 1] = (uint32_t)((acc * Z) / total);
    }
    row[0] = 0;
}

// Build the dense order-2 model into g_cum2. Every (q,p) row is normalised
// — observed contexts use their actual symbol counts, while never-observed
// contexts (all-zero counts) fall through to the floor-of-1 path and end
// up uniform. Counting uses linear positions only (i >= 2); the cyclic
// wrap that build_code does at the buffer boundary visits two un-counted
// triples per cycle, and those simply pick up the uniform-row default.
static void build_models(const uint8_t* inp, size_t n) {
    memset(g_cum2, 0, sizeof(g_cum2));
    for (size_t i = 2; i < n; i++) {
        uint32_t ctx = ((uint32_t)inp[i - 2] << 8) | inp[i - 1];
        g_cum2[ctx * CNUM1 + inp[i]]++;
    }
    for (int ctx = 0; ctx < N_BUCKETS; ctx++) {
        normalize_row(&g_cum2[ctx * CNUM1]);
    }
}

// One direct array access — no fallback branch, no offset indirection.
// c may be CNUM, in which case row[CNUM] = 0 (the wrapped 2^32) lets the
// caller compute (hi - lo) modularly to get the correct symbol range.
static inline uint32_t cum_lookup(uint8_t q, uint8_t p, uint32_t c) {
    return g_cum2[(((uint32_t)q << 8) | p) * CNUM1 + c];
}

// Bit layout of the 64-bit composite sort key.
static const int K_PREFIX_BITS = 16;                          // 2 raw symbols
static const int K_CODE_BITS   = 64 - K_PREFIX_BITS;          // 48

// Safety cap for the refining pass — in practice convergence happens
// inside the first call (early-exit at depth ~6 below n-1).
static const int K_MAX_REFINE_PASSES = 16;

// Walk j = start_j..0 once, applying the prepend recurrence
//
//   code[j] = mul_hi(~0, lo) + mul_hi(keys[(j+1)%n] << 16, r)
//
// and re-stamping the raw prefix bytes into the top 16 bits of keys[j].
// When TRACK is true, the function exits the moment a recomputed key
// matches the stored one: the recurrence is deterministic in keys[jp1],
// so once keys[j] is unchanged from the previous pass, every key for
// j' < j will be unchanged too. Returns true on early exit (converged),
// false if the walk ran to completion.
template <bool TRACK>
static bool refine_pass(const uint8_t* inp, size_t n,
                        uint64_t* keys, size_t start_j) {
    for (size_t j_inv = 0; j_inv <= start_j; j_inv++) {
        size_t j   = start_j - j_inv;
        size_t jp1 = (j   + 1 < n) ? j   + 1 : 0;
        size_t jp2 = (jp1 + 1 < n) ? jp1 + 1 : 0;

        uint8_t q = inp[j];
        uint8_t p = inp[jp1];
        uint8_t c = inp[jp2];
        uint32_t lo = cum_lookup(q, p, c);
        uint32_t hi = cum_lookup(q, p, (uint32_t)c + 1);
        uint32_t r  = hi - lo;

        // keys[jp1] << K_PREFIX_BITS shifts the raw prefix out the top
        // and zero-fills the bottom 16 bits, giving the 48-bit code as
        // the high 48 bits of a 64-bit value.
        uint64_t tail = keys[jp1] << K_PREFIX_BITS;
        uint64_t code = mul_u64_u32_hi(~0ULL, lo)
                      + mul_u64_u32_hi(tail, r);
        g_stats.code_iters++;

        uint64_t new_key = ((uint64_t)q << (K_CODE_BITS + 8))
                         | ((uint64_t)p <<  K_CODE_BITS)
                         | (code >> K_PREFIX_BITS);

        if constexpr (TRACK) {
            if (keys[j] == new_key) return true;
        }
        keys[j] = new_key;
    }
    return false;
}

// Build the n composite 64-bit sort keys without ever calling
// build_code or zero-initializing the whole keys[] array.
//
//   - Seed: a single 64-bit write to keys[n-1]. Pass 1 reads it via
//     `keys[n-1] << K_PREFIX_BITS`, so any value whose low 48 bits are
//     zero gives a clean zero tail; we just store 0.
//   - Pass 1 (TRACK=false): write every key once, in strict order
//     j = n-2..0. No cyclic wrap on j+1, no comparisons.
//   - Pass 2 (TRACK=true) and beyond: walk j = n-1..0, exit on the
//     first key that already matches. Saturation past the precision
//     floor (~6 prepends) makes pass 2 stop near j = n-7.
static void build_keys(const uint8_t* inp, size_t n, uint64_t* keys) {
    if (n == 0) return;
    if (n == 1) {
        uint8_t b = inp[0];
        keys[0] = ((uint64_t)b << (K_CODE_BITS + 8))
                | ((uint64_t)b <<  K_CODE_BITS);
        return;
    }

    keys[n - 1] = 0;
    refine_pass<false>(inp, n, keys, n - 2);
    for (int pass = 0; pass < K_MAX_REFINE_PASSES; pass++) {
        if (refine_pass<true>(inp, n, keys, n - 1)) break;
    }
}

// ---- distribution pass (top K_TOP_BITS bits) ----------------------------

// One-pass MSD radix on the top K_TOP_BITS of each key. Builds the
// per-bucket prefix-sum g_bbase[N_BUCKETS+1] and scatters (key, pos)
// pairs into keys_perm[]/pos_perm[] so bucket b occupies the range
// [g_bbase[b], g_bbase[b+1]).
static void distribute(const uint64_t* keys, size_t n,
                       uint64_t* keys_perm, uint32_t* pos_perm) {
    memset(g_bbase, 0, sizeof(g_bbase));
    for (size_t i = 0; i < n; i++) {
        uint32_t b = (uint32_t)(keys[i] >> K_CODE_BITS);
        g_bbase[b + 1]++;
    }
    for (int b = 1; b <= N_BUCKETS; b++) {
        g_bbase[b] += g_bbase[b - 1];
    }
    memcpy(g_cursor, g_bbase, sizeof(g_cursor));
    for (size_t i = 0; i < n; i++) {
        uint32_t b = (uint32_t)(keys[i] >> K_CODE_BITS);
        uint32_t dst = g_cursor[b]++;
        keys_perm[dst] = keys[i];
        pos_perm[dst] = (uint32_t)i;
    }
}

// ---- slab sort + tie resolution + emission ------------------------------

// One (key, position) pair under sort. `p` is the input position whose
// rotation the key represents.
struct KP { uint64_t k; uint32_t p; };

// Static slab and pre-histogram tables (no per-call malloc).
static KP       g_slab     [K_SLAB_BUF_SIZE];
static uint32_t g_slab_hist[K_MAX_SLABS];

// Sort one slab (or whole bucket) of (key, pos) pairs and stream the
// corresponding BWT bytes directly to `fp` (no in-memory output buffer).
// The comparator chains key < cmp_cyclic < position so equal keys resolve
// via full cyclic suffix comparison, with any remaining true-tie group
// broken by ascending position (same ordering the dumb path uses).
// *out_idx tracks bytes written so we can record pi when pos==0 lands.
static void sort_and_emit(KP* kp, size_t cnt,
                          const uint8_t* inp, size_t n,
                          FILE* fp, size_t* out_idx, uint32_t* pi) {
    g_stats.sort_and_emit_calls++;
    std::sort(kp, kp + cnt, [inp, n](const KP& a, const KP& b) {
        if (a.k != b.k) return a.k < b.k;
        int c = cmp_cyclic(inp, n, a.p, b.p);
        if (c != 0) return c < 0;
        return a.p < b.p;
    });
    for (size_t i = 0; i < cnt; i++) {
        uint32_t pos = kp[i].p;
        uint8_t  ch  = (pos == 0) ? inp[n - 1] : inp[pos - 1];
        fputc(ch, fp);
        if (pos == 0) *pi = (uint32_t)(*out_idx);
        (*out_idx)++;
    }
}

// Sort and emit one distribution-pass bucket of m elements directly to
// `fp`. Three paths:
//   small bucket  (m <= K_SLAB_BUF_SIZE)  — fits in g_slab, sort & emit
//   dense bucket  (slab partition fits)   — pre-histogram per-slab
//                                           sizes, then stream-compact
//                                           each slab into g_slab
//   overflow      (any slab > buffer, or
//                  S > K_MAX_SLABS)        — malloc once for the whole
//                                           bucket and sort it as one
//                                           block (fall-back path)
static void process_bucket(const uint64_t* bk, const uint32_t* bp, size_t m,
                           FILE* fp, size_t* out_idx, uint32_t* pi,
                           const uint8_t* inp, size_t n) {
    if (m == 0) return;

    // ---- small bucket: fits in static buffer --------------------------
    if (m <= K_SLAB_BUF_SIZE) {
        for (size_t i = 0; i < m; i++) { g_slab[i].k = bk[i]; g_slab[i].p = bp[i]; }
        sort_and_emit(g_slab, m, inp, n, fp, out_idx, pi);
        return;
    }

    const uint64_t K_CODE_RANGE = 1ULL << K_CODE_BITS;
    const uint64_t K_CODE_MASK  = K_CODE_RANGE - 1;
    size_t S = (m + K_SLAB_TARGET - 1) / K_SLAB_TARGET;
    uint64_t w = K_CODE_RANGE / S;
    if (w == 0) w = 1;

    // ---- pre-histogram or whole-bucket fall-back ----------------------
    bool fallback = (S > K_MAX_SLABS);
    if (!fallback) {
        memset(g_slab_hist, 0, S * sizeof(uint32_t));
        for (size_t i = 0; i < m; i++) {
            uint64_t k_low = bk[i] & K_CODE_MASK;
            size_t s = (size_t)(k_low / w);
            if (s >= S) s = S - 1;
            g_slab_hist[s]++;
        }
        for (size_t s = 0; s < S; s++) {
            if (g_slab_hist[s] > K_SLAB_BUF_SIZE) { fallback = true; break; }
        }
    }
    if (fallback) {
        g_stats.bucket_fallback++;
        KP* big = (KP*)malloc(m * sizeof(KP));
        for (size_t i = 0; i < m; i++) { big[i].k = bk[i]; big[i].p = bp[i]; }
        sort_and_emit(big, m, inp, n, fp, out_idx, pi);
        free(big);
        return;
    }

    // ---- slab partition with static buffer ----------------------------
    for (size_t s = 0; s < S; s++) {
        g_stats.slab_passes++;
        uint64_t slab_lo48 = s * w;
        uint64_t slab_hi48 = (s == S - 1) ? K_CODE_RANGE : (s + 1) * w;
        size_t cnt = 0;
        for (size_t i = 0; i < m; i++) {
            uint64_t k     = bk[i];
            uint32_t p     = bp[i];
            uint64_t k_low = k & K_CODE_MASK;
            int take = (int)((k_low >= slab_lo48) & (k_low < slab_hi48));
            g_slab[cnt].k = k;
            g_slab[cnt].p = p;
            cnt += (size_t)take;
        }
        sort_and_emit(g_slab, cnt, inp, n, fp, out_idx, pi);
    }
}

// ---- fast driver --------------------------------------------------------

// Append a 4-byte little-endian uint32 to fp.
static int write_u32_le(FILE* fp, uint32_t v) {
    uint8_t b[4] = {
        (uint8_t)(v & 0xff), (uint8_t)((v >> 8) & 0xff),
        (uint8_t)((v >> 16) & 0xff), (uint8_t)((v >> 24) & 0xff)
    };
    return (fwrite(b, 1, 4, fp) == 4) ? 0 : -1;
}

// End-to-end fast path: build models, build composite keys, radix-
// distribute into N_BUCKETS buckets, then walk buckets in ascending
// order, sort each (slab-partitioning dense ones), and stream the BWT
// bytes directly to `fp`. After all n bytes are emitted, a 4-byte LE
// primary index is appended — final file layout is [n BWT bytes][pi].
static void rcbwt(const uint8_t* inp, size_t n, FILE* fp) {
    if (n == 0) { write_u32_le(fp, 0); return; }
    if (n == 1) {
        fputc(inp[0], fp);
        write_u32_le(fp, 0);
        return;
    }

    build_models(inp, n);

    uint64_t* keys = (uint64_t*)malloc(n * sizeof(uint64_t));
    build_keys(inp, n, keys);

    uint64_t* keys_perm = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint32_t* pos_perm  = (uint32_t*)malloc(n * sizeof(uint32_t));
    distribute(keys, n, keys_perm, pos_perm);
    free(keys);

    size_t out_idx = 0;
    uint32_t pi    = 0;
    for (int b = 0; b < N_BUCKETS; b++) {
        uint32_t lo = g_bbase[b];
        uint32_t hi = g_bbase[b + 1];
        size_t   m  = hi - lo;
        if (m == 0) continue;
        g_stats.buckets_nonempty++;
        process_bucket(keys_perm + lo, pos_perm + lo, m,
                       fp, &out_idx, &pi, inp, n);
    }
    write_u32_le(fp, pi);

    free(keys_perm);
    free(pos_perm);
}

// ---- reference (std::sort on cyclic rotations) --------------------------

// Brute-force BWT used purely as a test oracle: std::sort on rotation
// indices with cmp_cyclic as the comparator (no doubled buffer — the
// three chained memcmps inside cmp_cyclic handle wrap on the single
// input buffer). Ties on truly-equal cyclic rotations break by ascending
// sa[i] to match the fast path.
static void dumb_bwt(const uint8_t* inp, size_t n,
                     uint8_t* out, uint32_t* pi) {
    *pi = 0;
    if (n == 0) return;
    uint32_t* sa = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (size_t i = 0; i < n; i++) sa[i] = (uint32_t)i;

    std::sort(sa, sa + n, [inp, n](uint32_t a, uint32_t b) {
        int c = cmp_cyclic(inp, n, a, b);
        if (c != 0) return c < 0;
        return a < b;
    });

    for (size_t i = 0; i < n; i++) {
        uint32_t pos = sa[i];
        out[i] = (pos == 0) ? inp[n - 1] : inp[pos - 1];
        if (pos == 0) *pi = (uint32_t)i;
    }
    free(sa);
}

// ---- inverse BWT (LF mapping) ------------------------------------------
// Reconstructs the original input from (L, pi). Returns 0 on success.
// Independent of the forward path; used to cross-check the dumb BWT.
static int inverse_bwt(const uint8_t* L, size_t n, uint32_t pi, uint8_t* out) {
    if (n == 0) return 0;
    if (pi >= n) return -1;

    // C[c] = number of bytes in L strictly less than c.
    size_t C[CNUM1];
    memset(C, 0, sizeof(C));
    for (size_t i = 0; i < n; i++) C[L[i] + 1]++;
    for (int c = 1; c <= CNUM; c++) C[c] += C[c - 1];

    // rank[i] = number of occurrences of L[i] in L[0..i-1].
    uint32_t* rank = (uint32_t*)malloc(n * sizeof(uint32_t));
    uint32_t cnt[CNUM];
    memset(cnt, 0, sizeof(cnt));
    for (size_t i = 0; i < n; i++) {
        rank[i] = cnt[L[i]]++;
    }

    // LF mapping: row LF[i] is the row whose last char precedes L[i] in T.
    // Walk from pi (the row of rotation 0) for n steps; this emits T in reverse.
    size_t cur = pi;
    for (size_t k = 0; k < n; k++) {
        uint8_t ch = L[cur];
        out[n - 1 - k] = ch;
        cur = C[ch] + rank[cur];
    }
    free(rank);
    return 0;
}

// ---- compare & report ---------------------------------------------------

// Compare two BWT outputs (a, b) plus their primary indices and print
// either "OK" or the first byte-level mismatch with a 32-byte hex
// surround. Returns 0 if identical, 1 otherwise.
static int compare_and_report(const uint8_t* a, const uint8_t* b, size_t n,
                              uint32_t pi_a, uint32_t pi_b) {
    int bad = 0;
    if (pi_a != pi_b) {
        printf("pi mismatch: fast=%u dumb=%u\n", pi_a, pi_b);
        bad = 1;
    }
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("BWT mismatch at offset %zu: fast=%02x dumb=%02x\n",
                   i, a[i], b[i]);
            size_t lo = (i >= 16) ? i - 16 : 0;
            size_t hi = (i + 16 < n) ? i + 16 : n;
            printf("  fast:");
            for (size_t k = lo; k < hi; k++) printf(" %02x", a[k]);
            printf("\n  dumb:");
            for (size_t k = lo; k < hi; k++) printf(" %02x", b[k]);
            printf("\n");
            bad = 1;
            break;
        }
    }
    if (!bad) printf("OK\n");
    return bad;
}

// ---- stats reporting ----------------------------------------------------

// Print the current algorithm counters under a section heading. Counters
// are not reset here; callers do that between phases via stats_reset().
static void print_algo_stats(const char* tag) {
    printf("[%s] algorithm:\n", tag);
    printf("  cmp_cyclic calls     : %zu\n", g_stats.cmp_cyclic_calls);
    printf("  memcmp calls (total) : %zu  (avg %.2f per cmp_cyclic)\n",
           g_stats.memcmp_calls,
           g_stats.cmp_cyclic_calls
               ? (double)g_stats.memcmp_calls / g_stats.cmp_cyclic_calls
               : 0.0);
    printf("  sort_and_emit calls  : %zu\n", g_stats.sort_and_emit_calls);
    printf("  non-empty buckets    : %zu\n", g_stats.buckets_nonempty);
    printf("  slab partition passes: %zu\n", g_stats.slab_passes);
    printf("  bucket-overflow falls: %zu\n", g_stats.bucket_fallback);
    printf("  build_code iterations: %zu", g_stats.code_iters);
    if (g_stats.code_iters) printf("  (key depth metric)");
    printf("\n");
}

// Sum and report dynamic and static memory in units of the input size n.
// The dense g_cum2 dominates the fixed-globals total (~64 MB BSS).
static void print_memory_stats(size_t n,
                               size_t fast_bwt_readback_bytes,
                               int with_dumb) {
    auto pct = [n](size_t bytes) -> double {
        return n ? (double)bytes / n : 0.0;
    };
    size_t fixed = sizeof(g_cum2) + sizeof(g_bbase) + sizeof(g_cursor)
                 + sizeof(g_slab) + sizeof(g_slab_hist);

    printf("Memory for n=%zu bytes:\n", n);
    printf("  inp           = %5.2fn  (%zu B)\n", pct(n),       n);
    printf("  keys          = %5.2fn  (%zu B)\n", pct(8 * n),   8 * n);
    printf("  keys_perm     = %5.2fn  (%zu B)\n", pct(8 * n),   8 * n);
    printf("  pos_perm      = %5.2fn  (%zu B)\n", pct(4 * n),   4 * n);
    if (with_dumb) {
        printf("  bwt_fast read = %5.2fn  (%zu B, verification only)\n",
               pct(fast_bwt_readback_bytes), fast_bwt_readback_bytes);
        printf("  bwt_ref       = %5.2fn  (%zu B)\n", pct(n),     n);
        printf("  sa (dumb)     = %5.2fn  (%zu B)\n", pct(4 * n), 4 * n);
        printf("  chk (inverse) = %5.2fn  (%zu B)\n", pct(n),     n);
        printf("  rank (inverse)= %5.2fn  (%zu B)\n", pct(4 * n), 4 * n);
    }
    printf("  static globals= %5.2fn  (%zu B fixed: cum2+bbase+cursor+slab+slab_hist)\n",
           pct(fixed), fixed);

    size_t dyn_fast = n + 8*n + 8*n + 4*n;
    size_t dyn_dumb = with_dumb
        ? (fast_bwt_readback_bytes + n + 4*n + n + 4*n)
        : 0;
    size_t total    = dyn_fast + dyn_dumb + fixed;
    printf("  total dynamic = %5.2fn  (fast %zu B + dumb %zu B)\n",
           pct(dyn_fast + dyn_dumb), dyn_fast, dyn_dumb);
    printf("  total all     = %5.2fn  (%zu B)\n", pct(total), total);
}

// ---- main ---------------------------------------------------------------

// Parse args, run the fast BWT (always) and the dumb BWT (when a third
// argument is given), write both containers, compare them, and round-
// trip each through inverse_bwt as a final sanity check. Returns 0 only
// when every check passed.
int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s inputfile outputfile [dumb-BWT-file]\n",
                argv[0]);
        return 1;
    }

    uint8_t* inp = NULL;
    size_t n = read_file(argv[1], &inp);
    if (n == 0) { fprintf(stderr, "empty input\n"); return 1; }

    // ---- fast path: stream directly to argv[2] ------------------------
    FILE* fp_fast = fopen(argv[2], "wb");
    if (!fp_fast) { perror(argv[2]); return 1; }
    stats_reset();
    rcbwt(inp, n, fp_fast);
    fclose(fp_fast);
    print_algo_stats("rcbwt");

    int rc = 0;
    if (argc == 4) {
        // ---- reference path in memory, then file --------------------
        uint8_t* bwt_ref = (uint8_t*)malloc(n);
        uint32_t pi_ref = 0;
        stats_reset();
        dumb_bwt(inp, n, bwt_ref, &pi_ref);
        print_algo_stats("dumb_bwt");
        if (write_file(argv[3], pi_ref, bwt_ref, n) != 0) return 1;

        // Read fast output back from disk for compare + inverse.
        uint8_t*  bwt_fast = NULL;
        uint32_t  pi_fast  = 0;
        if (read_bwt_file(argv[2], n, &bwt_fast, &pi_fast) != 0) return 1;

        rc = compare_and_report(bwt_fast, bwt_ref, n, pi_fast, pi_ref);

        // Cross-check both BWTs by inverting and matching the input.
        uint8_t* chk = (uint8_t*)malloc(n);
        inverse_bwt(bwt_ref, n, pi_ref, chk);
        if (memcmp(chk, inp, n) != 0) {
            printf("dumb_bwt inverse: MISMATCH\n"); rc = 1;
        } else {
            printf("dumb_bwt inverse: ok\n");
        }
        inverse_bwt(bwt_fast, n, pi_fast, chk);
        if (memcmp(chk, inp, n) != 0) {
            printf("rcbwt inverse: MISMATCH\n"); rc = 1;
        } else {
            printf("rcbwt inverse: ok\n");
        }
        free(chk);
        free(bwt_fast);
        free(bwt_ref);
    }

    print_memory_stats(n, argc == 4 ? n : 0, argc == 4);

    free(inp);
    return rc;
}
