// rcbwt: BWT via entropy-coded suffix keys (see RC_BWT.md, plan.md).
// Usage: rcbwt inputfile outputfile [dumb-BWT-file]
//
// Output format (both fast and reference paths):
//   [ 4 bytes LE primary index pi ][ n bytes BWT ]
//
// The fast path implements the streaming radix on a 64-bit composite key:
// 16 raw context bits + 48 bits of an order-2 arithmetic-coded tail.
// The reference (dumb) path uses std::sort on cyclic rotations.

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
static const size_t K_SLAB_TARGET = 256;

// All compile-time-sized tables live as static globals — no per-call malloc.
// The fast driver runs at most once per process, so these are not reset between
// calls; the only resets are explicit memset() / overwrite at the use sites.
static uint32_t g_cum1     [CNUM * CNUM1];             // order-1 cum table
static uint32_t g_ctx2_off [N_BUCKETS];                // order-2 row offsets
static uint32_t g_bbase    [N_BUCKETS + 1];            // distribution prefix sum
static uint32_t g_cursor   [N_BUCKETS];                // scatter cursors
static uint32_t g_cnt1     [CNUM * CNUM];              // raw order-1 counts
static uint8_t  g_ctx_seen [N_BUCKETS];                // non-empty (q,p) flag

// ---- helpers ------------------------------------------------------------

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
    if (a == b) return 0;
    int sign = 1;
    uint32_t lo = a, hi = b;
    if (lo > hi) { uint32_t t = lo; lo = hi; hi = t; sign = -1; }
    size_t m1 = n - hi;
    int r = memcmp(inp + lo, inp + hi, m1);
    if (r != 0) return sign * r;
    size_t m2 = hi - lo;
    r = memcmp(inp + lo + m1, inp, m2);
    if (r != 0) return sign * r;
    r = memcmp(inp, inp + m2, lo);
    return sign * r;
}

// ---- I/O ----------------------------------------------------------------

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

static int write_file(const char* path, uint32_t pi,
                      const uint8_t* bwt, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    uint8_t hdr[4];
    hdr[0] = (uint8_t)(pi & 0xff);
    hdr[1] = (uint8_t)((pi >> 8) & 0xff);
    hdr[2] = (uint8_t)((pi >> 16) & 0xff);
    hdr[3] = (uint8_t)((pi >> 24) & 0xff);
    if (fwrite(hdr, 1, 4, f) != 4) { perror("fwrite"); fclose(f); return -1; }
    if (n && fwrite(bwt, 1, n, f) != n) {
        perror("fwrite"); fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

// ---- order-1 / order-2 model -------------------------------------------

// Cumulative table convention: cum[0..CNUM] stored in CNUM1 uint32 entries.
// cum[0] = 0; cum[c+1] = sum of normalised counts up to c (out of 2^32);
// cum[CNUM] = 2^32 stored as 0 (32-bit wrap). r = cum[c+1] - cum[c] is then
// computed modularly and gives the right value even when c = CNUM-1.

// Order-2 row pool — only n-dependent allocation in the model. Each
// non-empty (q,p) context owns CNUM1 entries; offset 0 is a reserved
// sentinel meaning "context never observed, fall back to cum1".
static uint32_t* g_pool      = NULL;
static size_t    g_pool_cap  = 0;

static void normalize_row(uint32_t* row) {
    // row[0..CNUM-1] currently holds raw counts. Replace with cumulative
    // form (row[0]=0, row[c+1] = normalised cum, row[CNUM] wraps to 0).
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

static void build_models(const uint8_t* inp, size_t n) {
    // ---- order-1 model ----
    memset(g_cnt1, 0, sizeof(g_cnt1));
    for (size_t i = 1; i < n; i++) {
        g_cnt1[(size_t)inp[i - 1] * CNUM + inp[i]]++;
    }
    for (int p = 0; p < CNUM; p++) {
        for (int c = 0; c < CNUM; c++) {
            g_cum1[p * CNUM1 + c] = g_cnt1[p * CNUM + c];
        }
        g_cum1[p * CNUM1 + CNUM] = 0;
        normalize_row(&g_cum1[p * CNUM1]);
    }

    // ---- order-2 model ----
    memset(g_ctx2_off, 0, sizeof(g_ctx2_off));
    memset(g_ctx_seen, 0, sizeof(g_ctx_seen));

    size_t n_ctx = 0;
    for (size_t i = 2; i < n; i++) {
        uint32_t ctx = ((uint32_t)inp[i - 2] << 8) | inp[i - 1];
        if (!g_ctx_seen[ctx]) { g_ctx_seen[ctx] = 1; n_ctx++; }
    }

    g_pool_cap = (n_ctx + 1) * CNUM1;       // offset 0 reserved as sentinel
    free(g_pool);
    g_pool = (uint32_t*)calloc(g_pool_cap, sizeof(uint32_t));

    size_t next_off = CNUM1;
    for (int ctx = 0; ctx < N_BUCKETS; ctx++) {
        if (g_ctx_seen[ctx]) {
            g_ctx2_off[ctx] = (uint32_t)next_off;
            next_off += CNUM1;
        }
    }

    // Accumulate raw counts into the assigned row slots.
    for (size_t i = 2; i < n; i++) {
        uint32_t ctx = ((uint32_t)inp[i - 2] << 8) | inp[i - 1];
        uint32_t off = g_ctx2_off[ctx];
        g_pool[off + inp[i]]++;
    }

    // Normalise each populated row.
    for (int ctx = 0; ctx < N_BUCKETS; ctx++) {
        uint32_t off = g_ctx2_off[ctx];
        if (!off) continue;
        normalize_row(&g_pool[off]);
    }
}

static inline uint32_t cum_lookup(uint8_t q, uint8_t p, uint32_t c) {
    uint32_t off = g_ctx2_off[((uint32_t)q << 8) | p];
    if (off) return g_pool[off + c];
    return g_cum1[(uint32_t)p * CNUM1 + c];
}

// Code-build precision floor: stop the arithmetic-code loop once span
// drops below this. Below this point the (span * r) >> 32 multiply
// underflows to near-zero and contributes no information.
static const uint64_t K_SPAN_FLOOR = 1ULL << 16;

// Bit layout of the 64-bit composite sort key.
static const int K_PREFIX_BITS = 16;                          // 2 raw symbols
static const int K_CODE_BITS   = 64 - K_PREFIX_BITS;          // 48

// Build a 64-bit arithmetic code from inp[start..], wrapping cyclically.
// Cyclic wrap makes the key a function of the entire rotation, which
// preserves order across positions of different effective "tail length".
static inline uint64_t build_code(const uint8_t* inp, size_t n, size_t start) {
    uint64_t code = 0;
    uint64_t span = ~0ULL;
    size_t k = start;
    while (span >= K_SPAN_FLOOR) {
        size_t  kk = (k     < n) ? k     : k - n;
        size_t  k1 = (kk    > 0) ? kk - 1 : n - 1;
        size_t  k2 = (k1    > 0) ? k1 - 1 : n - 1;
        uint8_t c = inp[kk];
        uint8_t p = inp[k1];
        uint8_t q = inp[k2];
        uint32_t lo = cum_lookup(q, p, c);
        uint32_t hi = cum_lookup(q, p, (uint32_t)c + 1);
        uint32_t r  = hi - lo;             // modular: r = 2^32 when c=CNUM-1
        code += mul_u64_u32_hi(span, lo);
        span  = mul_u64_u32_hi(span, r);
        k++;
        if (k >= 2 * n) k -= n;            // keep k bounded
    }
    return code;
}

static void build_keys(const uint8_t* inp, size_t n, uint64_t* keys) {
    for (size_t j = 0; j < n; j++) {
        uint64_t code = build_code(inp, n, j + 2);
        uint8_t  b0 = inp[j];
        uint8_t  b1 = inp[(j + 1) % n];
        keys[j] = ((uint64_t)b0 << (K_CODE_BITS + 8))
                | ((uint64_t)b1 <<  K_CODE_BITS)
                | (code >> K_PREFIX_BITS);
    }
}

// ---- distribution pass (top K_TOP_BITS bits) ----------------------------

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

namespace {
struct KP { uint64_t k; uint32_t p; };
}

static void sort_and_emit(KP* kp, size_t cnt,
                          const uint8_t* inp, size_t n,
                          uint8_t* out, size_t* out_idx, uint32_t* pi) {
    // Sort by (key, then cyclic suffix). The cyclic compare is the safety
    // net for tied keys (truncation-induced or end-of-buffer short keys).
    std::sort(kp, kp + cnt, [inp, n](const KP& a, const KP& b) {
        if (a.k != b.k) return a.k < b.k;
        int c = cmp_cyclic(inp, n, a.p, b.p);
        if (c != 0) return c < 0;
        return a.p < b.p;
    });
    for (size_t i = 0; i < cnt; i++) {
        uint32_t pos = kp[i].p;
        uint8_t  ch  = (pos == 0) ? inp[n - 1] : inp[pos - 1];
        out[*out_idx] = ch;
        if (pos == 0) *pi = (uint32_t)(*out_idx);
        (*out_idx)++;
    }
}

static void process_bucket(const uint64_t* bk, const uint32_t* bp, size_t m,
                           uint8_t* out, size_t* out_idx, uint32_t* pi,
                           const uint8_t* inp, size_t n) {
    if (m == 0) return;
    if (m <= K_SLAB_TARGET) {
        KP* kp = (KP*)malloc(m * sizeof(KP));
        for (size_t i = 0; i < m; i++) { kp[i].k = bk[i]; kp[i].p = bp[i]; }
        sort_and_emit(kp, m, inp, n, out, out_idx, pi);
        free(kp);
        return;
    }
    // ---- slab partition over the K_CODE_BITS code subspace ------------
    const uint64_t K_CODE_RANGE = 1ULL << K_CODE_BITS;
    const uint64_t K_CODE_MASK  = K_CODE_RANGE - 1;
    size_t S = (m + K_SLAB_TARGET - 1) / K_SLAB_TARGET;
    uint64_t w = K_CODE_RANGE / S;
    if (w == 0) w = 1;

    size_t cap = 2 * K_SLAB_TARGET + 64;
    if (cap < m + 64) cap = m + 64;
    KP* slab = (KP*)malloc(cap * sizeof(KP));

    for (size_t s = 0; s < S; s++) {
        uint64_t slab_lo48 = s * w;
        uint64_t slab_hi48 = (s == S - 1) ? K_CODE_RANGE : (s + 1) * w;
        size_t cnt = 0;
        for (size_t i = 0; i < m; i++) {
            uint64_t k     = bk[i];
            uint32_t p     = bp[i];
            uint64_t k_low = k & K_CODE_MASK;
            int take = (int)((k_low >= slab_lo48) & (k_low < slab_hi48));
            if (cnt + 1 > cap) {
                cap *= 2;
                slab = (KP*)realloc(slab, cap * sizeof(KP));
            }
            slab[cnt].k = k;
            slab[cnt].p = p;
            cnt += (size_t)take;
        }
        sort_and_emit(slab, cnt, inp, n, out, out_idx, pi);
    }
    free(slab);
}

// ---- fast driver --------------------------------------------------------

static void rcbwt(const uint8_t* inp, size_t n,
                  uint8_t* out, uint32_t* pi) {
    *pi = 0;
    if (n == 0) return;
    if (n == 1) {
        out[0] = inp[0];
        *pi = 0;
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
    for (int b = 0; b < N_BUCKETS; b++) {
        uint32_t lo = g_bbase[b];
        uint32_t hi = g_bbase[b + 1];
        size_t   m  = hi - lo;
        if (m == 0) continue;
        process_bucket(keys_perm + lo, pos_perm + lo, m,
                       out, &out_idx, pi, inp, n);
    }

    free(keys_perm);
    free(pos_perm);
    free(g_pool);
    g_pool = NULL;
}

// ---- reference (std::sort on cyclic rotations) --------------------------

static void dumb_bwt(const uint8_t* inp, size_t n,
                     uint8_t* out, uint32_t* pi) {
    *pi = 0;
    if (n == 0) return;
    uint32_t* sa = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (size_t i = 0; i < n; i++) sa[i] = (uint32_t)i;

    uint8_t* d = (uint8_t*)malloc(2 * n);
    memcpy(d,     inp, n);
    memcpy(d + n, inp, n);

    std::sort(sa, sa + n, [d, n](uint32_t a, uint32_t b) {
        int c = memcmp(d + a, d + b, n);
        if (c != 0) return c < 0;
        return a < b;
    });

    for (size_t i = 0; i < n; i++) {
        uint32_t pos = sa[i];
        out[i] = (pos == 0) ? inp[n - 1] : inp[pos - 1];
        if (pos == 0) *pi = (uint32_t)i;
    }
    free(d);
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

// ---- main ---------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s inputfile outputfile [dumb-BWT-file]\n",
                argv[0]);
        return 1;
    }

    uint8_t* inp = NULL;
    size_t n = read_file(argv[1], &inp);
    if (n == 0) { fprintf(stderr, "empty input\n"); return 1; }

    uint8_t* bwt_out = (uint8_t*)malloc(n);
    uint32_t pi = 0;
    rcbwt(inp, n, bwt_out, &pi);
    if (write_file(argv[2], pi, bwt_out, n) != 0) return 1;

    int rc = 0;
    if (argc == 4) {
        uint8_t* bwt_ref = (uint8_t*)malloc(n);
        uint32_t pi_ref = 0;
        dumb_bwt(inp, n, bwt_ref, &pi_ref);
        if (write_file(argv[3], pi_ref, bwt_ref, n) != 0) return 1;
        rc = compare_and_report(bwt_out, bwt_ref, n, pi, pi_ref);

        // Cross-check both BWTs by inverting them and matching the input.
        uint8_t* chk = (uint8_t*)malloc(n);
        inverse_bwt(bwt_ref, n, pi_ref, chk);
        if (memcmp(chk, inp, n) != 0) {
            printf("dumb_bwt inverse: MISMATCH\n"); rc = 1;
        } else {
            printf("dumb_bwt inverse: ok\n");
        }
        inverse_bwt(bwt_out, n, pi, chk);
        if (memcmp(chk, inp, n) != 0) {
            printf("rcbwt inverse: MISMATCH\n"); rc = 1;
        } else {
            printf("rcbwt inverse: ok\n");
        }
        free(chk);
        free(bwt_ref);
    }

    free(bwt_out);
    free(inp);
    return rc;
}
