// rcbwt: BWT-replacement demo (see plan.md, RC_BWT.md).
//
// Usage:  rcbwt inputfile outputfile [dumb-BWT-file]
//
// Output format (both fast and dumb paths):
//   [ 4 bytes LE primary index pi ][ n bytes BWT ]
//
// BWT formulation: cyclic rotations, no sentinel; pi is the index of the
// rotation starting at position 0 in the sorted order.
//
// The fast path is a streaming, branchless-friendly radix on a 64-bit
// order-preserving composite key:
//
//   bits 63..56  inp[j]                    (raw)
//   bits 55..48  inp[(j+1) % n]            (raw)
//   bits 47..0   order-2 forward code tail (48 bits, cyclic over the suffix)
//
// One MSD distribution pass on the top 16 bits, then per-bucket sort
// (slab scan for large buckets), with recursive key extension and
// cyclic memcmp + pos as the final tie-breaker.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- tunables ---------------------------------------------------------------

static const int    K_TOP_BITS    = 16;
static const size_t K_SLAB_TARGET = 256;
static const int    K_MAX_LEVELS  = 4;

// ---- helpers ----------------------------------------------------------------

static inline uint64_t mul_u64_u32_hi(uint64_t a, uint32_t b) {
    return (uint64_t)(((__uint128_t)a * (__uint128_t)b) >> 32);
}

// Compare two cyclic rotations starting at a and b, given a doubled buffer.
// Final tie-breaker: by position (so order is deterministic on periodic input).
// Returns <0, 0 (only if a==b), or >0.
static int cmp_cyclic2(const uint8_t* inp2, size_t n, uint32_t a, uint32_t b) {
    int r = memcmp(inp2 + a, inp2 + b, n);
    if (r != 0) return r;
    if (a < b) return -1;
    if (a > b) return  1;
    return 0;
}

// Branchless conditional swap on (key, pos) pairs (sort by key, then pos).
static inline void cswap_kp(uint64_t* k, uint32_t* p, size_t i, size_t j) {
    uint64_t ki = k[i], kj = k[j];
    uint32_t pi = p[i], pj = p[j];
    int swap = (ki > kj) | ((ki == kj) & (pi > pj));
    uint64_t mk = (uint64_t)0 - (uint64_t)swap;
    uint32_t mp = (uint32_t)0 - (uint32_t)swap;
    k[i] = ki ^ ((ki ^ kj) & mk);
    k[j] = kj ^ ((ki ^ kj) & mk);
    p[i] = pi ^ ((pi ^ pj) & mp);
    p[j] = pj ^ ((pi ^ pj) & mp);
}

// ---- model storage ----------------------------------------------------------

// cum1[p*257 + c] = cumulative-frequency at edge c (0..256) in order-1
// context p. c=0 -> 0; c=256 -> 2^32 (returned by helper, not stored).
// ctx2_off[q*256 + p] = offset (in uint32_t) into ctx2_pool, or 0xFFFFFFFFu
// if the (q,p) context was never seen.

struct CtxPool2 {
    uint32_t* off;   // 65536 entries
    uint32_t* pool;  // packed rows of 257
    size_t    pool_len;
};

static void free_ctx2(CtxPool2* c) {
    if (!c) return;
    free(c->off);  c->off  = nullptr;
    free(c->pool); c->pool = nullptr;
    c->pool_len = 0;
}

static void build_models(const uint8_t* inp, size_t n,
                         uint32_t* cum1,        // [256 * 257]
                         CtxPool2* ctx2)
{
    // --- order-1 raw counts (linear positions only, i >= 1) ---
    uint64_t* o1 = (uint64_t*)calloc((size_t)256 * 256, sizeof(uint64_t));
    for (size_t i = 1; i < n; ++i) {
        uint8_t p = inp[i - 1];
        uint8_t c = inp[i];
        o1[(size_t)p * 256 + c]++;
    }
    for (int p = 0; p < 256; ++p) {
        uint64_t* row = o1 + (size_t)p * 256;
        uint64_t total = 0;
        for (int c = 0; c < 256; ++c) {
            if (row[c] == 0) row[c] = 1; // floor so r > 0
            total += row[c];
        }
        uint32_t* dst = cum1 + (size_t)p * 257;
        uint64_t acc = 0;
        for (int c = 0; c < 256; ++c) {
            dst[c] = (uint32_t)((acc * ((uint64_t)1 << 32)) / total);
            acc += row[c];
        }
        dst[256] = 0; // sentinel; caller treats edge==256 as 2^32 via cum_at()
    }
    free(o1);

    // --- order-2 counts, sparse-packed (linear positions only, i >= 2) ---
    ctx2->off  = (uint32_t*)malloc((size_t)65536 * sizeof(uint32_t));
    for (size_t i = 0; i < 65536; ++i) ctx2->off[i] = 0xFFFFFFFFu;

    uint8_t* seen = (uint8_t*)calloc(65536, 1);
    for (size_t i = 2; i < n; ++i) {
        size_t qp = (size_t)inp[i - 2] * 256 + inp[i - 1];
        seen[qp] = 1;
    }
    size_t n_ctx = 0;
    for (size_t qp = 0; qp < 65536; ++qp) {
        if (seen[qp]) { ctx2->off[qp] = (uint32_t)(n_ctx * 257); n_ctx++; }
    }
    free(seen);

    ctx2->pool_len = n_ctx * 257;
    ctx2->pool = (uint32_t*)calloc(ctx2->pool_len, sizeof(uint32_t));

    uint64_t* raw = (uint64_t*)calloc(n_ctx * 256, sizeof(uint64_t));
    for (size_t i = 2; i < n; ++i) {
        size_t qp = (size_t)inp[i - 2] * 256 + inp[i - 1];
        uint32_t off = ctx2->off[qp];
        size_t row_idx = off / 257;
        raw[row_idx * 256 + inp[i]]++;
    }
    for (size_t qp = 0; qp < 65536; ++qp) {
        uint32_t off = ctx2->off[qp];
        if (off == 0xFFFFFFFFu) continue;
        size_t row_idx = off / 257;
        uint64_t* rcnt = raw + row_idx * 256;
        uint64_t total = 0;
        for (int c = 0; c < 256; ++c) {
            if (rcnt[c] == 0) rcnt[c] = 1;
            total += rcnt[c];
        }
        uint32_t* dst = ctx2->pool + off;
        uint64_t acc = 0;
        for (int c = 0; c < 256; ++c) {
            dst[c] = (uint32_t)((acc * ((uint64_t)1 << 32)) / total);
            acc += rcnt[c];
        }
        dst[256] = 0; // sentinel
    }
    free(raw);
}

// Cumulative-frequency value at edge c (0..256) for context (q,p).
// Internal symbols use ctx2 if present, else cum1[p].
static inline uint64_t cum_at(uint8_t q, uint8_t p, int c,
                              const uint32_t* cum1, const CtxPool2* ctx2)
{
    if (c == 0) return 0;
    if (c == 256) return ((uint64_t)1 << 32);
    uint32_t off = ctx2->off[(size_t)q * 256 + p];
    if (off != 0xFFFFFFFFu) return ctx2->pool[off + c];
    return cum1[(size_t)p * 257 + c];
}

// ---- key build (cyclic, on a doubled buffer) --------------------------------

// Run the inner code loop for syms_consumed symbols starting at offset
// `start_offset` from position j. Reads use inp2 = inp || inp (size 2n) so
// no per-iteration mod is needed; the loop is capped at n - start_offset so
// the entire cyclic rotation is read at most once across all key levels.
//
// Requires: start_offset >= 2 (so inp2[k-1] and inp2[k-2] are valid for the
// initial k = j + start_offset).
static inline uint64_t run_code(const uint8_t* inp2, size_t n,
                                const uint32_t* cum1, const CtxPool2* ctx2,
                                size_t j, size_t start_offset,
                                size_t* out_cap)
{
    uint64_t code = 0;
    uint64_t span = ~(uint64_t)0;
    size_t   syms = 0;
    size_t   max_syms = (n > start_offset) ? (n - start_offset) : 0;
    size_t   k = j + start_offset;
    while (syms < max_syms && span >= ((uint64_t)1 << 16)) {
        uint8_t c  = inp2[k];
        uint8_t p  = inp2[k - 1];
        uint8_t q  = inp2[k - 2];
        uint64_t lo = cum_at(q, p, c    , cum1, ctx2);
        uint64_t hi = cum_at(q, p, c + 1, cum1, ctx2);
        uint64_t r  = hi - lo;
        if (r == 0) r = 1; // safety: never zero the span
        code += mul_u64_u32_hi(span, (uint32_t)lo);
        span  = mul_u64_u32_hi(span, (uint32_t)r);
        if (span == 0) span = 1;
        ++k;
        ++syms;
    }
    if (out_cap) *out_cap = syms;
    return code;
}

// Level-1 key: raw 2 bytes ++ top 48 bits of code.
static inline uint64_t build_level1_key(const uint8_t* inp2, size_t n,
                                        const uint32_t* cum1, const CtxPool2* ctx2,
                                        size_t j, size_t* out_cap)
{
    uint64_t code = run_code(inp2, n, cum1, ctx2, j, 2, out_cap);
    uint8_t b0 = inp2[j];
    uint8_t b1 = inp2[j + 1];
    return ((uint64_t)b0 << 56)
         | ((uint64_t)b1 << 48)
         | (code >> 16);
}

// Extension (level-2+) key: top 64 bits of code from the given offset.
// The leading raw bytes are already pinned by the tied parent's prefix.
static inline uint64_t build_ext_key(const uint8_t* inp2, size_t n,
                                     const uint32_t* cum1, const CtxPool2* ctx2,
                                     size_t j, size_t start_offset,
                                     size_t* out_cap)
{
    return run_code(inp2, n, cum1, ctx2, j, start_offset, out_cap);
}

static void build_keys(const uint8_t* inp2, size_t n,
                       const uint32_t* cum1, const CtxPool2* ctx2,
                       uint64_t* keys)
{
    for (size_t j = 0; j < n; ++j) {
        keys[j] = build_level1_key(inp2, n, cum1, ctx2, j, nullptr);
    }
}

// ---- distribution -----------------------------------------------------------

static void distribute(const uint64_t* keys, size_t n,
                       uint32_t* bucket_base,
                       uint64_t* keys_perm, uint32_t* pos_perm)
{
    const size_t NB = (size_t)1 << K_TOP_BITS;
    uint32_t* hist = (uint32_t*)calloc(NB + 1, sizeof(uint32_t));
    for (size_t j = 0; j < n; ++j) {
        size_t b = (size_t)(keys[j] >> (64 - K_TOP_BITS));
        hist[b]++;
    }
    uint32_t acc = 0;
    for (size_t b = 0; b <= NB; ++b) {
        bucket_base[b] = acc;
        acc += hist[b];
    }
    memcpy(hist, bucket_base, NB * sizeof(uint32_t));
    for (size_t j = 0; j < n; ++j) {
        size_t b = (size_t)(keys[j] >> (64 - K_TOP_BITS));
        uint32_t dst = hist[b]++;
        keys_perm[dst] = keys[j];
        pos_perm [dst] = (uint32_t)j;
    }
    free(hist);
}

// ---- slab sort --------------------------------------------------------------

// Sort (key, pos) pairs by key, breaking ties by pos. std::sort is used here
// for clarity / debug parity; the plan documents bitonic networks for the
// SIMD-friendly path. cswap_kp above is the branchless network primitive.
static void slab_sort(uint64_t* k, uint32_t* p, size_t cnt) {
    if (cnt < 2) return;
    struct KP { uint64_t k; uint32_t p; };
    KP* tmp = (KP*)malloc(cnt * sizeof(KP));
    for (size_t i = 0; i < cnt; ++i) { tmp[i].k = k[i]; tmp[i].p = p[i]; }
    std::sort(tmp, tmp + cnt, [](const KP& a, const KP& b){
        if (a.k != b.k) return a.k < b.k;
        return a.p < b.p;
    });
    for (size_t i = 0; i < cnt; ++i) { k[i] = tmp[i].k; p[i] = tmp[i].p; }
    free(tmp);
    (void)cswap_kp; // referenced for documentation; std::sort path used here
}

// ---- tie resolution ---------------------------------------------------------

// Resolve any runs of equal keys in [k, k+cnt) via cyclic memcmp + pos
// tiebreak. Tied keys can come from either truly-equal cyclic rotations or
// from bit truncation of the 48-bit code tail; the plan's recursive key
// extension is *unsafe* in the latter case because the divergent symbol may
// have already been consumed by the prior level (contributing only to the
// truncated low bits), so the extension would resume under *different*
// (q,p) contexts on the two sides, breaking the affine-map argument.
// Cyclic memcmp is always correct and ties are rare for natural inputs.
static void resolve_ties(uint64_t* k, uint32_t* p, size_t cnt,
                         const uint8_t* inp2, size_t n,
                         const uint32_t* /*cum1*/, const CtxPool2* /*ctx2*/,
                         size_t /*this_start_offset*/, int /*depth*/)
{
    if (cnt < 2) return;
    size_t i = 0;
    while (i < cnt) {
        size_t j = i + 1;
        while (j < cnt && k[j] == k[i]) ++j;
        size_t run = j - i;
        if (run >= 2) {
            uint32_t* items = (uint32_t*)malloc(run * sizeof(uint32_t));
            for (size_t t = 0; t < run; ++t) items[t] = p[i + t];
            std::sort(items, items + run, [inp2, n](uint32_t a, uint32_t b){
                return cmp_cyclic2(inp2, n, a, b) < 0;
            });
            for (size_t t = 0; t < run; ++t) p[i + t] = items[t];
            free(items);
        }
        i = j;
    }
}

// ---- per-bucket processing --------------------------------------------------

static void emit_block(const uint32_t* p, size_t cnt,
                       uint8_t* out, size_t* out_idx, uint32_t* pi,
                       const uint8_t* inp2, size_t n)
{
    for (size_t t = 0; t < cnt; ++t) {
        uint32_t pos = p[t];
        // inp[(pos - 1 + n) % n] == inp2[pos == 0 ? n - 1 : pos - 1]
        uint8_t b = (pos == 0) ? inp2[n - 1] : inp2[pos - 1];
        out[*out_idx] = b;
        if (pos == 0) *pi = (uint32_t)*out_idx;
        (*out_idx)++;
    }
}

static void process_bucket(const uint64_t* bkeys, const uint32_t* bpos, size_t m,
                           uint8_t* out, size_t* out_idx, uint32_t* pi,
                           const uint8_t* inp2, size_t n,
                           const uint32_t* cum1, const CtxPool2* ctx2)
{
    if (m == 0) return;

    if (m <= K_SLAB_TARGET) {
        uint64_t* k = (uint64_t*)malloc(m * sizeof(uint64_t));
        uint32_t* p = (uint32_t*)malloc(m * sizeof(uint32_t));
        memcpy(k, bkeys, m * sizeof(uint64_t));
        memcpy(p, bpos , m * sizeof(uint32_t));
        slab_sort(k, p, m);
        resolve_ties(k, p, m, inp2, n, cum1, ctx2, /*this_start_offset=*/2, 0);
        emit_block(p, m, out, out_idx, pi, inp2, n);
        free(k); free(p);
        return;
    }

    // Slab-scan path for dense buckets.
    size_t S = (m + K_SLAB_TARGET - 1) / K_SLAB_TARGET;
    uint64_t bucket_prefix = bkeys[0] & ((uint64_t)0xFFFF << 48);
    uint64_t span48 = (uint64_t)1 << 48;
    uint64_t w = span48 / S;
    if (w == 0) w = 1;

    size_t cap = 2 * K_SLAB_TARGET;
    uint64_t* sk = (uint64_t*)malloc(cap * sizeof(uint64_t));
    uint32_t* sp = (uint32_t*)malloc(cap * sizeof(uint32_t));

    for (size_t s = 0; s < S; ++s) {
        uint64_t lo_v = bucket_prefix | ((uint64_t)s * w);
        uint64_t hi_v;
        if (s == S - 1) {
            hi_v = bucket_prefix + span48; // next bucket prefix (exclusive)
        } else {
            hi_v = bucket_prefix | ((uint64_t)(s + 1) * w);
        }

        // Stream compaction.
        size_t cnt = 0;
        for (size_t i = 0; i < m; ++i) {
            uint64_t kv = bkeys[i];
            uint32_t pv = bpos [i];
            int take = (kv >= lo_v) & (kv < hi_v);
            if (cnt + 1 > cap) {
                cap *= 2;
                sk = (uint64_t*)realloc(sk, cap * sizeof(uint64_t));
                sp = (uint32_t*)realloc(sp, cap * sizeof(uint32_t));
            }
            sk[cnt] = kv;
            sp[cnt] = pv;
            cnt += (size_t)take;
        }

        if (cnt == 0) continue;

        slab_sort(sk, sp, cnt);
        resolve_ties(sk, sp, cnt, inp2, n, cum1, ctx2, /*this_start_offset=*/2, 0);
        emit_block(sp, cnt, out, out_idx, pi, inp2, n);
    }

    free(sk); free(sp);
}

// ---- fast driver ------------------------------------------------------------

static void rcbwt(const uint8_t* inp, size_t n,
                  uint8_t* out, uint32_t* pi)
{
    *pi = 0;
    if (n == 0) return;
    if (n == 1) { out[0] = inp[0]; *pi = 0; return; }

    // Doubled buffer for cyclic key building.
    uint8_t* inp2 = (uint8_t*)malloc(2 * n);
    memcpy(inp2,     inp, n);
    memcpy(inp2 + n, inp, n);

    uint32_t* cum1 = (uint32_t*)calloc((size_t)256 * 257, sizeof(uint32_t));
    CtxPool2 ctx2 = { nullptr, nullptr, 0 };
    build_models(inp, n, cum1, &ctx2);

    uint64_t* keys = (uint64_t*)malloc(n * sizeof(uint64_t));
    build_keys(inp2, n, cum1, &ctx2, keys);

    const size_t NB = (size_t)1 << K_TOP_BITS;
    uint32_t* bucket_base = (uint32_t*)malloc((NB + 1) * sizeof(uint32_t));
    uint64_t* keys_perm   = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint32_t* pos_perm    = (uint32_t*)malloc(n * sizeof(uint32_t));
    distribute(keys, n, bucket_base, keys_perm, pos_perm);
    free(keys);

    size_t out_idx = 0;
    for (size_t b = 0; b < NB; ++b) {
        size_t lo = bucket_base[b];
        size_t hi = bucket_base[b + 1];
        size_t m  = hi - lo;
        if (m == 0) continue;
        process_bucket(keys_perm + lo, pos_perm + lo, m,
                       out, &out_idx, pi, inp2, n, cum1, &ctx2);
    }

    if (out_idx != n) {
        fprintf(stderr, "internal error: emitted %zu of %zu bytes\n", out_idx, n);
        exit(2);
    }

    free(bucket_base);
    free(keys_perm);
    free(pos_perm);
    free(cum1);
    free_ctx2(&ctx2);
    free(inp2);
}

// ---- dumb reference BWT -----------------------------------------------------

static void dumb_bwt(const uint8_t* inp, size_t n,
                     uint8_t* out, uint32_t* pi)
{
    *pi = 0;
    if (n == 0) return;
    uint8_t* inp2 = (uint8_t*)malloc(2 * n);
    memcpy(inp2,     inp, n);
    memcpy(inp2 + n, inp, n);

    uint32_t* sa = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (size_t i = 0; i < n; ++i) sa[i] = (uint32_t)i;
    std::sort(sa, sa + n, [inp2, n](uint32_t a, uint32_t b){
        return cmp_cyclic2(inp2, n, a, b) < 0;
    });
    for (size_t i = 0; i < n; ++i) {
        uint32_t pos = sa[i];
        out[i] = (pos == 0) ? inp2[n - 1] : inp2[pos - 1];
        if (pos == 0) *pi = (uint32_t)i;
    }
    free(sa);
    free(inp2);
}

// ---- I/O --------------------------------------------------------------------

static size_t read_file(const char* path, uint8_t** buf) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); *buf = nullptr; return 0; }
    size_t n = (size_t)sz;
    *buf = (uint8_t*)malloc(n);
    if (fread(*buf, 1, n, f) != n) {
        perror("fread"); fclose(f); free(*buf); *buf = nullptr; return 0;
    }
    fclose(f);
    return n;
}

static int write_file(const char* path, uint32_t pi,
                      const uint8_t* bwt, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }
    uint8_t hdr[4];
    hdr[0] = (uint8_t)( pi        & 0xFF);
    hdr[1] = (uint8_t)((pi >>  8) & 0xFF);
    hdr[2] = (uint8_t)((pi >> 16) & 0xFF);
    hdr[3] = (uint8_t)((pi >> 24) & 0xFF);
    if (fwrite(hdr, 1, 4, f) != 4) { perror("fwrite"); fclose(f); return 0; }
    if (n > 0 && fwrite(bwt, 1, n, f) != n) { perror("fwrite"); fclose(f); return 0; }
    fclose(f);
    return 1;
}

static int compare_and_report(const uint8_t* a, const uint8_t* b, size_t n,
                              uint32_t pi_a, uint32_t pi_b)
{
    int ok = 1;
    if (pi_a != pi_b) {
        printf("pi mismatch: fast=%u dumb=%u\n", pi_a, pi_b);
        ok = 0;
    }
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("first BWT mismatch at offset %zu: fast=0x%02X dumb=0x%02X\n",
                   i, a[i], b[i]);
            size_t lo = (i >= 16) ? i - 16 : 0;
            size_t hi = (i + 16 < n) ? i + 16 : n;
            printf("  fast["); for (size_t j = lo; j < hi; ++j) printf("%s%02X", j==i?"|":" ", a[j]); printf(" ]\n");
            printf("  dumb["); for (size_t j = lo; j < hi; ++j) printf("%s%02X", j==i?"|":" ", b[j]); printf(" ]\n");
            ok = 0;
            break;
        }
    }
    if (ok) printf("OK\n");
    return ok;
}

// ---- main -------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s inputfile outputfile [dumb-BWT-file]\n", argv[0]);
        return 1;
    }

    uint8_t* inp = nullptr;
    size_t n = read_file(argv[1], &inp);
    if (n == 0) {
        fprintf(stderr, "empty input or read error\n");
        if (inp) free(inp);
        return 1;
    }

    uint8_t* bwt_out = (uint8_t*)malloc(n);
    uint32_t pi = 0;
    rcbwt(inp, n, bwt_out, &pi);

    if (!write_file(argv[2], pi, bwt_out, n)) {
        free(inp); free(bwt_out); return 1;
    }

    int rc = 0;
    if (argc == 4) {
        uint8_t* bwt_ref = (uint8_t*)malloc(n);
        uint32_t pi_ref = 0;
        dumb_bwt(inp, n, bwt_ref, &pi_ref);
        if (!write_file(argv[3], pi_ref, bwt_ref, n)) {
            free(inp); free(bwt_out); free(bwt_ref); return 1;
        }
        if (!compare_and_report(bwt_out, bwt_ref, n, pi, pi_ref)) rc = 2;
        free(bwt_ref);
    }

    free(inp);
    free(bwt_out);
    return rc;
}
