# RCBWT Demo — C++ Implementation Plan

A single-file C++ demo of the BWT-replacement scheme described in `RC_BWT.md`:
a streaming, branchless radix on an order-preserving entropy key, used as a
cache-resident kernel. The demo is single-threaded, scalar (SIMD-friendly but
no intrinsics), uses `fopen` / `printf` only, and includes an optional
`std::sort`-based reference BWT for output comparison.

```
Usage:  rcbwt inputfile outputfile [dumb-BWT-file]
```

If `dumb-BWT-file` is given, the program also runs the reference `std::sort`
BWT and writes its output there. The driver then compares the two byte-for-byte
and prints `OK` or the first mismatch position.

---

## 1. BWT formulation

To keep size arithmetic clean and avoid a "no 0x00" input restriction, the
demo uses the **cyclic-rotation + primary-index** formulation (no sentinel):

- Input is `n` bytes, `inp[0..n-1]`.
- Consider all `n` cyclic rotations of the input.
- Sort them lexicographically (cyclic comparison).
- The BWT is the sequence of last bytes of the sorted rotations.
  Equivalently, if sorted rotation `i` starts at position `pos[i]`, then
  `bwt[i] = inp[(pos[i] - 1 + n) mod n]`.
- The primary index `pi` is the value of `i` for which `pos[i] == 0`.

Output is exactly **`n` bytes of BWT plus a 4-byte little-endian `pi`
header**:

```
[ 4 bytes LE primary index pi ][ n bytes BWT ]
```

Both the fast and dumb paths produce this same format so the bodies can be
`memcmp`'d and `pi` compared explicitly.

---

## 2. Project layout

```
rcbwt.cpp        single translation unit, ~700 lines
plan.md          this document
RC_BWT.md        algorithm analysis (input)
LICENSE
```

Compile with `g++ -O2 -std=c++17 rcbwt.cpp -o rcbwt`. No external deps.
`std::sort` and `<algorithm>` are used **only** in the reference path
(§8) and (optionally, via a debug toggle) in slab sort (§6.3).

---

## 3. I/O and top-level driver

All file I/O via `fopen` / `fread` / `fwrite` / `fclose`. All terminal
output via `printf` / `fprintf(stderr, ...)`. No `iostream`, no `cout`.

```c
int main(int argc, char** argv);
```

Steps:

1. Parse args. Reject if `argc < 3 || argc > 4`. Print usage to stderr
   and return non-zero.
2. Read entire input file into a heap buffer `uint8_t* inp; size_t n`
   via `fopen` + `fseek`/`ftell` for size + single `fread`. Bail (print
   error to stderr, exit non-zero) on partial read or `n == 0`.
3. Allocate `uint8_t* bwt_out = (uint8_t*)malloc(n);` and declare
   `uint32_t pi;` for the fast path.
4. Run `rcbwt(inp, n, bwt_out, &pi);` (§4–§7).
5. Write `pi` (4 bytes LE) followed by `bwt_out` (n bytes) to
   `outputfile`.
6. If `argv[3]` given:
   - Allocate `uint8_t* bwt_ref = (uint8_t*)malloc(n);` and declare
     `uint32_t pi_ref;`.
   - Run `dumb_bwt(inp, n, bwt_ref, &pi_ref);` (§8).
   - Write `pi_ref` + `bwt_ref` to `dumb-BWT-file`.
   - Compare: `memcmp(bwt_out, bwt_ref, n)` plus `pi == pi_ref`. Print
     `OK` or the first mismatch offset and 16 bytes either side in hex,
     and any `pi` mismatch.
7. Free everything, return.

---

## 4. Stage 1 — static order-2 model and per-position code

This is essentially the existing `rcbwt.cpp` logic, refactored into a
function, **reoriented for forward-suffix sorting** (the prototype's
reverse convention codes reversed prefixes; we want forward suffixes),
and extended to a 64-bit code so the composite key is order-preserving
(§5) rather than the raw scalar code.

### 4.1 Model orientation — forward (preceding-byte context)

For sorting forward suffixes, the order-preservation argument requires
each coded symbol's context to be its **preceding** (already-pinned)
bytes. So the order-2 model the demo builds is:

```
o2[q][p][c] = count of byte c at position i where
              q = inp[i - 2], p = inp[i - 1].
```

The prototype's reverse convention (context = *following* bytes) sorts
reversed prefixes and would not be order-preserving on forward suffixes —
that flip is the single change vs the prototype.

For model building we use **linear positions only** (`i >= 2` in
`inp[0..n-1]`), ignoring cyclic wrap for the two leading samples. This
is a statistical-only shortcut: the keys remain order-preserving by the
affine-map argument regardless of where the model's counts came from —
the model only affects compaction density, not correctness.

### 4.2 Model build

Normalise each `o2[q][p][·]` distribution to a 32-bit cumulative-frequency
table `cum2[q][p][c]` with `cum2[q][p][256] = 2^32`, using `div64`-style
fixed-point so `r = h - l > 0` for every symbol present. Allocate zero
counts a 1-symbol floor before normalising so absent symbols stay
representable (keeps the code-build loop well-defined in pathological
tails).

Also build the order-1 cumulative table `cum1[p][c]` by marginalisation;
the demo doesn't use the §2.3 ramp itself (see §5.2), but `cum1` is the
fallback at code-build time when an order-2 context has zero observations.

Memory:
- `cum2`: 256·256·257 · 4 B ≈ 64 MB if dense → too big. Use 256·256
  contexts but store **only used contexts** via a flat indirection:
  a `uint32_t ctx2_off[65536]` offset table and a packed pool of
  257-entry rows for non-empty contexts. For BOOK1 only a few thousand
  contexts are non-empty.
- Absent contexts at code-build time fall back to `cum1[p][·]`.

Function:

```c
void build_models(const uint8_t* inp, size_t n,
                  uint32_t* cum1,           // [256][257] flat
                  CtxPool2* ctx2);          // packed order-2 table
```

### 4.3 Per-position code and key build

For each position `j` from `0` to `n-1`, the **code tail** packs the
forward suffix starting at `inp[j+2]` (the symbols beyond the two raw
prefix bytes of §5):

```c
uint64_t code = 0;             // 64-bit fixed-point in [0, 2^64)
uint64_t span = ~0ULL;         // current interval width
size_t   k    = j + 2;
while (k < n && span >= (1ULL << 16)) {
    uint8_t  c  = inp[k];
    uint8_t  p  = inp[k - 1];          // preceding byte (pinned)
    uint8_t  q  = inp[k - 2];          // 2-preceding byte (pinned)
    uint32_t lo = cum_lookup(q, p, c    , ctx2, cum1);
    uint32_t hi = cum_lookup(q, p, c + 1, ctx2, cum1);
    uint32_t r  = hi - lo;
    code += mul_u64_u32_hi(span, lo);  // (span * lo) >> 32
    span  = mul_u64_u32_hi(span, r);   // (span * r ) >> 32
    ++k;
}
keys[j] =  ((uint64_t)inp[j]                 << 56)
        |  ((uint64_t)inp[(j + 1) % n]       << 48)
        |  (code >> 16);               // top 48 bits of code tail
```

`mul_u64_u32_hi(a,b)` = `(__uint128_t)a * b >> 32` (GCC/Clang) — one
branchless 64×32 → 96 → take-high-64 op. `cum_lookup` returns the
order-2 cumulative count if the `(q,p)` context exists, else the
order-1 count from `cum1[p][·]`.

Stop conditions:
- `k >= n`: ran off the end of the input. The demo does **not** wrap
  the code loop cyclically (statistical shortcut); any keys that tie
  here are resolved via §6.5 cyclic `memcmp`.
- `span < 2^16`: precision is effectively exhausted — further multiplies
  shift the result to ~0 and add no useful information. Stopping early
  keeps the bottom code bits well-defined.

`keys` is one flat `uint64_t* keys = malloc(n*8)`; the composite key is
written directly without storing a separate `code[]` array.

### 4.4 Tail handling

For `j` near `n` (`j + 2 >= n`), the loop runs zero or one iteration
and the resulting key is essentially just the two raw bytes
`(inp[j] << 56) | (inp[(j+1) % n] << 48)` with `code ≈ 0`. Such keys
may tie with each other; §6.5 resolves via cyclic `memcmp`. For typical
text inputs this affects only the last two keys.

Computing `inp[(j+1) % n]` is one modulus; for `j < n-1` this is just
`inp[j+1]`, and only `j = n-1` actually wraps. Branchlessly:
`inp[(j + 1) - n * (j == n - 1)]` if a branch in the hot loop matters,
but a single predictable `% n` is fine here.

---

## 5. The sort key — composite, order-preserving

### 5.1 Layout (64 bits)

```
 bits 63..56   first symbol            inp[j]              (raw, 8 bits)
 bits 55..48   second symbol           inp[(j + 1) % n]    (raw, 8 bits)
 bits 47..0    order-2 code tail of    inp[(j + 2)..]      (48 bits)
```

Compared integer-wise. Integer order = lexicographic order of the
forward cyclic rotation starting at `j`. The BWT byte for the rotation
at `j` is `inp[(j - 1 + n) % n]` (§1).

### 5.2 Order ramp — and why we omit it

Per §2.3 of `RC_BWT.md`: levels 1..K of the key must use context-order
≤ (level − 1) — level 1 raw, level 2 order-1, level 3+ order-2.

For simplicity and SIMD-friendliness the demo uses **raw bytes for
levels 1 and 2** (16 bits flat) and order-2 code for level 3 onward
(48 bits). This is the "raw first byte, ramp/code thereafter" middle
ground from `RC_BWT.md` §2.4, extended to two raw bytes. It costs ~3
symbols of key depth vs the full ramp (≈22 symbols/64-bit vs ≈26 — see
`RC_BWT.md` §2.4 table row "2 raw + order-2 tail"), in exchange for:
- eliminating per-symbol context-order branching from the hot key-build
  loop;
- exposing two raw bytes to the radix distribution pass (§6.1), which
  can then index buckets directly on them — *and* guarantee the
  uniform-key assumption for slab partitioning (§6.2) holds, since
  within a bucket all keys share the raw prefix and differ only in the
  48-bit code.

### 5.3 Why ties, not inversions

Per §2.2 of `RC_BWT.md`: when two keys share the top 16 bits (same
first two bytes), their 48-bit code tails are produced by the **same**
affine map (same `lo, r` at every step) until the first divergent
input symbol, after which the diverging symbol's intervals are
disjoint and correctly-ordered. So an inversion is impossible — only
truncation-induced ties (and short-key end-of-buffer ties from §4.4).
The demo asserts this empirically by comparing against the reference
BWT (§8) on every test input.

---

## 6. Stage 2 — radix selection (branchless inner kernel)

The full per-suffix sort is **MSD-radix on the top 16 bits of the key**
(one bucket per `(inp[j], inp[(j+1) % n])` pair), then either a direct
sort or a per-bucket slab scan over the 48-bit code tail:

```
distribution pass (16 bits)  →  bucket  →  if m ≤ N: bitonic sort
   one O(n) pass                            else:     slab scan kernel
                                                      → emit BWT bytes in sorted order
```

### 6.1 Distribution pass (top 16 bits)

`K_TOP_BITS = 16` → `65 536` buckets, indexed by `key >> 48 =
(inp[j] << 8) | inp[(j+1) % n]`. Aligning the bucket split to the raw-byte
boundary means each bucket's keys differ only in the 48-bit code tail,
which is **near-uniform** by the entropy-coding argument — so when a
slab partition is needed (§6.2) it really does balance.

Memory cost: a 65 537-entry `uint32_t bucket_base[]` = 256 KB. One big
`uint64_t* keys_perm` and `uint32_t* pos_perm`, each of size `n`, hold
the bucketed key/pos pairs.

Algorithm:

1. First pass: histogram top 16 bits across all `keys[j]`.
2. Prefix-sum into `bucket_base`.
3. Second pass: scatter `(key, pos)` pairs into
   `keys_perm`/`pos_perm` at per-bucket cursors derived from the
   prefix sum.

This is the one unavoidable scatter. Keep it scalar — no software
write-combining for the demo, but written tightly (no branches in the
scatter body) so a SIMD port is mechanical.

Avoid per-bucket `vector<…>` (would pull in STL allocators and break
the SIMD-friendly contiguous-array discipline).

### 6.2 Per-bucket processing

For each non-empty bucket of size `m = bucket_base[b+1] - bucket_base[b]`:

- **If `m ≤ K_SLAB_TARGET`** (sparse bucket — most buckets on text):
  copy the `(key, pos)` block into the slab buffer, bitonic-sort
  (§6.3), resolve ties (§6.4–§6.5), and emit (§6.6). No slab partition
  needed.
- **If `m > K_SLAB_TARGET`** (dense bucket — densest 2-byte prefixes
  in natural text, e.g. " t", "th"): partition the 48-bit code
  subspace into `S = ceil(m / K_SLAB_TARGET)` slabs of width
  `w = (1ULL << 48) / S`. Process slabs in ascending `s = 0..S-1`. For
  each slab:

```c
size_t cnt = 0;
uint64_t bucket_prefix = (uint64_t)b << 48;
uint64_t lo_v = bucket_prefix | ((uint64_t)s     * w);
uint64_t hi_v = (s == S - 1) ? (bucket_prefix | ((1ULL << 48) - 1)) + 1
                             : bucket_prefix | ((uint64_t)(s + 1) * w);
for (size_t i = 0; i < m; ++i) {
    uint64_t k = bucket_keys[i];
    uint32_t p = bucket_pos[i];
    int take = (k >= lo_v) & (k < hi_v);   // branchless predicate
    slab_keys[cnt] = k;                    // speculative store
    slab_pos [cnt] = p;
    cnt += (size_t)take;                   // advance only if taken
}
```

This is the **stream-compaction kernel** (`RC_BWT.md` §4.2). On a real
SIMD port: `VPCMP` + `VPCOMPRESSQ`. Here it is scalar but branch-free
in the body; only the loop back-edge is a branch. The speculative
store with conditional counter advance is the canonical SIMD-friendly
form. After compaction, sort the slab (§6.3), resolve ties
(§6.4–§6.5), emit (§6.6), then move to slab `s+1`.

Slab buffer is sized at `2 * K_SLAB_TARGET` entries; an assert guards
`cnt <= 2*N` per pass. On the rare `cnt > 2*N` event (tie group larger
than the buffer, §6.4 below), `realloc` to fit.

### 6.3 Slab sort — branchless network

For up to 32 elements use a fixed-shape Batcher bitonic sorting
network over `(key, pos)` pairs, sorting by key. The comparator is a
branchless conditional swap:

```c
static inline void cswap(uint64_t* k, uint32_t* p, size_t i, size_t j) {
    uint64_t ki = k[i], kj = k[j];
    uint32_t pi = p[i], pj = p[j];
    int swap = (ki > kj);
    uint64_t mk = -(uint64_t)swap;
    uint32_t mp = -(uint32_t)swap;
    k[i] = ki ^ ((ki ^ kj) & mk);
    k[j] = kj ^ ((ki ^ kj) & mk);
    p[i] = pi ^ ((pi ^ pj) & mp);
    p[j] = pj ^ ((pi ^ pj) & mp);
}
```

The network shape is unrolled per size class (8, 16, 32). For
`cnt > 32` (the sparse-bucket bitonic path with naturally-large
buckets — slab kernel paths are bounded near N=256) the demo uses an
in-place 8-bit MSD radix on the next byte of the key, recursing into
bitonic networks at the leaves. A compile-time toggle
`K_USE_STD_SORT_IN_SLAB = 1` swaps in `std::sort` as a debug aid.

### 6.4 Tie handling — kept out of the scan

Per `RC_BWT.md` §3.4, the scan loop does **no** tie resolution; ties
are detected and resolved after the per-slab (or per-bucket) sort.
Under the value-based slab partition (§6.2), a tie group cannot
straddle a slab boundary (the last paragraph of `RC_BWT.md` §3.4: "a
value boundary never splits an equal-value group"), so every tie group
is fully contained in the slab it falls in. The only residual hard
case is a tie group larger than the slab buffer — handled by `realloc`
in §6.2 and then by §6.5.

### 6.5 Recursive key extension for ties

Given a tie group of indices into `inp`, compute a **second-level
key** = next 48 bits of the same code stream, resumed from the offset
`cap[j]` (the number of symbols already absorbed by the first key —
identical for all tied members, since equal level-1 keys mean
identical decoded prefixes to that depth, per `RC_BWT.md` §3.3). Sort
the group by the new key with the same slab kernel. Recurse until
groups are singletons or the depth reaches a configurable cap
`K_MAX_LEVELS = 4`; **beyond that depth**, fall back to direct cyclic
`memcmp` of the suffixes as the safety net.

For the demo, compute `cap[j]` lazily for tie members only: re-run the
inner loop of §4.3 from `j+2`, counting symbols, stopping at the same
condition that ended the level-1 build (`span < 2^16` or `k >= n`).
This keeps Stage 1 free of an extra `cap[]` array.

The cyclic-`memcmp` safety net is the same comparator the dumb BWT
uses (§8), promoted to a helper:

```c
static int cmp_cyclic(const uint8_t* inp, size_t n,
                      uint32_t a, uint32_t b);   // <0 if a<b, 0 eq, >0 if a>b
```

### 6.6 Emission

Walk buckets in ascending order of `b = 0..65535`, within each bucket
process slabs (or the whole bucket if `m ≤ N`) in ascending value,
within each sorted slab walk the `pos[]` array, and emit:

```c
bwt_out[out_idx] = inp[(pos - 1 + n) % n];
if (pos == 0) *pi = (uint32_t)out_idx;
++out_idx;
```

After the last bucket, `out_idx == n` (asserted).

---

## 7. Driver function

```c
static void rcbwt(const uint8_t* inp, size_t n,
                  uint8_t* out, uint32_t* pi);
```

Internally:
1. Allocate working buffers (model pool, `keys`, `keys_perm`,
   `pos_perm`, `bucket_base`, per-slab buffers).
2. `build_models(inp, n, cum1, ctx2)` (§4.1–§4.2).
3. `build_keys(inp, n, cum1, ctx2, keys)` (§4.3 inner loop).
4. `distribute(keys, n, bucket_base, keys_perm, pos_perm)` (§6.1).
5. For each non-empty bucket: `process_bucket(...)` writes to `out` and
   updates `*pi` (§6.2–§6.6).
6. Free working buffers.

The `out` buffer is allocated by the caller (the driver) at size `n`.

---

## 8. Dumb reference BWT (`std::sort`)

Used only when `argv[3]` is given. Same direction and conventions as
the fast path (forward cyclic rotations, no sentinel, `pi` header),
just done the obvious naive way:

```c
void dumb_bwt(const uint8_t* inp, size_t n,
              uint8_t* out, uint32_t* pi);
```

1. Build `uint32_t* sa = malloc(n * 4)` initialised to `0..n-1`.
2. `std::sort(sa, sa + n, cmp)` with `cmp(a, b)` doing direct cyclic
   byte-by-byte comparison of the rotations starting at `a` and `b`:

```c
auto cmp = [&](uint32_t a, uint32_t b) -> bool {
    for (size_t k = 0; k < n; ++k) {
        uint8_t ca = inp[(a + k) % n];
        uint8_t cb = inp[(b + k) % n];
        if (ca != cb) return ca < cb;
    }
    return false;   // identical cyclic rotations — std::sort handles stably
};
```

3. Walk sorted `sa`: emit `out[i] = inp[(sa[i] - 1 + n) % n]`, record
   `*pi = i` when `sa[i] == 0`.

This is brute force — `O(n^2 log n)` worst case (long common cyclic
prefixes). Its sole purpose is to validate the fast path; do not
optimise. The dumb BWT and the fast path use the exact same BWT
definition (forward cyclic rotations, `pi`-indexed), so their outputs
must be byte-for-byte identical.

---

## 9. SIMD-friendliness rules (followed throughout)

1. **No data-dependent branches in hot loops.** Replace
   `if (cond) x = a; else x = b;` with
   `x = b ^ ((a ^ b) & -(int64_t)cond);` or the `cnt += take`
   speculative-store pattern (§6.2).
2. **Fixed-shape sort networks** for small `N`, not heap/`std::sort` in
   inner loops.
3. **All hot-loop arrays are flat `T*` from `malloc`**, indexed by
   `size_t`. No `std::vector::push_back` in the inner kernel.
4. **No virtual calls, no exceptions** in hot paths.
5. **64-bit keys** throughout (`RC_BWT.md` §3.1); 32 bits is only for
   toy `n ≤ 64K`.
6. **Single pass per array** wherever possible; the only multi-pass
   loop is the inner slab scan (§6.2), which is the intended kernel.

What we explicitly do **not** do (out of scope for this demo):

- `_mm*` intrinsics or `<immintrin.h>`.
- `std::thread`, `pthread`, OpenMP.
- Software write-combining or non-temporal stores.
- The 512 KB remap-table prefix scheme (`RC_BWT.md` §2.4); raw
  two-byte prefix is the SIMD-friendliest middle ground.

---

## 10. Verbose diagnostics (printf only)

A `--verbose` flag (or a compile-time `K_VERBOSE`) prints, via `printf`:

- Stage 1: file size, H0/H1/H2 in bits/symbol, mean / min / max key
  depth (inner-loop iterations before stop).
- Distribution: top-16-bit bucket-size histogram (decile summary;
  count of empty buckets, largest 10 buckets).
- Stage 2: per-bucket slab count, tie-run count, max recursion depth
  used.
- Validation (when `argv[3]` given): `OK`, or first mismatch offset
  plus 32 bytes of `bwt_out` and `bwt_ref` side-by-side in hex, plus
  any `pi` mismatch.

Keep diagnostic output to one line per phase by default; gate the
histograms behind verbose.

---

## 11. Function inventory (single TU)

```c
/* I/O */
static size_t   read_file (const char* path, uint8_t** buf);
static int      write_file(const char* path, uint32_t pi,
                           const uint8_t* bwt, size_t n);

/* helpers */
static inline uint64_t mul_u64_u32_hi(uint64_t a, uint32_t b);
static inline uint32_t div64(uint64_t num, uint64_t den);
static inline void     cswap(uint64_t* k, uint32_t* p, size_t i, size_t j);
static int             cmp_cyclic(const uint8_t* inp, size_t n,
                                  uint32_t a, uint32_t b);   /* §6.5, §8 */

/* stage 1 */
struct CtxPool2;
static void build_models(const uint8_t* inp, size_t n,
                         uint32_t* cum1, CtxPool2* ctx2);
static void build_keys  (const uint8_t* inp, size_t n,
                         const uint32_t* cum1, const CtxPool2* ctx2,
                         uint64_t* keys);

/* stage 2 */
static void distribute    (const uint64_t* keys, size_t n,
                           uint32_t* bucket_base /* [1<<16 + 1] */,
                           uint64_t* keys_perm, uint32_t* pos_perm);
static void process_bucket(const uint64_t* k, const uint32_t* p, size_t m,
                           uint32_t bucket_idx,
                           uint8_t* out, size_t* out_idx, uint32_t* pi,
                           const uint8_t* inp, size_t n,
                           const uint32_t* cum1, const CtxPool2* ctx2);
static void resolve_ties  (uint64_t* k, uint32_t* p, size_t cnt,
                           const uint8_t* inp, size_t n,
                           const uint32_t* cum1, const CtxPool2* ctx2,
                           int depth);
static void bitonic_sort  (uint64_t* k, uint32_t* p, size_t cnt);

/* fast driver */
static void rcbwt(const uint8_t* inp, size_t n,
                  uint8_t* out, uint32_t* pi);

/* reference */
static void dumb_bwt(const uint8_t* inp, size_t n,
                     uint8_t* out, uint32_t* pi);

/* compare */
static int compare_and_report(const uint8_t* a, const uint8_t* b, size_t n,
                              uint32_t pi_a, uint32_t pi_b);

int main(int argc, char** argv);
```

Approx LOC budget:

| section            | lines |
|--------------------|------:|
| I/O + driver       |    80 |
| models + keys      |   180 |
| distribute + bucket|   180 |
| ties + recursion   |   120 |
| bitonic sort       |    80 |
| dumb_bwt + cmp     |    60 |
| total              |  ~700 |

---

## 12. Build & test plan

### 12.1 Build

```
g++ -O2 -std=c++17 -Wall -Wextra rcbwt.cpp -o rcbwt
```

No `-march=native`; the demo should be reproducible across machines.
No SIMD flags needed since we use none.

### 12.2 Tests

1. **Tiny strings**: `"banana"`, `"mississippi"`, all-equal `"aaaa"`,
   alternating `"abab"`, single byte `"a"`. Run with `dumb-BWT-file`;
   require `OK`.
2. **Random bytes** at sizes 1 KB, 64 KB, 1 MB (any bytes, including
   `0x00`). Require `OK`.
3. **BOOK1 8 KB slice** (whatever the prototype used). Require `OK`,
   and `printf` the bucket-size histogram and tie counts so the
   measurements in `RC_BWT.md` §2.4 can be reproduced.
4. **Adversarial repeats**: `"abc"` × 10 000. Exercises §6.5 tie
   recursion. Require `OK`.
5. **Boundary sizes**: `n = 0` (driver prints "empty input" to stderr
   and exits 1 — tested by checking the exit code), `n = 1`, `n = 2`,
   `n = 65 536` (one key per bucket on average), `n = 65 537`.

The driver's `OK` / first-mismatch-offset report is the test oracle;
no separate test framework.

### 12.3 What the demo deliberately does **not** prove

- Asymptotic competitiveness with industrial suffix sorters: the demo
  is small-block, single-threaded, scalar.
- Stage-3 pipelining win (the headline benefit per `RC_BWT.md` §5.1):
  wiring an entropy coder is out of scope; the demo produces the BWT
  bytes in final order, which is the *enabler*, but doesn't itself
  encode them.
- AVX-512 throughput claims: the kernel is written branchlessly so the
  structure is faithful, but no measurements are made.

What it **does** prove:

- The composite key (raw 2 bytes + 48-bit forward-context order-2
  tail) yields **zero inversions** relative to the dumb sort on real
  inputs, validating `RC_BWT.md` §2.
- The branchless slab-filter kernel produces correct BWT output end to
  end, in final order, with deferred tie resolution.

---

## 13. Open choices (defaults picked, easy to change)

| knob               | default                  | rationale |
|--------------------|--------------------------|-----------|
| `K_TOP_BITS`       | 16 (65 536 buckets)      | aligns with raw 2-byte prefix; 48-bit code tail is near-uniform within a bucket |
| `K_SLAB_TARGET`    | 256                      | bitonic-sortable, branchless |
| `K_MAX_LEVELS`     | 4                        | each level resolves ~22 syms |
| key width          | 64 bits (16 raw + 48 code)| `RC_BWT.md` §3.1 ties bound |
| BWT formulation    | cyclic, no sentinel, with `pi` header | works on any byte input, clean off-by-one |
| context-2 storage  | indirect (offset table)  | dense table is 64 MB |
| dumb sort cmp      | cyclic rotation `memcmp` | clarity over speed |

---

## 14. Out of scope (deferred to follow-ups)

- True parallelism across buckets (`RC_BWT.md` §5.1).
- SIMD intrinsic versions of `distribute` and the slab scan.
- Multi-level radix on the code tail (we use a single 16-bit
  distribution + slab scan within each bucket; deeper buckets would
  help on very large inputs).
- The full `RC_BWT.md` §2.3 order ramp at the leading levels (we use
  two raw bytes for SIMD-friendliness).
- The remap-table prefix variant (`RC_BWT.md` §2.4).
- Entropy-coding stage 3 and the pipelining demo (item 4 of the
  next-experiments list in `RC_BWT.md` §6).
