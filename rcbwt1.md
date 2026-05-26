# rcbwt1.cpp vs rcbwt_sparse.cpp — comparison

`rcbwt1.cpp` is an alternative implementation of the same scheme. Both files
implement the streaming radix on the 64-bit order-preserving composite key
described in `plan.md` and `RC_BWT.md`, both use a sparse order-2 model
(offset table + packed 257-entry rows) with `cum1` fallback, and both produce
byte-identical BWT output on every input we tested (book1, tiny strings,
random bytes). They differ in several engineering choices and in one
non-trivial theoretical observation.

Verified equivalence on book1 (n = 768 771):

```
rcbwt1  == rcbwt
rcbwt1  == rcbwt_sparse
rcbwt1  == BWT.cpp
rcbwt1 -> unBWT.cpp roundtrips
```

## What the two files share

| Aspect                                                | rcbwt1 | rcbwt_sparse |
|-------------------------------------------------------|:------:|:------------:|
| Composite 64-bit key (16 raw + 48 code)               | yes    | yes          |
| Sparse order-2: `ctx2_off[ctx]` + packed pool         | yes    | yes          |
| `cum1` fallback for unobserved (q,p)                  | yes    | yes          |
| MSD radix on top 16 bits → per-bucket processing      | yes    | yes          |
| Slab partition over the 48-bit code subspace          | yes    | yes          |
| Branchless `cnt += take` stream-compaction kernel     | yes    | yes          |
| Cyclic comparison as the final tie-breaker            | yes    | yes          |
| `(key, pos)`-stable order across paths                | yes    | yes          |

## What is genuinely new in rcbwt1.cpp

### 1. Doubled buffer `inp2 = inp || inp` used throughout key building

`rcbwt1.cpp` allocates a `2n`-byte buffer up front and reads
`inp2[k]`, `inp2[k-1]`, `inp2[k-2]` straight — no modulo, no wrap branch.
`rcbwt_sparse.cpp` keeps the single `inp` buffer and recomputes
`kk = (k < n) ? k : k - n` plus a similar pair of conditionals for the two
preceding bytes on every inner iteration.

For book1 the inner loop runs ~12.5 M times, so removing 3 branches per
iteration is a measurable win. The cost is +n bytes peak memory (and one
extra `memcpy`).

### 2. The theoretical caveat about recursive key extension

This is the most interesting difference. The plan (§6.5) says: on a tie,
"compute a second-level key = next 48 bits of the same code stream, resumed
from the offset `cap[j]`". `rcbwt1.cpp` includes a long comment explaining
why a naive recursive extension is **unsafe**:

> "the divergent symbol may have already been consumed by the prior level
> (contributing only to the truncated low bits), so the extension would
> resume under *different* (q,p) contexts on the two sides, breaking the
> affine-map argument."

In other words: when level-1 stops because `span < 2^16`, the last few
symbols folded into `code` only contributed to bits that got truncated
away by `code >> 16`. Two positions that tie on the kept 48 bits may still
have diverged inside those discarded low bits; their level-2 resumption
offsets are therefore *not* guaranteed to be equal, and that breaks the
"both tails are produced by the same affine map" argument that makes the
code order-preserving in the first place.

Neither file actually does recursive extension — both fall back to cyclic
`memcmp` for ties. But `rcbwt1.cpp` plumbs the future-safe form anyway:

- `run_code(..., start_offset, out_cap)` returns the number of symbols
  consumed by this level (`*out_cap`), which is what `cap[j]` would be.
- `build_ext_key(..., start_offset)` is a wrapper that produces a level-2+
  key from a given offset.
- `K_MAX_LEVELS = 4` is declared as a placeholder.

These pieces are defined but not invoked anywhere — they document the
intended shape of a future correct implementation. `rcbwt_sparse.cpp` does
not address this caveat at all.

### 3. `cum_at(q, p, c)` with explicit edge cases

```c
static inline uint64_t cum_at(uint8_t q, uint8_t p, int c, ...) {
    if (c == 0)   return 0;
    if (c == 256) return ((uint64_t)1 << 32);
    ...
}
```

Returns `uint64_t` and handles both endpoints explicitly. `r = hi - lo` is
then computed in 64-bit before the multiply.

`rcbwt_sparse.cpp` instead relies on a uint32 wrap trick: it stores
`cum[CNUM] = 0` (i.e. 2³² mod 2³² = 0), and computes `r = hi - lo` in
32-bit arithmetic so the subtraction is modularly correct for `c = 255`.
The wrap version is one less branch per lookup but takes more care to
reason about; `cum_at` is the more readable form.

### 4. Defensive guards in the code-build loop

```c
if (r == 0)    r = 1;       // safety: never zero the span
...
if (span == 0) span = 1;
```

These are belt-and-braces — neither condition should actually occur given
`normalize_row`'s floor-of-1 path. `rcbwt_sparse.cpp` omits them and
trusts the invariants. Useful when prototyping a new model, harmless in
production.

### 5. `cswap_kp` — a branchless `(key, pos)` conditional swap

Defined in `rcbwt1.cpp` but never used (the sort still calls `std::sort`,
with a `(void)cswap_kp` to silence the unused-function warning). It
documents the SIMD-friendly comparator primitive from the plan (a bitonic
network leaf), even though the demo doesn't switch to a network sort.

### 6. Sort, resolve-ties, and emit are separate passes

```c
slab_sort(k, p, m);                   // sort by (key, then pos)
resolve_ties(k, p, m, inp2, n, ...);  // find equal-key runs, re-sort each via cmp_cyclic
emit_block(p, m, out, ...);           // write BWT bytes for this block
```

`rcbwt_sparse.cpp` fuses these into one `sort_and_emit` whose comparator
is `key < cmp_cyclic < pos`. The two designs are equivalent in the limit
but have different constant factors:

- `rcbwt1`: the inner `std::sort` comparator only compares `(key, pos)`
  pairs; `cmp_cyclic` is invoked only between members of an already-equal-
  key group. For sparse-tie inputs (typical text), most `std::sort`
  comparator calls never touch the cyclic compare.
- `rcbwt_sparse`: every `std::sort` comparator call *could* enter
  `cmp_cyclic` — but only when keys are equal, and `==` is checked first,
  so in practice the cyclic path is also rare.

In our run on book1 both paths end up with `cmp_cyclic_calls ≈ 7K` (out of
~768 K positions), so the constant-factor difference is small on real
inputs. The `rcbwt1` arrangement is structurally cleaner for porting to a
real bitonic-network sort that doesn't accept a custom comparator at all.

### 7. `dumb_bwt` uses the doubled buffer

`rcbwt1.cpp`'s reference path memcmps cyclic rotations against `inp2`
(doubled). This is what `rcbwt_sparse.cpp` did originally, but the current
`rcbwt.cpp` switched to `cmp_cyclic` with three chained memcmps on a single
buffer. Same conventions, same answer.

## Engineering details that go the other way

A few decisions in `rcbwt_sparse.cpp` are tighter than `rcbwt1.cpp`:

- **Static slab buffer + pre-histogram fallback.** `rcbwt1.cpp` keeps the
  realloc-on-growth slab buffer (the original plan §6.2 form). The current
  `rcbwt.cpp` / `rcbwt_sparse.cpp` use a fixed-size static `g_slab` and
  pre-histogram each bucket's slab sizes; if any slab would overflow, a
  single `malloc(m * sizeof(KP))` sorts the whole bucket. Zero reallocs on
  the hot path.
- **Streaming output to `FILE*`.** `rcbwt1.cpp` keeps the in-memory
  `bwt_out` buffer (1 n bytes); `rcbwt.cpp` streams bytes via `fputc` and
  appends pi at the end, never allocating `bwt_out` at all.
- **Static globals for fixed-size tables** (`g_ctx2_off`, `g_bbase`,
  `g_cursor`, etc.) — `rcbwt1.cpp` allocates these via `malloc` inside
  `rcbwt()`/`build_models()` and frees on exit. Same lifetime, more heap
  churn.
- **Algorithm/memory stats printout** — only in `rcbwt.cpp` /
  `rcbwt_sparse.cpp`.

## Summary

The genuinely new contribution in `rcbwt1.cpp` is **the documented reason
not to do recursive key extension naively**: when level-1 truncation
discards bits below the divergence point, the level-2 resumption offsets
are not guaranteed to match across tied items, so the affine-map
order-preservation argument from RC_BWT.md §2.2 stops applying. This is
a subtle point absent from the plan, and the file's `run_code` /
`build_ext_key` / `K_MAX_LEVELS` plumbing is shaped to support a future
*safe* recursive scheme (one that detects the truncation hazard and only
recurses when the divergent symbol is guaranteed to be in the kept bits).

The other differences — doubled buffer in the hot loop, explicit `c==256`
handling in `cum_at`, separated sort/tie/emit passes, the `cswap_kp` SIMD
primitive shown as a reference — are useful presentational and engineering
choices, but they don't change the algorithm or its output.
