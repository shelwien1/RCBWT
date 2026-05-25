# Entropy-Coded Suffix Keys for Streaming BWT Construction — Analysis

*Analysis of the `rcbwt3_0` idea: pack many input symbols into fixed-width "substring-code"
buckets with O(1) order-N statistics, then multi-pass-scan those buckets to emit the BWT
output in sorted chunks.*

> **Revision note.** An earlier draft of §2 claimed the order-2 code could not be an
> order-preserving sort key and that one therefore had to drop to an order-0 / alphabetic code.
> That was wrong, and the error was treating the scalar code value as the *whole* sort key. The
> code is a *within-context refinement*; the real key is composite (pinned context + code), and a
> static arithmetic code is order-preserving once comparison happens within a pinned context. §2
> is rewritten around that, with measurements.

---

## 0. What the prototype actually does (verified against `rcbwt.cpp`)

`rcbwt.cpp` does **not** build a BWT. It is a *density probe* for stage 1 only. Concretely:

1. It collects a **static reverse order-2** model: `o2[q][p][c]` counts symbol `c` seen with the
   two *following* bytes `p = inp[i+1]`, `q = inp[i+2]`, over the whole file. Cumulative-frequency
   tables are normalised to a 32-bit scale with `div64`. The model is static (built once), not
   adaptive.
2. It computes a **nested arithmetic code** per position by folding each symbol's interval into
   the code already accumulated for the neighbour:

   ```
   l = cumfreq(c-1 | q,p);  h = cumfreq(c | q,p);  r = h - l;
   code = l + mul64(code, r);          // mul64(a,b) = (uint64)a*b >> 32
   ```

   `code` is a 32-bit fixed-point number in `[0,1)`, computed in **O(1) per symbol** (one multiply,
   one add). This is exactly the "adjacent symbol's interval encoded into the current symbol's in
   one step" primitive. Because the loop runs left-to-right and makes the *current* symbol the most
   significant, `code[j]` ranks the reversed prefix `T(j) = inp[j], inp[j-1], …, inp[0]`.
3. It tracks ideal code length with a `-log2(p)` table `cl[]` and, with a sliding tail pointer
   `cj`, reports `mcl[j] = ci - cj` — **how many consecutive symbols' entropy fits inside one
   32-bit bucket** (plus a per-context granularity slack `q2[q][p]`). The sliding window is
   amortised O(1), so the whole density map is O(n).
4. The decode-verification loop confirms the code is invertible *position-by-position* given the
   context. The `!!!` markers in `result.txt` flag positions where 32-bit precision ran out before
   `mcl` symbols could be reproduced.

**Measured on the bundled 8 191-byte BOOK1 slice:**

| quantity | value |
|---|---|
| ideal size `CL` | 19 283 bits for 8 189 symbols → **2.355 bit/sym** (this is H2) |
| density (forward window) | min 1, **mean ≈ 12.3**, max 53 symbols per 32-bit code |
| density shape | bell around 11–13, long tail to 53 in near-deterministic runs |
| conditional entropies | **H0 = 4.466**, **H1 = 3.431**, **H2 = 2.352** bit/sym |

So the *first* claim — "an order-2 code packs ~12 symbols into a 32-bit bucket in O(1)" — is
empirically solid. Everything below concerns stage 2 (using these codes as BWT sort keys).

---

## 1. The core idea, stated precisely

Let `S[i]` be the suffix (or reversed prefix) ranked at position `i`. Define a sort key from the
nested code, B bits wide (B = 32 or 64). The pipeline is:

1. **Stage 1 (O(n), done):** compute the code for every position by one-step interval nesting.
2. **Stage 2 (proposed):** produce the BWT permutation by emitting positions in increasing key
   order, in chunks of `N`. For each chunk: scan all buckets, collect the `N` smallest keys above
   a running `low_bound`, sort those `N`, emit the corresponding BWT symbols (`inp[pos-1]`), set
   `low_bound = max(chunk)+1`, repeat until exhausted.
3. **Stage 3:** entropy-code the BWT bytes — *concurrently* with stage 2, because chunks are
   produced in final output order.

The bet is that scanning + selection can be made branchless / SIMD / cache-friendly and parallel,
so the constant-factor advantage beats the asymptotically-better `O(n log n)` suffix machinery on
realistic blocks, while also unlocking the stage-2/stage-3 overlap.

Three things must hold up: the key must actually sort (§2), ties must be cheap to resolve (§3),
and the selection must be the claimed branchless kernel (§4).

---

## 2. The sort key: order preservation via a composite (pinned-context + code) key

### 2.1 The scalar code alone does **not** sort; the composite key does

The whole scheme needs **integer order of the key = lexicographic order**. The *scalar* code does
not satisfy this, and it is worth seeing exactly why, because the fix follows directly.

The interval `[l,h)` for the most-significant symbol of each key depends on that key's context —
in the prototype, the two *following* bytes, which lie **outside** the key (they are not part of
`T(j)`). Two keys whose leading symbols differ then draw their leading intervals from two
unrelated context partitions, which can interleave. Measured on BOOK1 by sorting every position by
the scalar code and comparing against the true order of the strings the codes represent:

| sort key | strict inversions | of which at the leading symbol | code-equal ties |
|---|---|---|---|
| scalar **order-2** code (prototype) | 3880 | **3496** | 6 |
| scalar **order-0** code | 1 | 0 | 474 |
| **composite `(inp[j], inp[j-1], code[j-2])`** | **0** | **0** | **6** |

A concrete inversion from the order-2 run: position 7988 has `inp='b'` (0x62) coded in
right-context `e d` → code `0x0018AA55`; position 3102 has `inp=','` (0x2C) in context `· t` →
code `0x001A7753`. `'b' > ','` as bytes, yet `code[7988] < code[3102]`, so the scalar code orders
`b…` before `,…` — not lexicographic. The inversions concentrate at the leading symbol (3496 of
3880); the rest are cascade once a key lands in the wrong region.

The resolution (this corrects the earlier draft): **the code is a within-context refinement, not
the key.** Within a *fixed* context, arithmetic intervals are perfectly order-preserving — symbol
`X < Y` gives disjoint, correctly-ordered intervals, and any codes inside them compare correctly.
So make the key a composite: **pin the leading context, then compare the code.** Concretely, pin
the two most-significant symbols and use the order-2 code of the *tail* (which starts at the
third symbol):

```
key(j) = ( inp[j],  inp[j-1],  code[j-2] )      // compared most-significant first
```

Sorting BOOK1 by this composite key gives **0 inversions** and only **6 ties** — i.e. exactly the
full order-2 resolution depth (the raw context prefix alone would leave 7567 adjacent ties; the
code tail collapses them to 6). Order preservation *and* order-2 depth, simultaneously — the thing
the earlier draft wrongly said was impossible.

### 2.2 Why it is order-preserving — the affine-map argument

This is provable, not just empirical. When two keys have `inp[j]` and `inp[j-1]` pinned equal, the
tail `code[j-2]` codes `inp[j-2]` in context `(inp[j-1], inp[j])` — which is now *identical* for
both keys. So both tails are produced by the **same affine map** `code = l + r·(·)` with the same
`l, r`. An order-preserving map applied to both operands preserves their order, and the recursion
repeats at every deeper level: each new context symbol is pinned equal *before* the comparison
reaches the symbol that depends on it. The only thing finite precision can produce is equal codes
(ties), never an inversion. Hence 0 inversions by construction; truncation shows up purely as the
6 ties.

### 2.3 The general rule: a context-order ramp

The pin width is not arbitrary. At significance level `k` (1 = most significant), only `k−1`
more-significant symbols are pinned by the time the comparison reaches level `k`, so the maximum
*order-preserving* context available there is `min(k−1, K)` for a model of order `K`. The
prototype's defect is using order-2 at levels 1 and 2, where only order-0 and order-1 are legal.
The correct construction **ramps the order**:

```
level 1: order-0     (no context — it is the MSB)
level 2: order-1     (context = the single pinned level-1 symbol)
level 3+: order-K    (context fully pinned)         e.g. order-2 from level 3 on
```

This is precisely "1 unpacked, 1 order-1, rest order-2". Note this needs no special alphabetic
(Hu–Tucker) code: plain arithmetic coding is order-preserving here because the ramp guarantees
each symbol is coded in a context that is pinned-equal before it matters. (The earlier draft's
alphabetic-code recommendation was therefore unnecessary.)

### 2.4 Three ways to realise the pinned prefix, and their costs

All three produce a single comparable key; they differ in bits spent and cache footprint.

* **Raw prefix** — store the leading 1–2 symbols as literal bytes, then the order-K code tail.
  Simplest; the key is a flat integer (e.g. 16 bits context + 48 bits code in a 64-bit word). For
  an order-2 model you need **2** raw symbols, not 1 — pinning one leaves the level-2 symbol's
  context with an unpinned future neighbour. Wastes bits on common leading symbols (8 flat bits
  each).
* **Order ramp** (§2.3) — entropy-code the leading symbols at their maximal *legal* orders (H0
  then H1) instead of spending raw bytes. Bit-optimal, still one packed integer, tiny tables.
* **Remap table** — keep the uniform order-K code and, at key-construction time, map the leading
  k-symbol prefix to its true global rank via a precomputed table, e.g. a 2-symbol-prefix →
  global-base table. **This is fully branchless and SIMD-able** — a gather (`VPGATHERDQ`) plus an
  add, no data-dependent branches. Its cost is **footprint**: 256×256 entries × 8 B ≈ **512 KB**,
  which overflows L1 and occupies a large fraction of L2, so you pay gather latency from L2/memory
  per element. Attractive only when storing extra prefix bits is the binding constraint.

**Depth comparison (measured, BOOK1, 64-bit key).** Symbols packed per key, i.e. resolution depth:

| scheme | symbols / 64-bit key |
|---|---|
| 0 raw + order-0 throughout | 14.3 |
| 1 raw + order-1 tail | 17.3 |
| 2 raw + order-2 tail | **22.4** |
| order-2 with ramp (order-0/1/2) | **25.9** |

So spending bits to *keep* order-2 pays off: even paying 16 flat bits for two raw prefix bytes,
order-2 reaches ~22 symbols vs order-0's ~14 (a ~57% deeper key), because the H2 tail is ~2.1
bit/sym tighter than H0 and more than repays the prefix. The ramp recovers most of the prefix
overhead (25.9). The one caveat: per-suffix code length only crosses over in order-2's favour
around `L ≈ 7–8` symbols (below that the prefix is not yet amortised), but typical depth is 14–26,
comfortably past it.

A practical middle ground: **raw first byte, ramp/code thereafter.** A literal first byte lets the
early MSD-radix passes (§4) bucket directly on actual symbol bytes — gather-free and cache-friendly
— before switching to the coded tail. Pure ramp is denser; raw-first integrates better with the
distribution pass.

### 2.5 Invertibility is not an issue

Once the key reproduces true lexicographic order, it is **only a sort accelerator** — it never
enters the transform definition. The resulting permutation *is* the genuine BWT, invertible by the
standard O(n) LF mapping. (The earlier draft's invertibility worry was downstream of the
order-preservation mistake and evaporates with it.)

---

## 3. Problem 1 — equal keys (ties) and per-bucket capacity

A correctly-ordered, B-bit-truncated key produces **ties, never inversions**. Equal keys mean the
two suffixes share a prefix that fills ≥ B bits. Three questions: how often, how to resolve, and
how to find the skip length cheaply.

### 3.1 How often — the birthday bound, and why 32 bits is for toys only

For `n` suffixes and B-bit near-uniform keys, expected colliding pairs ≈ `n²/2^(B+1)`:

| n | B = 32 | B = 64 |
|---|---|---|
| 2^16 (64 K) | ≈ 1 pair | ~0 |
| 2^24 (16 M) | key space *saturates* → massive ties | ~10⁻⁵ |
| 2^30 (1 G) | hopeless | ≈ 0.03 pairs |

So **32-bit keys are usable only up to ~10⁵ suffixes**; any real BWT block needs a **64-bit** key.
Worse, real text violates uniformity *upward*: repeated phrases create genuinely equal long
prefixes, so tie groups cluster above the random estimate — which only strengthens the case for a
deep key (more entropy packed ⇒ ties pushed deeper and rarer).

### 3.2 How to resolve — recursive key extension, not string compare

A tie group shares the same first B bits *and the same symbol prefix*. Resolve it the way
prefix-doubling / DC3 suffix sorts refine equal groups:

1. For each member, compute a **second-level key** = the next B bits of the *same* code stream,
   resuming the packing from where the first key's bits ended (see §3.3 for that offset).
2. Sub-sort the group by the level-2 key (same branchless machinery, §4).
3. Recurse on any still-equal subgroups.

Everything stays in the integer-key domain (SIMD-friendly) and touches only tie-group members;
each level resolves another ~B bits ≈ another ~14–26 symbols, so even pathological repeats settle
in a couple of levels. A byte-wise `strcmp` is the safety net for truly periodic data.

### 3.3 The skip length — and why tied members share it

The worry "how do I measure each bucket's capacity in symbols without O(n log n)?" resolves
cleanly:

* **It is O(n) by a forward sliding window.** Define `cap[i]` = number of symbols from `i` whose
  per-symbol code lengths sum to ≥ B bits. Maintain a running prefix-sum of code lengths (one
  `cl[]` lookup per symbol) and advance a window head — exactly the prototype's `cl[]` + sliding
  pointer, run forward. Amortised O(1) per position, O(n) total, no sort, no log factor.
* **Tied members share `cap`.** Equal B-bit keys ⇒ identical decoded symbol prefixes to the
  resolution depth ⇒ identical per-symbol code lengths there ⇒ identical `cap`. So the resume
  offset for level-2 extension is one number per group, computed once. The per-position variation
  in `cap` across the array is benign — each position carries its own `cap[i]`.

### 3.4 Deferring tie resolution out of the scan (so the hot loop stays branchless)

A key simplification: **resolve no ties during scanning.** The scan only collects codes; tie
handling happens in the post-scan chunk sort, on a small, *contained* set — keeping the hot loop
(§4) free of string comparisons.

This is sound because of where ties can land relative to a chunk's max `M`:

* A tie group with value **strictly below `M`** is **fully contained** in the chunk. The
  collection holds the `N` smallest keys above `low_bound`, so anything below its maximum is
  necessarily included in full. Such groups are resolved locally during the chunk sort (§3.2),
  with no cross-chunk dependency.
* The only group that can **straddle** the chunk boundary is the one **at value `M`**: the
  count-bounded collection may admit some `M`-valued codes and evict others. Detect this (track
  whether any code equal to the final `M` was evicted) and simply **shrink the chunk** — drop the
  whole `M`-group, set the boundary to the largest value `< M` (which is fully contained), and
  defer the `M`-group intact to the next pass, where it sits at the low end and is collected in
  full.

Deferring works unless the `M`-group alone exceeds `N`. So the entire tie problem reduces to a
single residual hard case: **a collision group larger than the chunk size** (equivalently, the
chunk fills with all-equal codes). That is exactly where §3.2's recursive key extension is
invoked — on that one oversized group — and nowhere else.

(Under the *value-based* slab partitioning of §4.2 the straddle cannot even occur: a value
boundary never splits an equal-value group. There, an oversized group instead overflows one slab —
the same residual hard case, reached differently.)

---

## 4. Problem 2 — the branchless / SIMD / cache-friendly selection

### 4.1 Why "unsorted top-N with eviction" is a trap

Maintaining the maximum of an *unsorted* bag under deletion of that maximum is the one thing you
cannot do cheaply: evicting the max leaves the new max unknown without an O(N) rescan. Only partial
orders support cheap insert+delete-max — a **heap** (O(log N), branchy, pointer-chasing, anti-SIMD)
or a fully sorted buffer (O(N) shifts). So branchless *and* unsorted *and* cheap eviction cannot
coexist; something must give.

### 4.2 The right abstraction: uniform keys ⇒ fixed slab bounds ⇒ stream compaction

An entropy-coded key is **near-uniform on `[0, 2^B)` by construction**, so you need not discover
the chunk boundary adaptively — fix the boundaries a priori as equal divisions of the key space.
Partition into `n/N` slabs of width `w = 2^B·N/n`; by uniformity each holds ≈ N keys (Poisson,
mean N). Each pass then collects one slab with a **branchless range filter (stream compaction)** —
no heap, no eviction, no max-tracking:

```c
size_t cnt = 0;
for (size_t i = 0; i < n; i++) {
    uint64_t k = K[i];
    int take = (k >= lo) & (k < hi);     // 0/1, no branch
    out_key[cnt] = k; out_pos[cnt] = i;  // speculative store
    cnt += take;                         // advance only if taken
}
```

On AVX-512 this is a `VPCMP` mask + `VPCOMPRESSQ` (`_mm512_mask_compressstoreu_epi64`) — the
textbook compaction kernel; AVX2 does it via shuffle-LUT. Reads are sequential and prefetchable;
writes bounded by slab size; no data-dependent branches. Then sort the ≈N collected keys with a
**bitonic network** (small N) or a sub-radix (larger N), resolve contained ties (§3.4), and emit
`inp[pos-1]`. This is the "sort the chunk only at the end" the prompt gestures at — realised as
*filter-then-sort-the-slab*, strictly better than incremental insertion.

So stage 2 is **MSD radix on the top bits of an entropy key, with a SIMD network per slab.** The
selection is branchless (a range predicate), SIMD (compaction), and cache-friendly (a linear
stream with localised output).

### 4.3 What entropy-keying buys over plain prefix-radix sort

A naive MSD radix on 4 raw bytes resolves suffixes only 4 symbols deep, so on text the top buckets
are huge and recursion deep. The order-preserving entropy key spends its B bits proportionally to
information, resolving ~14–26 symbols in 64 bits (measured §2.4). Deeper per-bit resolution ⇒
flatter, more even bucket occupancy ⇒ fewer recursion levels ⇒ fewer ties. The same uniformity
that validates the a-priori slab boundaries makes the buckets balanced. That is the core win.

### 4.4 Handling non-ideal occupancy

* **Overflow (> N):** size slab buffers at ~2N (Poisson tails are thin), or split an overfull slab
  by recursing on the next B bits — the same kernel one level down.
* **Underflow / empty:** merge adjacent slabs, or just accept variable chunk sizes; nothing
  downstream needs exactly N.
* **The one genuinely hard overflow** is a single value with more occurrences than the slab budget
  (the §3.4 residual case): a collision group bigger than N. Splitting on the *next* B bits does
  nothing (the top bits are all equal) — this group needs **recursive key extension** (§3.2), i.e.
  a deeper code, applied to just that group.

---

## 5. Complexity, scaling, and where this wins

Let `P = ⌈n/N⌉` passes, each an O(n) scan plus an O(N log N) (or network) slab sort:

```
T = Θ( n²/N           +    n·log N )
        └ scanning ┘        └ slab sorts ┘
```

* The scan term `n²/N` is the catch. Keeping it ≤ a comparison sort's `n log n` needs `N ≥ n/log n`
  — but then a slab is a large fraction of the array, not cache-resident, defeating the premise.
* Sizing slabs to fit L2 (N·8 B ≤ 256 KB → N ≤ 32 K) gives, for n = 128 M keys (1 GB / 8 B),
  P ≈ 4 000 passes and ~5×10¹¹ key-reads — **infeasible**. Pure multi-pass is quadratic; sane only
  for moderate n.

The fix is the standard **hierarchical radix**:

1. **One** distribution pass over the top `b` bits into `2^b` buckets (software write-combining /
   streaming stores to tame the single unavoidable scatter), sized so each bucket fits in cache.
2. Within each cache-resident bucket, apply the branchless multi-pass slab scan of §4 — now
   `n_bucket/N` is tiny, so `n²/N` is harmless and every pass is L2-local.
3. Recursive key extension (§3.2) only inside oversized tie groups.

So, honestly: **the streaming multi-pass scan is an excellent *inner kernel* on already-localised
data and for moderate blocks; it is not, by itself, a full large-block suffix sort.** With the
64-bit-key requirement (§3.1), the sweet spot is per-thread cache-resident sub-blocks, 64-bit
ramped order-2 keys, SIMD compaction inner loop, recursive extension for ties.

### 5.1 The parallelism is genuine, on three axes

* **Across slabs/buckets:** independent once partitioned → embarrassingly parallel, emitting in
  disjoint key ranges so output order is automatic.
* **Within a pass:** range-partition the input across threads, compact each share, prefix-sum the
  counts. Standard parallel filter.
* **Stage 2 ∥ Stage 3 (the headline benefit):** slabs are processed low-key → high-key, so BWT
  bytes are produced *in final order, incrementally*. The downstream stage (MTF+RLE+EC, or a
  low-order CM) can consume chunk 0 while the sorter is still on chunk k. A monolithic
  suffix-array → BWT → encode pipeline must finish the whole (multi-GB) sort before the encoder
  sees byte 0. This latency/throughput overlap is independent of asymptotics and is the most
  compelling architectural argument for the approach.

---

## 6. Summary judgement and recommended path

What holds up:

* **Stage 1 is sound and demonstrated.** O(1) interval nesting packs ~12 symbols per 32-bit code
  on BOOK1; the density map is O(n) via the sliding-window `cl[]` trick.
* **The key sorts correctly** as a composite (pinned context + code): 0 inversions on BOOK1, full
  order-2 depth (6 ties). Order-2 with a ramp packs ~26 symbols per 64-bit key — deeper than
  order-1 (17) or order-0 (14) even after the order-preservation overhead. Plain arithmetic coding
  suffices; no alphabetic code, and the BWT inverts normally.
* **Ties have a clean answer:** O(n) forward-window `cap[i]`; resolution deferred out of the scan
  to the contained post-scan chunk sort (§3.4); recursive key extension only for groups larger
  than a chunk.
* **The selection is the claimed kernel:** fixed-slab range-filter (SIMD stream compaction) on a
  near-uniform key — branchless, vectorisable, cache-local, parallel.
* **The pipelining win (stage 2 ∥ stage 3) is real and is the strongest reason to build this.**

What to get right:

1. **Ramp the context order** at the start of each key: order-0, order-1, …, order-(K−1), then
   order-K. Using full order-K at the leading levels is the prototype's only ordering bug.
2. **Use 64-bit keys.** 32-bit collisions saturate by ~10⁵–10⁶ suffixes.
3. **Make the scan an inner kernel under one MSD distribution pass.** Pure multi-pass is `n²/N`,
   competitive only on cache-resident sub-blocks.

### Concrete next experiments (cheap, decisive)

1. **Composite-key SA check at scale:** build the ramped order-2 key, integer-sort, and diff
   against a reference suffix array on a few MB (not just 8 KB). Confirm 0 inversions and record
   the tie-group-size histogram vs key width B and model order K.
2. **Ramp vs raw-prefix vs remap-table:** measure depth (symbols/key) and the SIMD scan throughput
   for each prefix strategy, including the 512 KB remap table's effect on L2 miss rate.
3. **Kernel microbench:** AVX-512 `VPCOMPRESSQ` slab-filter throughput (keys/s) vs scalar and vs
   `std::sort`-on-the-block, on one cache-resident sub-block — the constant-factor thesis.
4. **Pipeline latency demo:** feed chunked BWT output into a trivial MTF+RLE0 encoder and measure
   time-to-first-output-byte vs a finish-the-whole-sort baseline — the headline claim, easy to make
   visible.

The idea is worth pursuing **as a streaming, SIMD radix on an order-preserving (ramped order-2)
entropy key, used as a cache-resident kernel and pipelined with the entropy stage** — not as a
drop-in large-block suffix-array replacement.
