# Step 6a findings: the code step 6 needs before any GPU is booked

Done, no GPU used. All four gaps the audit found (parent plan §0, step file §1-4)
are closed:

1. Per-k validation loss (`train_adapter.py`).
2. Position means recomputed from vectors and pooled with equal weight per k
   (`dataset.py`, D13/D14).
3. Pooled centring in the retrieval eval, across several extraction
   directories in one invocation (`evaluate_retrieval.py`).
4. The mmap-based memory-bounded vector path for the grouped mixture
   (`grouped_examples.py`), replacing the `torch.cat`-concatenated table
   `step3_example_builder.md` §3 flagged.

251 non-`hf_cache`/non-`gpu_exec` tests pass (`pytest tests/ -m "not hf_cache
and not gpu_exec"`). `--help` stays fast for both `train_adapter.py` and
`evaluate_retrieval.py` (~30ms each, no torch import).

## 1. Per-k validation loss

`build_mixture`'s val call already computed `val_k_ranges`; only the train
call's ranges reached anywhere before (`run_config.json`). `load_grouped_train_and_val`
now returns both `train_k_ranges` and `val_k_ranges`, `train()` takes
`val_k_ranges` and, in the final full-val pass, scores each k's own slice
with the same `evaluate()` the whole-mixture number already used, writing
`final_eval.json["val_loss_by_k"] = {"1": {...}, "2": {...}, "3": {...}}`
beside the existing `measured_loss` key (left untouched, so nothing
downstream that reads the whole-mixture number breaks).

## 2. Position means: recomputed, pooled, equally weighted

`dataset.compute_position_means(directory, records)` streams a per-position
mean from `vectors.pt` (mmap'd, so only the rows `records` addresses are
touched) instead of trusting the directory's stored `position_means.pt` --
closing the mismatch where the k=1 `;`-filter (9 topics dropped from
`outputs/bg_think_l19`) would otherwise weight an *unfiltered* stored mean by
a *filtered* record count.

`pooled_position_means` now calls this per source and averages the three
results with **equal weight**, not by vector count (D14). Verified by hand:
at position 0 the real directories are 470,010 / 467,990 / 774,330 rows
(k=1/2/3) -- count-weighting would have put the origin closest to k=3 (the
same imbalance D14's rationale describes); equal weighting treats the three
as one vote each regardless.

The existing `test_pooled_means_equal_the_mean_over_every_directorys_vectors`
test needed no change (its fixture already had `vectors == means`, so
recomputation reproduces the same numbers by construction). The old
count-weighted test (`test_pooled_means_weight_positions_by_how_many_records_reached_them`)
tested behaviour that no longer exists; replaced with two tests that pin the
new contract directly: recomputation from vectors (not the stored file), and
equal weighting across sources whose vector counts differ sharply (10 vs 1).

## 3. The mmap-based memory-bounded mixture path

`_load_mixture_store` no longer builds a concatenated fp32 table. Each
source directory's `vectors.pt` is `torch.load(..., mmap=True)` and kept
bf16; three small `int64` index tensors (`directory_of`, `local_index_of`,
`position_of`, one entry per global row, ~14 MB each at the real k=1/2/3
scale) map a global `vector_index` to `(source tensor, local row, position)`.
A new `_MixtureVectors` object (`VectorStore.vectors`, drop-in for the plain
tensor every caller already indexes with `store.vectors[[i, j, ...]]` or
`store.vectors[i, 0]`) does the fp32 cast and the pooled-mean subtraction
*only on the rows a batch actually gathers*.

**Measured against the real extraction directories**
(`outputs/bg_think_l19` + `outputs/bg_think_many_l19_k2` +
`outputs/bg_think_many_l19_k3`, 1,712,330 rows total, 4096 hidden, 13.4 GB of
`vectors.pt` combined, on a 30 GB machine):

- `_load_mixture_store` (pooled-mean computation + building the index
  tensors, no vector data materialised beyond what `compute_position_means`
  streams): 92-110s per call, peak RSS **6.3-7.3 GB** (two separate
  measurement runs; page-cache pressure on this 30 GB machine, 2.2 GB free
  at the time, likely accounts for the spread) -- against the ~52-78 GiB the
  `torch.cat` design would have needed on train+val combined, which this
  machine does not have.
- One 256-row batch gather (fp32 cast + pooled-centre subtraction, the
  per-step cost during training): **34ms**, no measurable RSS growth.
- A second `_load_mixture_store` call in the same process (simulating
  `load_grouped_train_and_val`'s separate train-split and val-split calls):
  91.5s then 120.9s (both I/O-bound on this machine's page cache under
  memory pressure -- 2.2 GB free at the time of the first measurement -- so
  the second call was not meaningfully faster from warm cache), but **peak
  RSS only grew from 7.3 GB to 7.5 GB with both stores held live** --
  confirming the claim below that mmap'd double-loading costs time, not host
  memory.

**One deviation from the step file's letter, not its spirit.** §4 says "let
train and val share one set of mappings instead of each building a table."
This implementation does not do that: `build_mixture` still calls
`_load_mixture_store` once per split, as it did before, so each split builds
its own `_MixtureVectors` (its own mmap handles onto the same files, its own
small index tensors). The reason: with `torch.cat` gone, the actual cost the
sentence was written to avoid -- doubling a ~26 GiB concatenated table --
no longer exists. `torch.load(..., mmap=True)` on the same file twice maps
the same page-cache-backed pages both times (measured above: peak RSS grew
by only ~200 MB, not the ~7 GB a second independent load would cost without
mmap), so the two calls cost roughly double the time (~92s then ~121s,
both I/O-bound rather than compute-bound) for the second pooled-mean
recomputation, not extra host memory. Sharing the mapping object
between train and val would remove that redundant computation but adds
plumbing (both splits would need to be built from the same call, and
`build_mixture`'s existing per-split interface -- and its existing tests --
would have to change to expose that). Given the memory goal is already met,
this was judged not worth doing inside step 6a; flagging it here rather than
silently declaring the step file's letter satisfied. If step 6's real run
finds the ~110s x 2 pooled-mean recomputation is a meaningful fraction of
total run time, revisit.

Centring math is unchanged from the pre-mmap design: `_gather` casts the
mmap'd bf16 rows to fp32 and subtracts the pooled fp32 mean, bit-identical to
casting the whole (bf16-backed) table up front and then indexing it, since
the cast is a pure upcast either way.

`tests/test_grouped_examples.py`'s `build_mixture` tests needed updating: the
fixture's `position_means.pt` files are all zero, which used to make
centring a no-op (the code trusted the stored file). Recomputing from the
fixture's own (nonzero, per-k, per-split constant) vectors makes centring
*not* a no-op any more -- this is D13 working as intended, not a bug, and the
tests now assert the actual pooled-centred values (documented in the
fixture's own docstring) instead of the raw constants.

## 4. Pooled centring in the retrieval eval

`evaluate_retrieval.py` gains `--vectors-k K=DIR` (repeatable, mirrors
`train_adapter.py`'s own flag) as an alternative to `--vectors`, mutually
exclusive with it and with `--grouped` (meaningless there -- `--vectors-k`
always scores by set-level recall, per k, bridging a `topics.json` directory
into a one-topic group with `group_record_from_topic` +
`drop_semicolon_topics`, the same bridge `grouped_examples._load_mixture_store`
uses for k=1).

Step 6 §4's two runs need two different populations to define the pooled
reference: run 1 queries only `outputs/bg_think_l19` but must still be
centred against the pool of all three k's (the condition the adapter
actually trained under), while run 2 queries and pools over the same three.
A single `--vectors-k` couldn't express "query fewer than you pool", so a
second flag, `--pool-vectors-k K=DIR` (also repeatable), names the pooling
population separately; it defaults to `--vectors-k`'s own directories when
omitted, which is exactly run 2's shape with zero extra flags. Run 1 becomes:

    --vectors-k 1=bg_think_l19 \
    --pool-vectors-k 1=bg_think_l19 --pool-vectors-k 2=bg_think_many_l19_k2 \
    --pool-vectors-k 3=bg_think_many_l19_k3

The index (GTE-large over the full topic corpus) and the model are built
once regardless of how many k's are queried, satisfying the step file's
"three separate passes would not share an index" concern structurally, not
just for run 2.

Tested against hand-built fixtures with a fake `evaluate_grouped_positions`
(`tests/test_retrieval_eval.py`): equal pooling across two directories with
different single vectors, and the "query one k, pool over more" case that
run 1 actually needs, both checked by reading the exact centred value that
reached the (captured) query vectors.

## Anything that contradicted the plan

- The §4 "share one set of mappings" line, addressed in §3 above.
- Nothing else did. D13/D14's arithmetic, the `outputs/bg_think_l19` k=1
  bridge, and the "three separate passes would not share an index" framing
  all matched the real code once written.
