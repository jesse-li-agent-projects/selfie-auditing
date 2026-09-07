# Step 6a: the code step 6 needs before any GPU is booked

Part 6a of 6 in the execution of `plans/bg_think_many/bg_think_many.md`. Read
the parent plan's D12, D13 and D14 first.

Found by an audit of `step6_run_and_report.md` against the code (2026-09-07),
after steps 1-5 had merged. **None of this needs a GPU and none of it is a
hyperparameter change.** It is split out of step 6 so it can be done while the
GPU is busy, and so the expensive step stays a run-and-report step.

Steps 1-5 are merged (#75-#83). Nothing here blocks on another step.

## 1. Per-k validation loss

Gate 2 (step 6 §3) compares the k=1 validation loss against `bg_think`'s 1.4844
and reports k=2 and k=3 beside it. Nothing computes those slices.

`build_mixture` already returns each k's `(start, end)` range into the examples
it produced. `load_grouped_train_and_val` keeps the **train** ranges, which
reach `run_config.json`, and discards the val ones. `final_eval.json` therefore
holds one whole-mixture `measured_loss`.

Keep the val ranges, and have the final evaluation score each k-slice as well
as the whole pool. `evaluate()` already takes an example list, so each slice is
a slice of the list and one more call. Write both into `final_eval.json`, with
the whole-mixture number keeping its current key so nothing downstream breaks.

**This one is recoverable after the fact**, unlike the rest of this file:
`build_mixture` is seeded per k and split (`f"{seed}-k{k}-{split}"`), so the
val pool rebuilds exactly from `run_config.json` and a trained checkpoint can
be scored per-k later. Doing it in the trainer is preferred only because it
avoids re-loading the vectors. Do not let this become a reason to skip it -- the slices are what
tell "the mixture traded single-topic accuracy away" apart from "multi-topic
training does not help" (step 6 §5).

## 2. Position means: recomputed, pooled, equally weighted

Three changes to one area, in `dataset.py`.

**Recompute rather than read** (D13, as amended 2026-09-07).
`pooled_position_means` re-weights each directory's *stored* `position_means.pt`
by the counts of the records passed in, so a filtered record set weights an
unfiltered mean -- which is what the k=1 `;` filter does, for 9 of 47,001
topics. Add a streaming `compute_position_means(directory, records)` that
accumulates per-position fp32 sums over the mmap'd `vectors.pt`. Measured: it
reproduces the largest directory's stored file to 4.8e-7 in **9.6s** on CPU, so
all three cost well under a minute. Keep it on CPU -- the pass is a cast and a
sum, not a matmul, and it runs once.

**Weight the three k equally** (D14). Not by vector count, which is what a
naive pooling does and which at position 0 is 0.274 : 0.273 : 0.452 -- an
artefact of D8's round counts. Within a k, positions keep their own counts.

**Keep the stored-file path** for single-directory callers, so earlier runs and
the `bg_think` comparisons stay reproducible. Only the grouped mixture
recomputes.

`load_vector_store` already accepts a `means` argument and needs no change.

## 3. Pooled centring in the retrieval eval

`evaluate_retrieval.py --center` has no way to pass a reference, so it falls
back to the directory's own `position_means.pt` -- scoring the adapter in a
condition it never trained in. Step 6 §4 needs two runs:

- one over `outputs/bg_think_l19` alone (a `topics.json` directory, not
  `--grouped`)
- one over all three k directories at once, reported per k (D12)

So both the single-directory and the grouped paths need the pooled reference,
and the eval needs to accept several extraction directories in one invocation
so the pooled mean and the retrieval index are built once. Three separate
passes would not share an index and could not produce a single per-k report.

## 4. The memory-bounded vector path

Specified in `step3_example_builder.md` §3; that section is the contract, this
is the pointer. In short: stop building the ~26 GiB concatenated fp32 table
(twice), by mapping the vectors with `torch.load(..., mmap=True)`, keeping them
bf16, and moving the fp32 cast and the centring to the per-batch gather.

Two things reviewers of this get wrong:

- Deferring the fp32 cast is **not** an accuracy trade. `vectors.pt` is bf16 on
  disk, so the current fp32 table is an upcast of bf16 data and the deferred
  values are bit-identical.
- "Centre per batch" means subtracting the one precomputed per-position
  reference at gather time. It does **not** mean computing a mean from the
  batch.

Without this, step 6 needs a machine with ~80 GiB of free host RAM.

## 5. Gates for this step

The unit tests, plus: a run of the existing single-directory training path must
produce the same loss it did before, since §2 and §4 both touch code that path
uses. **Note the project rule: passing unit tests do not mean the codebase
works.**

`--help` must stay fast (the light-imports rule in `CLAUDE.local.md`).

## 6. Notes

`plans/bg_think_many/notes/step6a_code.md`, in the style of the other notes:
what the measured startup time and peak RSS actually were once the mmap path is
in, and anything that contradicted this file.
