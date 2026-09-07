# Step 6 results: bg_think_many

**Status: in progress.** Gate 2 is complete. Gate 3 is running; the OOD
comparisons (§5) have not been run.

## The question

Quoted from the parent plan §1, which quotes the user (2026-09-05):

> Preliminary results have convinced me to pivot somewhat from previous plans
> (plans/archive/pangram_extraction_adapter.md and its execution steps). The
> original idea was to train an adapter on prompts like "think about X while
> writing Y", with the adapter's goal being to reproduce X. However, the
> resulting adapter (outputs/adapters/bg_think) doesn't perform better than the
> baseline adapter (outputs/adapters/wikipedia-scalar-affine.safetensors) on the
> OOD tasks I care about.
>
> The new thing I want to try is having the adapter training prompt involve
> thinking about multiple topics in the background. In other words, training on
> prompts like "think about X1, X2, and X3 while writing Y".

and, on the mixture:

> Off a hunch, I think it's more important that the model learn to verbalize
> multiple topics at once, so let's say (without much justification) that the
> ratio of dataset size for 1,2,3 topics being thought about, should also be
> 1:2:3.

## The run

Hardware is an RTX 5090 (31.36 GiB usable), not the A100 the plan's estimate
assumed. Code is `ffbfdc7` on `worktree-step6-eval-microbatch-fix` (PR #87).

Command as in the plan §2 (corrected -- see "What contradicted the plan"), with
`--micro-batch-size 16` and `--resume`, writing to
`outputs/adapters/bg_think_many/`. 5,902 steps at a measured ~3.1 s/step
including validation.

`--val-total-examples 300000`: the plan §2 records the reasoning and the user's
confirmation (2026-09-07). It sizes the loss pool only, splits 1:2:3 into
50,000 / 100,000 / 150,000, and leaves the k=1 slice at 10x `--val-subsample`,
enough to compare against 1.4844.

## What contradicted the plan

These are the lines the plan asks for most.

### The plan's own §2 command could not run. Three faults.

1. `--vectors-k` values carried an `outputs/` prefix that the script already
   prepends, so it resolved `outputs/outputs/bg_think_l19` and died on
   `FileNotFoundError`. The flag's own `--help` text has it right.
2. `--micro-batch-size` was absent. It defaults to `--batch-size`, so the
   documented command asked for all 256 examples in one forward pass.
3. Nothing recorded what that flag has to be, which on this hardware is not a
   free choice.

Fixed in `ffbfdc7`.

### Micro-batch size is a memory bound set by target length, not a free knob

The run OOMed repeatedly, and the failure looked like a leak or fragmentation.
It was neither: `torch.cuda.memory_allocated()` returned to exactly 16.136 GiB
after every step, of which 14.96 GiB is the frozen 8B weights. The OOM is a
transient spike inside one micro-batch's forward/backward, at `lm_head`
producing a `(micro_batch, max_target_len, vocab)` tensor.

Both that tensor and the model's saved activations scale with `max_target_len`,
and the mixture's composed k=2/k=3 labels reach **77 tokens against a 29-token
median**. `bucketed_batches` sorts by length, so the long targets arrive as one
batch and every micro-batch in it hits the worst case together -- which is why
the crash looked random and late rather than length-driven.

Measured peaks on the longest bucket, 31.36 GiB card:

| `--micro-batch-size` | worst-case peak |
|---|---|
| 16 | 23.4 GiB |
| 24 | 27.6 GiB |
| 32 | OOM at 30.0 GiB |

Two corollaries worth carrying forward:

- `gc.collect()` / `torch.cuda.empty_cache()` cannot fix this. Clearing cache
  does not create room for a peak that genuinely needs it. An earlier attempt to
  buy headroom that way was abandoned.
- **Throughput is nearly independent of micro-batch size here.** Cutting the
  micro-batches per step from 16 to 9 bought ~5%. Step time tracks target length
  (1.6 s on the shortest bucket to 4.9 s on the longest), because the model is
  compute-bound on this card. Do not spend GPU time tuning this flag for speed.

`728f533` makes the chunker budget on `examples x (template + longest target)`
so peak memory is bounded whatever bucket arrives, rather than relying on one
measured constant.

### Validation never chunked its batches

Separate bug, same symptom. `optimizer_step` chunked by `micro_batch_size`, but
all three `evaluate()` call sites passed `batch_size` (256), so validation ran at
16x the memory of training. This alone caused the first OOM. Fixed in `41c70cb`.

### The trained checkpoint could not be loaded by any evaluation

Every evaluation of `best.pt` died immediately:

    ValueError: scalar_affine_plus_low_rank requires 'low_rank_init_factor'

`checkpoint_config` claims to record everything `load_adapter` needs to rebuild
the projection, and omitted exactly that field, which the factory demands for
`low_rank_only` and `scalar_affine_plus_low_rank`. Nothing catches it at train
time: the run trains, validates and saves normally, and the gap appears only
when something tries to load the result -- five hours later, on a checkpoint
that is otherwise complete.

The field is consumed only by `initialize_weights`, whose output a restored
`projection_state` overwrites, so it has no effect on a loaded checkpoint and
merely has to be present. Fixed in code, with a test that rebuilds the
projection from the recorded config for each projection type; this run's
`best.pt` and `last.pt` were patched in place with the run's true 0.01
(`.pt.prepatch` backups beside them).

### The untrained floor is not 0.0013 as an aggregate

Plan §4 cites 0.0013 from `outputs/retrieval_reports/untrained_floor_centred.json`.
That file's aggregate recall@1 is **0.00068** (recall@5 0.00169, MRR 0.00160);
0.0013 is its *position-0* recall@1, and it was measured on `outputs/pangram_l19`,
not on a bg_think directory. All of these are ~1e-3, so the plan's use of it as
an order-of-magnitude reference stands -- but the precise figure should be quoted
as 0.00068, not 0.0013.

### Run provenance is silently lost from a synced worktree

`run_config.json` recorded `"git_commit": null`. The remote worktree's `.git`
file points at the local container's gitdir, which does not exist on the remote,
so `git rev-parse` fails with `fatal: not a git repository`. This affects every
run launched from a synced worktree, and it is silent -- training is unaffected,
only the provenance the field exists to record. Recorded by hand here after
verifying the remote sources are md5-identical to `ffbfdc7`.

Relatedly, `--run-dir` and `--vectors-k` must be given as absolute paths when
launching on the remote: they resolve against the cwd, which is the synced
worktree, and that is read-only for the remote `agent` user by design.

## Gate 2: validation loss -- passed

The run completed cleanly (exit 0) at 23:25 UTC, all 5,902 steps.

| slice | val loss | examples |
|---|---|---|
| whole mixture (`best_val_loss`) | **1.7087** | 300,000 |
| whole mixture (`measured_loss`, final pass) | 1.7116 | 300,000 |
| k=1 | **1.3294** | 50,000 |
| k=2 | 1.6505 | 100,000 |
| k=3 | 1.8797 | 150,000 |

**The k=1 slice is the only one with a prior**, and `bg_think` reached 1.4844.
This run lands at 1.3294 -- the same region, on the better side.

The plan says to treat "far better" as a bug signal and check the slice really
is k=1 before believing it, so that was checked rather than assumed. A k-group's
composed label is k labels joined by `"; "`, and semicolon-bearing titles were
filtered at extraction, so the separator count identifies k. Every one of the
300,000 val examples falls in the right slice: k=1 all have 0 separators, k=2
exactly 1, k=3 exactly 2, at 100% in each case. `val_k_ranges` is
`{1: (0, 50000), 2: (50000, 150000), 3: (150000, 300000)}` -- the intended
1:2:3 split.

**D13's caveat applies and is not a formality.** This run is centred against the
pooled reference, while 1.4844 was measured under per-directory centring, which
carries a constant per-position offset. The two are therefore not cleanly
comparable, and 1.3294 should not be reported as an improvement over `bg_think`
on that basis alone.

k=2 and k=3 have no prior and are not comparable to anything published. Loss
rising with k (1.33 -> 1.65 -> 1.88) is what a harder target looks like -- more
topics to verbalize per vector -- not evidence of degradation.

## Gate 3: set-level retrieval

TODO -- run in progress.

## OOD evaluations

TODO -- run in progress.

## The confound

TODO -- restate parent plan §7 once the OOD numbers exist.
