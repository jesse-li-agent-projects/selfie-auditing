# Step 4: finer training-loss logging

Part 4 of 6 in the execution of `plans/bg_think_many/bg_think_many.md`.

**Small, self-contained and independent of every other step.** No GPU, no
network. Good first task for an agent working alone.

## Why

Quoted from the user's request (2026-09-05):

> A note that I don't want to forget about - I'd like finer resolution logging
> of the training loss during training, maybe 2x finer? Ideally, training loss
> is logged more frequently than validation loss (the latter can stay at its
> current frequency).

## What is there now

`train_adapter.py` writes one record to `metrics.jsonl` only inside the
`if global_step % config.validate_every == 0 or is_last_step:` branch. So
`train_loss` and `val_loss` are both written every 100 steps, and the training
curve is sampled at exactly the same rate as the (much more expensive) validation
curve. `train_loss` in that record is the loss of the single batch at that step,
not an average -- which is why the existing curve is visibly noisy.

## What to build

Add `--log-every`, default **50**, independent of `--validate-every` (default
100, unchanged).

- Every `log_every` steps, write a metrics record with `examples_seen`, `step`,
  `train_loss`, `lr`, `grad_norm`, and the projection metrics.
- Every `validate_every` steps (and on the last step), run validation as now and
  write a record that additionally has `val_loss`, then do the checkpointing and
  resume-state saving exactly as now.
- When a step is both, write **one** record with both keys, not two records.
- Validation, checkpointing and `--resume` behaviour must not change at all.
  `check_validation_compute_ratio` concerns validation cost only and takes
  `validate_every`; do not pass it `log_every`.

**Log the mean train loss over the interval, not the last batch's.** Accumulate
the per-step losses since the previous log and write their mean, alongside the
step's own `train_loss`. With `log_every=50` that is a much more readable curve
at no cost, and it is the thing a reader actually wants when asking for finer
resolution. Name the key `train_loss_mean` and keep `train_loss` as the raw
per-step value, so old and new runs stay comparable.

## Two consumers to keep honest

1. **`plot_train_dynamics.tmp.py` will break.** It does
   `val_loss = [r["val_loss"] for r in rows]`, which raises `KeyError` on a
   log-only record. It is a `*.tmp.py` throwaway, so quality does not matter, but
   it must still run: change it to skip rows without the key when building the
   validation series. Do not restructure it otherwise.
2. **Add the new projection metrics while you are here.** `_metric` already
   returns `None` for a getter a projection does not have, so this is safe for
   every architecture. `ScalarAffinePlusLowRankProjection` (the architecture this
   experiment uses, parent plan D9) exposes `get_low_rank_norm` and
   `get_low_rank_to_diagonal_ratio`. Log both as `low_rank_norm` and
   `low_rank_to_diagonal_ratio`. They answer "is the rank-64 part doing anything,
   or did the run collapse to scalar-affine?", which is a question step 6 will
   otherwise have no way to answer after the fact.

## Tests

Extend `tests/test_train_adapter.py`:

- with `log_every=50, validate_every=100`, a 200-step run writes 4 records, of
  which 2 have `val_loss`
- a step that is both logs and validates writes exactly one record
- `log_every` does not change how many validations run (assert the validation
  call count, so a regression here is caught rather than merely being slow)
- `--resume` mid-run still appends correctly and does not duplicate a record at
  the resume boundary
- `train_loss_mean` over an interval equals the mean of that interval's
  per-step losses

## Note

Nothing else in this step. Resist widening it -- it is deliberately the one task
in this plan that can be finished, reviewed and merged in isolation while the
GPU work queues behind other steps.
