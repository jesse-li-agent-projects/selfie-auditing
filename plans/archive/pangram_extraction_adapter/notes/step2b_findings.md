# Step 2b findings

Execution of `plans/pangram_step2b_training_loop.md`. Built
`adapter_training/train_adapter.py` and `tests/test_train_adapter.py`; moved
`FakeCharTokenizer` from `test_loss.py` into `tests/conftest.py` since this
step's tests need it too (`test_loss.py` updated to import it, no behavior
change).

## The published checkpoint's config, re-read from the file

`load_projection` (step 2a) against
`keenanpepper/selfie-adapters-llama-3.1-8b-instruct:wikipedia-scalar-affine.safetensors`
confirms the plan's settings table exactly, with no disagreement worth
recording:

```
projection_type: scalar_affine
model_dim: 4096
normalize_input: true
checkpoint_format_version: 0
global_step: 2951
best_val_loss: 1.3662405303030303
init_scale: 5.0
```

`global_step=2951` and `best_val_loss≈1.3662` match the table verbatim.
`compute_total_steps(755391, 256) == 2951` (test 1), so the budget-in-
examples math reproduces the published step count exactly.

## Deviations from the plan's literal sketch

All additive, same reasoning as step 2a's note -- the plan's sketch still
works unchanged.

- `TrainConfig` is a dataclass built either from parsed CLI `args`
  (`TrainConfig.from_args`) or directly (as the fast tests and the smoke
  test do, without going through `argparse` at all). This is what let the
  smoke test (`hf_cache`) build a `TrainConfig` with `batch_size=8` and
  `budget_examples=160` without a CLI round-trip.
- `load_train_and_val` returns `(train_store, train_examples, val_store,
  val_examples)` rather than a single store, to support `--pool-positions`
  (arm C): pooling only touches the pooled store, and train/val need
  independent pooled stores so a pooled train vector is never built from
  vectors that include a held-out topic.

## Testing notes

- All 17 fast tests pass locally (fake model, no GPU).
- The `hf_cache`-marked 20-step smoke run against Llama-3.2-1B (`gpu-exec`,
  ~8 minutes on CPU) passes: `best.pt`, `last.pt`, `metrics.jsonl` and
  `final_eval.json` all appear, val loss decreases over the run, and the
  checkpoint loads through `selfie_adapters.load_adapter` exactly as
  `interpret.py`'s own loader does.
- Full repo suite (`pytest`, fast-only): 119 passed, 1 pre-existing failure
  unrelated to this step (`test_extract_pangram_vectors.py::
  test_pangram_prompt_is_the_requested_wording`, a stray-quote mismatch
  already flagged in step 2a's findings note -- not touched here).

## Done-when checklist

- [x] All plan tests pass; `--help` costs no torch import (confirmed via
      `python -X importtime`, no `torch` frame before argparse exits).
- [x] A 20-step smoke run against 1B produces a checkpoint
      `interpret.py`'s adapter loader accepts.
- [x] This note.
- [ ] Committed on a worktree branch with an undrafted PR -- next.
