# Step 2b: the training loop

Part 2 of 6 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read its §4.1, §4.5, §5.4 and §6 step 2 before starting). Read
`plans/pangram_adapter_handoff.md` for execution state.

**Depends on** `plans/pangram_step2a_loss_and_eval.md` being merged: this plan uses
`adapter_training/dataset.py`, `loss.py` and `checkpoints.py` and adds nothing to them
except imports.

**Deliverable**: `adapter_training/train_adapter.py` -- a CLI that trains one projection to
a budget expressed in **examples seen**, validates on a fixed subsample, and writes
checkpoints `selfie_adapters.load_adapter` can read.

**No GPU work here.** Correctness is established locally (fake model, 1B smoke model, tiny
budgets). Benchmarking and the real runs are
`plans/pangram_step0_benchmarks.md` and `plans/pangram_phase0_run.md`.

## Research question (quoted, parent plan §1 -- never paraphrase)

> Instead of the extraction prompt `Tell me about <topic>`, use the prompt `Write "The
> quick brown fox jumps over the lazy dog". Think about the topic "<topic>" while writing
> the sentence. Do not write anything else or change the words.` The model should respond
> with "The quick brown fox jumps over the lazy dog"; extract the activations from all of
> the response tokens. As an initial test, let's only train the adapter for layer 19.
> Let's test over the Wikipedia dataset, from the reference code.
>
> Note - testing occasionally shows the model failing to reproduce the phrase "The quick
> brown...", so there should be a topic filter for generations that correctly reproduce
> the phrase.

## The settings to match, and where they come from

The published checkpoint `keenanpepper/selfie-adapters-llama-3.1-8b-instruct` /
`wikipedia-scalar-affine.safetensors` (this repo already names it in `config.py`) stores
its real training config in its safetensors header, and the parent plan (§3.1, D10) treats
that as **authoritative over both the paper's table and the YAML in
`resources/selfie-adapters/training/configs/`**, which are for the SAE runs.

| field | value |
|---|---|
| optimizer | AdamW, `weight_decay` 0.01 (upstream default) |
| learning rate | 0.01 |
| scheduler | cosine, 10 warmup steps |
| batch size | 256 |
| gradient clip | 0.5 |
| init scale | **5.0** (not the `ProjectionConfig` default of 30.0) |
| `normalize_input` | **true** |
| `strip_labels` / eos | true / `<|eot_id|>` |
| seed | 42 |
| `global_step` | 2951 |
| `best_val_loss` | 1.3662 |

Re-read these from the file rather than trusting this table -- `evaluate_adapter` from step
2a already surfaces the checkpoint metadata, and any disagreement with the table above is a
finding worth recording in the handoff.

**Budget is examples seen, never epochs** (parent plan §4.1, D4). The full budget is
**755,391 examples = 2,951 steps at batch 256**, which is upstream's single epoch over the
*train split* and exactly its recorded `global_step`. Arm B spends the same budget as ~0.1
epochs of its 10× larger pool; that is the point of the design, not a shortcut.

## Build: `adapter_training/train_adapter.py`

### CLI

```
python -m adapter_training.train_adapter \
    --vectors vectors/pangram_l19 \
    --run-dir runs/armB_scalar_affine \
    --budget-examples 755391 \
    --batch-size 256 --micro-batch-size 64 \
    --projection-type scalar_affine \
    --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 --seed 42 \
    --val-subsample 5000 --validate-every 100 \
    --pool-positions            # arm C: mean the 10 positions before the adapter
```

- `--vectors` and `--run-dir` are implicitly under `outputs/` (project convention).
- `--projection-rank` for `scalar_affine_plus_low_rank`. **Rank is a config field, never a
  literal in code** (parent plan §5.5): r=16 and r=64 must need no code change.
- `--restrict-topics-to <other vectors dir>` exposes step 2a's `restrict_to_titles`, for
  arm comparisons where the topic populations differ.
- `--max-steps` for debug runs (step 0 item 5 wants ~50 steps) -- it caps the loop but
  **does not** change the cosine schedule's horizon; see below.
- Light imports first, `args = parse_args()` before `import torch`, per project convention.

### The loop

1. **Seed everything** from `--seed` (Python, numpy, torch), and record it.
2. **Steps** = `ceil(budget_examples / batch_size)`. The cosine schedule is laid out over
   *those* steps (parent plan §5.4.1: "the cosine schedule is laid out over its own 2,951
   steps"), so a smaller budget is a complete run at a smaller scale rather than a truncated
   one. `--max-steps` stops early without altering the schedule -- a debug run must exercise
   the same schedule code the real run does.
3. **Warmup** for 10 steps, then cosine. Upstream chains a `LinearLR` warmup into
   `CosineAnnealingLR` and switches on `global_step` (`_setup_scheduler`,
   `_get_current_scheduler` in `resources/selfie-adapters/training/trainer.py`). Reproduce
   the effective LR curve; a single `LambdaLR` computing it in closed form is fine and
   easier to test.
4. **Sampling**: draw from the train examples with a seeded shuffle. If the budget exceeds
   the pool, wrap around into a fresh shuffle (arms A and C do exactly one pass; arm B does
   ~0.1 of one). Never draw val examples.
5. **Length-bucketed batching** (parent plan §4.2.1, D8, unconditional). Tokenize every
   train label's target length **once at startup** and cache it. Fill a shuffle buffer of
   ~50 batches (`50 × batch_size` examples), sort the buffer by target length, cut it into
   batches, shuffle the *batch order*, emit. Upstream pads to the batch's longest label and
   pays 53.0 tokens/example against 39.3 useful; bucketing brings that to ~39.7. This is
   exact -- it changes only which examples share a step.
6. **Micro-batching**: split each global batch into `--micro-batch-size` chunks, accumulate
   gradients, and take one optimizer step per global batch. The optimizer step must stay
   equivalent to a single batch of 256, so **scale each micro-batch's loss by
   `micro_len / batch_len`** rather than averaging micro-batch means -- micro-batches from a
   bucketed batch have equal length here, but do not rely on that.
7. Clip to `--grad-clip` on the projection parameters only, then step.
8. **Validation** every `--validate-every` steps on a fixed subsample of
   `--val-subsample` examples drawn once with the seed and reused every time (parent plan
   §4.5: comparable point to point, ~10% of run cost). Log `(examples_seen, step, val_loss,
   lr, scale, bias_norm)` to a JSONL in the run dir -- the parent plan §5.6 wants the
   *curve*, not just the endpoint.
9. **Checkpoints**: best-by-val and last, only (parent plan §4.4 caps disk this way), via
   step 2a's `save_checkpoint`. Also write `run_config.json` (every CLI arg, the resolved
   step count, the vectors directory and its `positions.json`, git commit) and copy the
   vectors' `position_means.pt` path into it, so a checkpoint can always be traced to the
   centring it was trained under.
10. **Final full-val pass** at the end, on the whole val split, written to
    `final_eval.json` -- this is the number the parent plan reports.

### Validation is not free

Upstream has a `_check_validation_compute_ratio` guard because validating the full split
every 50 steps costs more than the training it monitors (parent plan §4.5). Port the guard:
compute `val_batches_per_run × runs / train_batches` and **refuse to start** above 0.5,
naming `--val-subsample` in the error. This is cheap insurance against a config that
silently doubles the bill.

### Multi-GPU

Parent plan §4.3: the default is **one run per GPU** (a `--device` flag and nothing else),
because phases 1-2 are four independent runs and that reaches full utilisation with no
distributed code. DDP is optional and only helps phase 0's single run. Build the
`--device` path now; leave a `--ddp` flag unimplemented or behind a clear
`NotImplementedError` unless phase 0's wall clock turns out to matter. **Do not use
`device_map="auto"`** -- the 8B fits on one card, and pipeline-sharding it only adds
latency.

### Gradient checkpointing

Off by default. Parent plan D8: it is a ~1.5× tax and it silently nulls `past_key_values`,
which blocks the prefix cache of `plans/pangram_step2c_prefix_cache.md`. Expose
`--gradient-checkpointing` for the memory-constrained case and make it mutually exclusive
with the prefix-cache flag when that lands.

## Tests -- `tests/test_train_adapter.py`

Fast (fake model + fake tokenizer, tiny budgets):

1. Step count is `ceil(budget / batch_size)`; 755391 at 256 gives **2951**, matching the
   published `global_step`. Pin this number.
2. The LR curve: 10 warmup steps rising linearly to `--lr`, then cosine decaying to ~0 at
   the final step. Assert values at steps 0, 10, midpoint, last.
3. `--max-steps 50` stops after 50 steps but the LR at step 50 equals what the full-horizon
   schedule gives at step 50.
4. Bucketing: batches are length-homogeneous, every example appears exactly once per pass,
   and the multiset of examples over a budget equals what an unbucketed sampler would draw
   with the same seed (i.e. bucketing reorders, it does not resample).
5. Gradient accumulation: training one step with `micro_batch=batch` and with
   `micro_batch=batch/4` produces the same projection parameters within fp32 tolerance.
6. The validation subsample is identical across validations within a run and across two
   runs with the same seed.
7. The validation-compute guard raises on a config that would validate the full split every
   50 steps, and does not on the plan's `5000 / 100` setting.
8. Determinism: two runs, same seed, same budget -> bit-identical `projection_state`.
9. A checkpoint written mid-run loads through `selfie_adapters.load_adapter`.
10. Arm C (`--pool-positions`) trains on one vector per topic and its example count equals
    the topic count × labels.

`hf_cache`-marked (Llama-3.2-1B): a ~20-step end-to-end run on a hand-built 5-topic vectors
directory completes, loss decreases, and the artefacts (`best.pt`, `last.pt`,
`run_config.json`, the metrics JSONL, `final_eval.json`) all appear.

## Done when

- All of the above passes; `--help` costs no torch import.
- A 20-step smoke run against 1B produces a checkpoint that `interpret.py`'s adapter loader
  accepts.
- `plans/pangram_adapter_handoff.md` updated: step 2b done, the measured step count,
  anything that disagreed with the published checkpoint's recorded config.
- Committed on a worktree branch with an undrafted PR.

## Do not

- Do not budget by epochs, and do not "helpfully" scale the budget with the pool size --
  equal examples seen across arms is the comparison (parent plan §4.1, D4).
- Do not train on val examples, and do not split per vector: splits are **topic-level** and
  inherited from the upstream dataset (parent plan §5.2). A per-vector split leaks, because
  a val vector would be a near-duplicate of a train vector with an identical label set.
- Do not centre at evaluation time (step 2a covers why).
- Do not reach for the prefix cache here. It is deliberately a separate, gated step.
