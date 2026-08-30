# Step 0 findings (`plans/pangram_step0_benchmarks.md`)

## Headline: the 1.3662 gate PASSED

Published upstream adapter (`keenanpepper/selfie-adapters-llama-3.1-8b-instruct:wikipedia-scalar-affine.safetensors`),
scored through this repo's loss path on the baseline extraction's val split
(`outputs/baseline_l19`, 84,211 examples, 2,632 batches at batch size 32),
centred:

| | loss |
|---|---|
| **measured** | **1.3636** |
| published `best_val_loss` (checkpoint metadata, `global_step=2951`) | 1.3662 |
| gap | **0.0026** |
| untrained floor (same split, centred) | 3.8794 |

**Tolerance applied: the plan's top band, "within ~0.02 of 1.3662 --
agreement; proceed".** 0.0026 is an order of magnitude inside it, so the
plan's own known-drift list (§9.4: batching differences, per-batch vs
per-example val aggregation -- `position_ids` was ruled out by PR #43) does
not need to be invoked to explain a residual -- there is effectively no
residual to explain. The second half of the stop condition also does not
trigger: the trained adapter is clearly better than the untrained floor
(1.3636 vs 3.8794, a 2.52-nat margin). The floor was re-measured centred on
the fixed vectors -- the old 3.8059 was an uncentred measurement and is
superseded, though the two are close enough that this comparison was never in
doubt.

**Verdict: `plans/pangram_phase0_run.md` is cleared to start.**

Reports: `outputs/upstream_published_centred.json` and
`outputs/untrained_floor_centred.json`. Both were captured from the runs'
stdout rather than written via `--report`, which the commands omitted; the
contents are the runs' own emitted JSON verbatim.

### Correction: this section previously reported a FAILURE

Until 2026-08-30 this table read 1.7800 measured / 0.414 gap / 3.8059 floor,
and recommended not starting phase 0. **All three of those numbers were
measured through a broken centring path and are invalid.**
`extract_baseline_vectors.py` wrote `position_means.pt` as a 1-D `[hidden]`
tensor, while `dataset.py::load_vector_store` indexes a leading position axis
(`means[:n]`). With the baseline style's `count == 1` that sliced to a single
scalar which broadcast over all 4096 dimensions, so the vectors reached the
loss effectively **uncentred**, silently and with no error. Fixed in PR #51,
which also makes `load_vector_store` raise on any means file that is not
`[n_positions, hidden]`, so no pre-fix extraction can be scored silently
again. Full write-up: `plans/notes/step0_reference_repro_handoff.md`
(update #6). The superseded artefacts are in
`outputs/archive/2026-08-30-broken-centring/`.

The investigation list that used to follow this section is resolved and has
been removed, with one exception worth keeping on the record:

- ~~Whether the checkpoint's `normalize_input`/`init_scale` are being applied
  identically to how `evaluate_adapter.py` constructs the projection at eval
  time~~ -- **ruled out.** Downloaded the actual published
  `wikipedia-scalar-affine.safetensors` and inspected its safetensors
  metadata header directly: it carries a `config_json` key (so
  `SelfIEAdapter._load_safetensors` takes the structured-parse branch, not
  the flat-string fallback), and `normalize_input: true` agrees between both
  the structured and flat copies of the field. `init_scale` is inert at eval
  time regardless (`load_state_dict` overwrites `log_scale`/`bias` right
  after `initialize_weights` sets them). Every training config this project
  has seen -- upstream's three shipped configs and this checkpoint's own
  metadata -- uses `normalize_input=true`; the only `false` anywhere in the
  reference repo is an unrelated downstream caller override
  (`evals/generation_scoring/label_generator.py`) for an SAE-scale-sweep
  eval script, not a training-time choice. Consequently the CLI's
  `--normalize-input`/`--no-normalize-input` flags were dropped
  (`train_adapter.py` now hardcodes `True`), and `checkpoints.py::load_projection`
  now raises if a loaded checkpoint's metadata ever records `false`, so a
  genuine future deviation is loud instead of silent.
The list's remaining entries -- a layer-19 indexing off-by-one, and a
topic-population mismatch in the centring means -- are answered by the gate
now passing: both would have left a systematic residual, and none remains.

**The one open question the fix does not settle** is that no recorded training
config exists for this checkpoint. None of the three YAML configs in
`resources/selfie-adapters/training/configs/` trains on Wikipedia topics; all
three point `labels_file` at Goodfire SAE decoder vectors. This did not matter
for the gate, but it means the exact recipe behind
`wikipedia-scalar-affine.safetensors` is still inferred rather than known.

## Benchmark table (item 3)

*Unaffected by the centring bug.* This item and item 5 below both run on
`pangram_l19`, whose `position_means.pt` was always correctly `[10, 4096]` --
`extract_pangram_vectors.py` never had the shape defect. Nothing in either
section was re-measured or changed.

Real arm-B config on real pangram vectors (`/home/agent/outputs/pangram_l19`,
470,630 vectors), scored via the trainer's own per-step functions directly
(bypassing `train()`'s expensive whole-val-set final pass), 10 warmup + 50
measured steps, batch size 256, `benchmark_step0.tmp.py`:

| configuration | micro-batch | grad checkpointing | result |
|---|---|---|---|
| reference, checkpointed | 256 (no accum) | on | **OOM** |
| + length bucketing | 256 (no accum) | on | **OOM** |
| + checkpointing off | 32 | off | 25.4 examples/s, 26.8 GB peak |
| + checkpointing off | 64 | off | **OOM** |
| + checkpointing off | 128 | off | **OOM** |

**This rental was 4x 16 GB GPUs, not the plan's assumed single 24 GB card**
(see `plans/notes/step0_handoff.md` for why: the 8B model alone is 16.06 GB
in bf16, so it was split across 2 GPUs per job via `device_map="auto"`, a
change made this session -- see the 7-file diff below). Only micro-batch 32
without gradient checkpointing fit in the ~32 GB (2x16 GB) available to one
job; every other configuration, including checkpointed at full batch, OOM'd.
This is a real memory-budget finding for this hardware, not a bug: with the
model itself occupying most of each 16 GB card, there is much less headroom
than the plan's single-24GB-card benchmarks assumed.

**Chosen setting for phase 0 (on this hardware): `--micro-batch-size 32`,
no `--gradient-checkpointing`.**

Re-derived full-budget cost at the measured rate: 755,391 examples /
25.4 examples/s = **~8.3 hours** on a 2x16GB GPU pairing. (The parent plan's
own cost table carries ±40% and assumed different hardware, so this
supersedes it for this rental; a 24 GB single card should do meaningfully
better once checkpointing is available at a larger micro-batch, but that
config wasn't reachable here to compare directly.)

## Item 4: prefix-cache equivalence

**Not attempted.** `plans/pangram_step2c_prefix_cache.md` has not landed
(no `--prefix-cache` flag or module exists in this worktree).

## Item 5: the 50-step debug run

Real arm-B config (`--lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip
0.5 --seed 42 --projection-type scalar_affine --budget-examples 755391`, no
`--pool-positions`), on real pangram vectors, `--micro-batch-size 32`,
`--max-steps 50 --validate-every 25 --val-subsample 2000`. Run dir:
`/home/agent/outputs/debug_armB`.

`metrics.jsonl`:

```
{"examples_seen": 6400,  "step": 25, "train_loss": 2.359, "val_loss": 2.443, "lr": 0.0099994, "scale": 5.640, "bias_norm": 5.271}
{"examples_seen": 12800, "step": 50, "train_loss": 2.106, "val_loss": 2.021, "lr": 0.0099957, "scale": 5.251, "bias_norm": 8.017}
```

Checks (per the plan):

- **Loss region**: both val losses (2.44, 2.02) are well below the untrained
  floor's own vector population trend and falling steadily -- no sign of a
  mis-wired pipeline. (Not directly comparable in absolute terms to the
  baseline-extraction numbers above -- different vector population, pangram
  vs baseline -- but the *within-run* trend is what this check is for.)
- **Loss falls over the run**: train 2.359 -> 2.106, val 2.443 -> 2.021. Yes.
- **LR matches the 2,951-step horizon, not a 50-step one**: closed-form
  `lr_at_step(24, base_lr=0.01, warmup_steps=10, total_steps=2951) =
  0.009999440889456266` and `lr_at_step(49, ..., total_steps=2951) =
  0.00999566173470822` -- both match the logged values to full float
  precision. (A 50-step-horizon schedule would instead give ~0.00727 and
  ~0.0000154 at those same steps -- nowhere close.) Confirms `--max-steps`
  only truncates the loop, as designed; the schedule horizon is correct.
- **Soft-token norms / scale finite and moving**: `scale` 5.640 -> 5.251,
  `bias_norm` 5.271 -> 8.017. Both finite, both moving, no NaN/Inf.
- **Centring applied once, train and val both centred**: `run_config.json`
  records `position_means_path` pointing at `pangram_l19/position_means.pt`;
  `load_train_and_val` always calls `load_vector_store(..., center=True)`
  for both splits when not pooling (`train_adapter.py`'s own module
  docstring: "this trainer always uses centred vectors"). No raw path was
  exercised.

The run's final full-val pass (over the entire 793,690-example val split, at
~25 examples/s -- roughly 8.7 hours) was killed once the 50 optimizer steps
and both scheduled validations had completed and been flushed to
`metrics.jsonl`; that pass is not part of what item 5 asks for and would
have been a pure waste of GPU time for a debug run whose job was already
done. `final_eval.json` was consequently never written -- not needed, all
the checks above come from `metrics.jsonl`.

**Not a result** (per the plan): this loss is not comparable to, and does
not supersede, the gate failure above.

## Multi-GPU fix (why this was necessary)

The rental was 4x 16 GB GPUs; the 8B model's bf16 weights alone are 16.06 GB,
so it does not fit on one card. Fixed via `device_map="auto"` plus a new
`resolve_device()` helper (`model_loading.py`) that callers use instead of
`args.device` for any tensor they create themselves (input ids, the
trainable projection, template embeddings) -- with a sharded model,
`accelerate`'s dispatch hooks move activations between devices during the
forward pass, but a caller-created tensor must still start on the embedding
layer's device. Call sites updated in `extract_baseline_vectors.py`,
`extract_pangram_vectors.py`, `evaluate_adapter.py`, `train_adapter.py`,
`evaluate_retrieval.py`.

This also surfaced a real, previously-latent bug in `loss.py`: under
sharding, `lm_head` can land on a different GPU than the embedding layer
(observed cuda:0 vs cuda:3), so `SoftPromptLoss.__call__` crashed on a
cross-device mismatch the first time it ran on a genuinely multi-GPU model
(not caught by the existing tests, which use single-device fakes). Fixed by
moving `target_ids`/`target_mask` to `logits.device` right before the loss
combines them.

## Extraction artefacts

Both live at, remotely (vast.ai instance `vai-0`, `agent` account):

- `/home/agent/outputs/baseline_l19/` (443 MB, 49,637 topics -- also
  step 0's own reproduction-check extraction, so `pangram_phase0_run.md`
  does not need to redo it)
- `/home/agent/outputs/pangram_l19/` (3.9 GB, 470,630 vectors from
  47,063/49,637 topics that passed the compliance filter)

Both are synced back (via the vast-remote-broker's automatic outputs/ sync,
confirmed with an explicit `sync_flush`) to this repo's local
`outputs/baseline_l19/` and `outputs/pangram_l19/` (gitignored, not
committed).

## Skipped

- Item 4 (prefix cache): not attempted, dependency plan hasn't landed (see above).
- The full benchmark grid: 4 of 5 configurations OOM'd on this rental's
  2x16GB-per-job GPU pairing (see table above); only the one working
  configuration was actually measured.
