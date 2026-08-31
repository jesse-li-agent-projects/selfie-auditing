# Phase 0 results (`plans/pangram_phase0_run.md`)

Executed on a single rented RTX 5090 (32 GB) via the vastai remote, 2026-08-30/31. All
artefacts are under `outputs/` (gitignored) on the same machine this note was committed
from; commands, checkpoints and reports referenced below are reproducible from there.

## Verdict

**The phase-0 gate passes, decisively.** Arm B's centred recall@1 (40.80%) exceeds the
untrained floor on the same pangram vectors (0.068%) by a ~600x margin -- nowhere close to
the "clears its own floor by a margin worth attributing" bar the parent plan sets (§5.5,
§5.4.1). The pangram activations carry topic signal this adapter could learn, and the
question phase 1 asks -- whether that signal comes from the prompt or from per-position
training -- is worth asking. That is a recommendation, not a decision: rolling into
`plans/pangram_phases12_and_report.md` is the user's call.

## Retrieval table (the headline, D11)

All rows query the 4,681 val topics that survive the pangram compliance filter, against the
same GTE-large index of all 49,637 topics (title + all labels, `IndexStrategy.
TITLE_PLUS_ALL_LABELS`), same generation settings (`max_new_tokens=30, temperature=0.7,
seed=42`), same embedding model (`thenlper/gte-large`).

| row | vectors | condition | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| **arm B (headline)** | pangram_l19 | centred | **0.4080** | 0.5611 | 0.6221 | 0.4842 |
| arm B | pangram_l19 | raw | 0.0228 | 0.0531 | 0.0724 | 0.0409 |
| **untrained floor** | pangram_l19 | centred | **0.00068** | 0.00169 | 0.00265 | 0.0016 |
| OOD control (published adapter) | pangram_l19 | centred | 0.2339 | 0.3705 | 0.4282 | 0.3034 |
| OOD control (published adapter) | pangram_l19 | raw | 0.0109 | 0.0271 | 0.0375 | 0.0209 |
| upstream reference (published, own task) | baseline_l19 | centred | 0.5727 | 0.7144 | 0.7708 | 0.6435 |

The paper's own published scale for contrastive-vector retrieval is 94% recall@1 (trained)
against a 1% untrained floor -- context for reading these numbers, not a target (parent plan
§5.4.1).

**Centred is the primary condition** (matches the paper's contrastive-vector eval); **raw is
a labelled secondary condition**, reported because raw is what the taboo pipeline will
actually feed the adapter (D6). Both drop by an order of magnitude or more going from
centred to raw, for every row that has both -- expected, since raw activations are
dominated by "which pangram word is this" rather than topic (parent plan §5.3).

**Do not compare the upstream-reference row to arm B as a target.** It is scored on a
different task (baseline extraction, one vector per topic) and a different population
(all 49,637 topics vs. the pangram-filter-surviving 47,001); it is a labelled reference
point only, per parent plan §5.4's comparison table.

### The OOD control (not in the original plan -- added on request)

The published adapter (`wikipedia-scalar-affine.safetensors`, trained on the *baseline*
extraction prompt, never exposed to a pangram activation) was also scored on `pangram_l19`
val -- the same task and query set as arm B and the untrained floor, just with a
mismatched-training adapter. It lands at 23.39% recall@1: far above the untrained floor
(0.068%), but well below arm B's task-specific 40.80%.

This is informative on its own: it says the pangram activations' topic signal is partly
generic -- decodable by an adapter that never saw this extraction prompt, consistent with
the paper's own claim that SelfIE adapters transfer across extraction prompts (parent plan
§3.2, the TwoHopFact result) -- and partly specific, since training on the matching
activations still buys a further ~17.4-point lift. Neither number is a gate; both are
context for interpreting arm B's result.

## Per-position recall breakdown (exploratory, parent plan §5.6)

Arm B, centred, one row per response-token position (`n_labels=4,681` each):

| position | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| 0 (`The`) | 0.4035 | 0.5815 | 0.6496 |
| 1 (` quick`) | 0.4747 | 0.6426 | 0.7073 |
| 2 (` brown`) | 0.4614 | 0.6355 | 0.7022 |
| 3 (` fox`) | 0.4399 | 0.5749 | 0.6261 |
| 4 (` jumps`) | 0.3328 | 0.4837 | 0.5518 |
| 5 (` over`) | 0.3196 | 0.4565 | 0.5217 |
| 6 (` the`) | 0.3882 | 0.5443 | 0.6041 |
| 7 (` lazy`) | 0.4185 | 0.5759 | 0.6353 |
| 8 (` dog`) | 0.3388 | 0.4570 | 0.5074 |
| 9 (`.`) | **0.5029** | 0.6593 | 0.7154 |

`best_position = 9`, the trailing full stop -- but the spread across positions (0.32-0.50
recall@1) is modest and non-monotonic, not a clean "later is better" story. This does not
feed the B/C conclusion (parent plan §5.6); it is noted for shaping a possible follow-up
(e.g. whether a subset of positions carries most of the signal), not acted on here.

## Loss table

| what | vectors | population | loss |
|---|---|---|---|
| arm B, full val split | pangram_l19 | 793,910 examples | 1.48716 |
| arm B, seeded subsample | pangram_l19 | 84,211 examples, seed 42 | 1.48587 |
| untrained floor, same subsample | pangram_l19 | 84,211 examples, seed 42 | 3.92159 |
| D10 trainer-correctness check (reused, `plans/notes/step0_findings.md`) | baseline_l19 | 84,211 examples (full val) | 1.3636 measured vs. 1.3662 published |

**Arm B's loss is not comparable to the D10 check's 1.3636/1.3662, and neither is a gate on
arm B.** Different extraction prompt, different vector population (~842k pangram val
vectors before filtering vs. 84,211 baseline), different task -- a lower or higher number
either way says nothing about the pangram method (parent plan §5.4). The two are shown
together only so a reader who knows the published numbers has a scale to read arm B's loss
on.

Arm B's full-split and subsample losses agree closely (1.48716 vs. 1.48587), which is
mostly a sanity check that the subsample is representative -- it was scored anyway because
the training run's own end-of-run pass already covers the full split at negligible extra
cost on this hardware (see wall-clock section).

## Loss vs. examples seen

From `outputs/phase0_armB/metrics.jsonl` (subsampled log points, `val_loss` on the 5,000-example
in-run subsample):

| step | examples seen | train_loss | val_loss |
|---|---|---|---|
| 100 | 25,600 | 1.748 | 1.756 |
| 1,000 | 256,000 | 1.614 | 1.511 |
| 2,000 | 512,000 | 1.583 | 1.489 |
| 2,951 (final) | 755,456 | 1.433 | 1.484 |

Val loss falls sharply over the first ~500k examples, then flattens: by step 2,700 (of
2,951) it has already reached its recorded `best_val_loss` of 1.484375 and does not improve
further through the end of the cosine schedule. **This is a sign the budget was adequate,
not under-budget** -- unlike the "still improving at the cap" case the parent plan flags
(§5.6) as a reason to consider extending it, arm B's curve is flat well before the cap, so
extending the budget would not obviously buy more.

## Extraction filter, full scale (parent plan §9.3, §9.4)

| | full corpus (this run) | 500-topic probe |
|---|---|---|
| keep rate | 94.69% (47,001 / 49,637) | 95.4% |
| dominant real failure mode | word substitution at position 3 (` fox`), 1,909/49,637 = 3.85% | ~4% |

The keep rate and the dominant failure mode both land close to the probe's numbers -- no
red flag at full scale. `first_mismatch_histogram` peaks sharply at index 3 (1,909 of 2,636
rejections), consistent with the probe's finding that the model's main failure mode is
substituting a topic word into the pangram (e.g. "The quick brown **monarch** jumps...")
rather than quoting, adding a preamble, or refusing.

**One extraction detail worth recording, not a defect:** every one of the 47,001 kept
topics matched via the with-stop variant (`variant_counts` shows a single key, the full
sentence with its period); none needed the no-stop fallback. This differs from the probe's
68%/27% with-stop/no-stop split, but is explained: the pangram instruction's exact wording
was revised between the probe and this run (quoting the sentence as `fox."` rather than
`fox".`), removing the ambiguity that produced the original split. Not re-verified against
a fresh probe: full-scale extraction moved directly to using both candidate variants
per the existing two-variant extractor (§9.2), which still functions correctly regardless
of the split's shape.

## Wall clock and cost

Measured on a single rented RTX 5090 (32 GB), not the parent plan's assumed A100 -- these
numbers supersede the plan's ~3.0 A100-hour estimate for *this* hardware; see
`plans/notes/step0_findings.md` for the previous (2x16 GB) rental's numbers.

| stage | wall clock |
|---|---|
| Extraction, pangram, 49,637 topics | ~11m51s |
| Extraction, baseline, 49,637 topics | ~3m25s |
| Batch-size tuning (micro-batch sweep, 256/128/64/32) | ~35m, mostly OOM probing |
| Arm-B training loop (2,951 steps, budget 755,391 examples) | ~1h50m (21:15-23:05 UTC) |
| Arm-B full end-of-run validation (793,910 examples) | ~57m (23:05-00:02 UTC) |
| Retrieval eval, 6 rows (index build + 5 generation/scoring passes, one retry after a missing checkpoint) | ~2h (00:11-02:17 UTC, including one avoidable ~15min stall from a checkpoint that hadn't been pre-fetched) |
| Val-loss comparators (2 rows, 84,211 examples each) | ~a few minutes |

Total, extraction through the last retrieval report: roughly 5 hours wall clock, single
GPU, no parallelism attempted (D5 -- the machine was user-nominated). Measured steady-state
training throughput was ~128 examples/s at micro-batch 32 with gradient checkpointing off
-- both larger micro-batches (128, 256) OOM'd on this 32 GB card, tighter than the parent
plan's sizing assumptions for a single 24 GB card (§4.2); the two-point timing method used
to get this number is in the session transcript, not reproduced here.

## Real environment gaps hit and fixed this run

Two fixes landed in `adapter_training/retrieval_eval.py` on this worktree branch, both
needed to get the retrieval eval running at all on a freshly rented instance:

1. **`resources/selfie-adapters` (the vendored SelfIE reference code) is gitignored and
   was never reaching any fresh vastai instance** -- the project's sync only carries
   tracked `.py` files from the main repo plus worktree files in full, and this directory
   is neither. Worked around by copying it into this worktree (which does sync in full)
   and making the module's `_REFERENCE_ROOT` overridable via a `SELFIE_REFERENCE_ROOT`
   env var, so a future run on a different fresh machine has a documented way to point at
   wherever the reference code actually landed, rather than assuming `resources/` next to
   `cwd`.
2. **The `datasets` pip package is absent on the remote `agent` account, and that account
   has no network egress to install it.** It is a dead import for our usage --
   `TopicRetrievalIndex.load_dataset()` is the only caller, and this module's own
   `build_index()` deliberately never calls it (uses a local topics list instead, per the
   module's docstring). Stubbed `sys.modules["datasets"]` with a loud-failing placeholder
   rather than requiring the real package, so a genuine accidental call surfaces instead of
   silently misbehaving.

Neither fix changes any measured number above; both were required just to get the
retrieval eval's imports to succeed.

## Done-when checklist

- [x] Arm-B checkpoint, metrics JSONL, `final_eval.json`, and all six retrieval reports
      exist under `outputs/` (gitignored) and are synced back to this machine.
- [x] Retrieval table, loss table, and the loss-vs-examples-seen curve written up above.
- [x] Verdict and gate decision recorded (passes).
- [ ] Committed on a worktree branch with an undrafted PR -- this note and the
      `retrieval_eval.py` fixes are committed locally; opening the PR is a separate step,
      pending the user's go-ahead.
