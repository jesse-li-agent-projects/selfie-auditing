# Step 6: train bg_think_many, evaluate it, and report

Part 6 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read §6 Gates, §7 and D7, D9, D12).

**Depends on all of steps 1-5 being merged.** Check that before booking GPU
time; a half-merged step here wastes the most expensive hours in the plan.

**This is the large GPU step**: ~5-6 A100-hours for the training run, plus the
evaluation passes. **One agent at a time on GPU work** (project rule). Do not
start while step 2's extraction, or any other agent's GPU job, is running.

## Research question

Quoted in the parent plan §1. Read it there; do not restate it in your own
words, here or in the report.

## 1. Preconditions

- `outputs/bg_think_l19` (k=1, reused per D1), `outputs/bg_think_many_l19_k2`
  and `outputs/bg_think_many_l19_k3` all exist, and step 2's Gate 1 passed.
- Step 1's bucketing rule is merged, and step 3 uses it.
- Step 4's `--log-every` and step 5's `score_sets` are merged.
- The unit tests pass. **Note the project rule: passing unit tests do not mean
  the codebase works.** They cover parts of this pipeline only.

## 2. The training run

    python -m adapter_training.train_adapter \
        --vectors-k 1=outputs/bg_think_l19 \
        --vectors-k 2=outputs/bg_think_many_l19_k2 \
        --vectors-k 3=outputs/bg_think_many_l19_k3 \
        --mixture-ratio 1:2:3 \
        --run-dir adapters/bg_think_many \
        --budget-examples 1510782 --batch-size 256 \
        --projection-type scalar_affine_plus_low_rank --projection-rank 64 \
        --low-rank-init-factor 0.01 \
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 \
        --weight-decay 0.01 --seed 42 \
        --validate-every 100 --log-every 50 --val-subsample 5000

5,902 steps. Every hyperparameter above except the mixture, the budget and
`--log-every` is what `bg_think` used and what upstream's
`scalar_plus_low_rank_8b.yaml` specifies -- see the parent plan §3. Do not tune
them; a tuned run would not be comparable to `bg_think`.

Use `--resume` if it crashes. It resumes from the last validation step, so at
most 100 steps of work is redone.

**Watch `low_rank_norm` and `low_rank_to_diagonal_ratio`** in `metrics.jsonl`
(step 4 added them). If the ratio stays near zero throughout, the rank-64
component never engaged and the run is effectively `scalar_affine` with extra
steps -- which is worth knowing *during* the run, not after.

## 3. Gate 2: validation loss

From `final_eval.json` and the per-k slices step 3 provides.

- The whole-mixture `best_val_loss` is the headline correctness number.
- **The k=1 slice is the only one with a prior**: `bg_think` reached 1.4844 on
  single-topic contrastive val vectors. The k=1 slice here should land in the
  same region. Far worse means the mixture hurt single-topic performance, which
  is a real and reportable finding. Far better is more likely a bug -- check the
  slice really is k=1 examples before believing it.
- k=2 and k=3 have no prior and are not comparable to anything published. Report
  them; do not compare them to 1.3662 or to 1.4844.

Remember what these numbers are: loss on the **same contrastive, in-distribution
vectors the adapter trained on**. That is the correctness gate for a newly
trained checkpoint. It is not evidence about OOD behaviour, and `interpret.py`
is not this check.

## 4. Gate 3: set-level retrieval, in distribution

Score four adapters on the **same grouped val vectors**, with identical decoding
settings (D10: `--max-new-tokens 110`, temperature 0.7, `n_samples` 1, seed 42):

| adapter | what it is |
|---|---|
| `bg_think_many` | this run's `best.pt` |
| `bg_think` | `outputs/adapters/bg_think/best.pt` |
| `baseline` | `outputs/adapters/wikipedia-scalar-affine.safetensors` |
| untrained floor | `checkpoints.untrained_projection` |

Run **centred**, which is what the paper uses for contrastive-vector retrieval
and what makes these numbers comparable to the existing
`outputs/retrieval_reports/*_centred.json`. Report per k (D12) and always with
the `segments` block beside each recall (step 5).

For orientation, the single-topic numbers already on disk, position 0, centred:
`bg_think` 0.404, `baseline` 0.289, untrained floor 0.0013.

Gate 3 asks only that `bg_think_many` clears the untrained floor by a wide
margin. Failing that means the run is broken, not that the hypothesis is wrong.

Also run the **raw (uncentred)** pass, for the record. On single-topic vectors
both trained adapters collapsed there (0.042 and 0.028), and whether the
multi-topic training changes that is directly relevant to the OOD tasks, which
all read raw activations.

## 5. The OOD evaluations -- the actual object of the experiment

These are **not gates** (parent plan §6). A negative result here is the finding,
and must be reported plainly rather than explained away.

Run each existing evaluation with `bg_think_many` and compare against the
`baseline` and `bg_think` runs already on disk. Reuse the existing harnesses; do
not write new ones:

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, against
   `outputs/taboo_baseline` and `outputs/taboo_bg_think`.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, against
   `outputs/taboo_assistant_bg_think`. Note `outputs/taboo_assistant_baseline`
   is currently an empty reserved directory (`outputs/README.md`); if the
   baseline arm was never run, either run it or say clearly that the comparison
   is missing. Do not quietly compare against nothing.
3. **Bridge entity (TwoHopFact)** -- the `bridge_entity/` sweep, against
   `outputs/bridge_entity`.
4. `compare_taboo_arms.py` for the side-by-side.

Match every setting to the existing runs -- same layers, positions, prompts,
decoding and question set. If a setting cannot be matched, say which one and
why, rather than reporting a number that is not comparable.

**One thing to be careful about when reading these.** The taboo task has a
single secret word, so it exercises the k=1 behaviour of an adapter trained on a
1:2:3 mixture that is mostly k>=2. If `bg_think_many` does worse than `bg_think`
on taboo, "the mixture traded single-concept accuracy away" and "multi-topic
training does not help" are different explanations, and the k=1 validation slice
(§3) plus the k=1 retrieval slice (§4) are what tell them apart. Report both
before drawing a conclusion.

## 6. Housekeeping

- Save the checkpoint to `outputs/adapters/bg_think_many/` and **add an entry to
  `outputs/adapters/README.md`**, in the style of the existing `bg_think` entry:
  what it was trained on, the architecture, the budget. That file already has a
  truncated `bg_think` entry ending in a bare "1" -- finish it while you are
  there if you know what it meant, or leave it and say so.
- Add the new extraction and evaluation directories to `outputs/README.md`.
- Keep commits small and self-contained (project rule).

## 7. The report

`plans/bg_think_many/notes/step6_results.md`, and a summary to the user. It must
contain:

- the research question, quoted from the parent plan §1, not paraphrased
- the run's configuration and realised cost against the ~5-6 A100-hour estimate
- Gate 2: whole-mixture and per-k validation loss, with the k=1 comparison to
  1.4844
- Gate 3: set-level recall per k, centred and raw, every number with its
  `segments` block, against all four adapters
- the three OOD comparisons, each stated plainly as better, worse or
  indistinguishable, with the numbers
- **the confound from parent plan §7**, restated: this run changed both the data
  and the architecture, so an improvement cannot be attributed to the multi-topic
  data alone. If the OOD numbers did improve, recommend the ~3-4 A100-hour
  control run (rank-64 on the existing single-topic vectors at the same budget)
  and let the user decide.
- anything that contradicted this plan. Those are the most valuable lines in the
  note.

Then move `plans/bg_think_many/` into `plans/archive/` and update
`plans/CLAUDE.md`, per that file's own instruction.
