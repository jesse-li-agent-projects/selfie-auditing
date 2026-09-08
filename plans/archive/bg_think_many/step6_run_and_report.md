# Step 6: train bg_think_many, evaluate it, and report

Part 6 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read §6 Gates, §7 and D7, D9, D12).

**Depends on all of steps 1-5 being merged.** Check that before booking GPU
time; a half-merged step here wastes the most expensive hours in the plan.

**This is the large GPU step**: ~5-6 A100-hours for the training run, plus two
retrieval passes (§4) and the OOD evaluations (§5). **One agent at a time on
GPU work** (project rule). Do not start while step 2's extraction, or any other
agent's GPU job, is running.

## Research question

Quoted in the parent plan §1. Read it there; do not restate it in your own
words, here or in the report.

## 0. Code that must land before the run

**`step6a_pre_run_code.md`.** An audit of this file against the code
(2026-09-07) found four things step 6 assumes and the codebase does not have:
per-k validation loss for Gate 2, recomputed and equally-weighted pooled means
(D13, D14), pooled centring in the retrieval eval for §4, and a vector path that
does not need ~80 GiB of host RAM.

None of it needs a GPU, so it is a separate step and a separate agent. Do not
book GPU time until it has merged.

## 1. Preconditions

- `outputs/bg_think_l19` (k=1, reused per D1), `outputs/bg_think_many_l19_k2`
  and `outputs/bg_think_many_l19_k3` all exist, and step 2's Gate 1 passed.
- Step 1's bucketing rule is merged, and step 3 uses it.
- Step 4's `--log-every` and step 5's `score_sets` are merged.
- The unit tests pass. **Note the project rule: passing unit tests do not mean
  the codebase works.** They cover parts of this pipeline only.

## 2. The training run

    python -m adapter_training.train_adapter \
        --vectors-k 1=bg_think_l19 \
        --vectors-k 2=bg_think_many_l19_k2 \
        --vectors-k 3=bg_think_many_l19_k3 \
        --mixture-ratio 1:2:3 \
        --run-dir adapters/bg_think_many \
        --budget-examples 1510782 --batch-size 256 --micro-batch-size 16 \
        --projection-type scalar_affine_plus_low_rank --projection-rank 64 \
        --low-rank-init-factor 0.01 \
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 \
        --weight-decay 0.01 --seed 42 \
        --validate-every 100 --log-every 50 --val-subsample 5000 \
        --val-total-examples 300000

5,902 steps. Every hyperparameter above except the mixture, the budget,
`--micro-batch-size`, `--log-every` and `--val-total-examples` is what
`bg_think` used and what upstream's `scalar_plus_low_rank_8b.yaml` specifies
-- see the parent plan §3.
Do not tune them; a tuned run would not be comparable to `bg_think`.

**`--micro-batch-size` is required here and is a memory bound, not a
hyperparameter** -- it does not change the gradient, only how the batch is
chunked to compute it. It defaults to `--batch-size`, so leaving it out asks
for all 256 examples at once and dies immediately. It sets the count at the
pool's *worst* target length; short-target batches take more. The mixture's
composed k=2/k=3 labels reach 77 tokens against a 29-token median, and
because batches are length-bucketed the long ones arrive together, so a size
that survives a typical batch says nothing about the tail. Measured peaks on
the longest bucket, 31.36 GiB card: 16 -> 23.4 GiB, 24 -> 27.6 GiB, 32 ->
OOM. Step time is set by target length (1.6 s short to 4.9 s worst), not by
this flag, so raising it buys almost no throughput -- pick it for headroom.
Scale it for a larger card.

**`--val-total-examples` is not the train/val split** -- the split is over
topics and is fixed in the extraction directories. It is how many *examples* to
draw from the val side, which has to be chosen because grouped examples are
sampled, not enumerated (D4). `bg_think` needed no such number: it enumerated
every (val vector, label) pair, 79,391 val labels x 10 positions = 793,910. A
k=3 group spans ~29,000 composed labels per position, so the same exhaustive
pool does not exist here.

**300,000, confirmed with the user (2026-09-07).** It is a judgement call, not a
derived number. It splits 1:2:3 into 50,000 / 100,000 / 150,000. The val side holds ~46,800 k=1,
~46,600 k=2 and ~77,100 k=3 vectors, so that is roughly one to two examples per
val vector at every k, and the k=1 slice is 10x `--val-subsample`, which is
enough to compare against 1.4844. It is also cheaper than `bg_think`'s final
eval, not dearer. Note this sizes the **loss** pool only; Gate 3's retrieval
fraction is sized by the val groups in the extraction directories and is
untouched by this flag.

`--vectors-k` and `--run-dir` values are relative to `outputs/`, which the
script prepends -- do not write the prefix yourself.

Use `--resume` if it crashes. It restores the projection, the optimizer, the
step and the best-val from `run_dir/resume.pt`, written on every validation
step, so at most 100 steps of work is redone. Note that a restart re-pays the
whole vector load, so §0(c) matters for restarts too.

**Watch `low_rank_norm` and `low_rank_to_diagonal_ratio`** in `metrics.jsonl`
(step 4 added them). If the ratio stays near zero throughout, the rank-64
component never engaged and the run is effectively `scalar_affine` with extra
steps -- which is worth knowing *during* the run, not after.

## 3. Gate 2: validation loss

From `final_eval.json`. The per-k slices need §0(a) -- step 3 computes the
ranges but nothing currently keeps the val ones.

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

**Scope, set by the user (2026-09-07): the new adapter only, pooled centring
only, two runs.** An earlier draft of this section asked for four adapters in
two centring conditions, which is 24 GPU passes; that was a mistake and is
withdrawn.

| run | vectors | what it answers |
|---|---|---|
| 1 | `outputs/bg_think_l19` (k=1) | single-topic recall for the new adapter |
| 2 | all three k directories | set-level recall on the mixture, broken down per k (D12) |

Both score `bg_think_many`'s `best.pt`, both **centred against the pooled
reference** (D13) -- the condition the adapter trained in. Decoding is identical
in both (D10: `--max-new-tokens 110`, temperature 0.7, `n_samples` 1, seed 42).
Always report the `segments` block beside each recall (step 5).

Run 2 needs §0(b): one invocation over several directories, so the pooled mean
and the retrieval index are built once. **Report run 2 per k, never pooled
across k** (D12) -- the two-run scope changes how many invocations there are,
not how the numbers are broken down.

**Two consequences of this scope, to state in the report rather than paper
over.**

- *Gate 3 no longer has a floor of its own.* The gate is "clear the untrained
  floor by a wide margin", and the untrained arm is not being run. Use the
  existing single-topic figure of 0.0013
  (`outputs/retrieval_reports/untrained_floor_centred.json`) as the
  order-of-magnitude reference: an
  untrained projection scores at chance whatever the centring, so a floor is a
  floor. If `bg_think_many` lands anywhere near 0.0013 the run is broken, which
  is all this gate was ever asked to catch.
- *Run 1 is not comparable to `bg_think`'s 0.404.* That number was measured
  under per-directory centring; run 1 uses the pooled reference, which carries a
  constant per-position offset `bg_think` never saw. D13 already flags this for
  Gate 2's loss, and it applies here identically. Report run 1 as a standalone
  number, and do not put 0.404, 0.289 or the raw-vector figures (0.042, 0.028)
  beside it as though they were a comparison.

## 5. The OOD evaluations -- the actual object of the experiment

These are **not gates** (parent plan §6). A negative result here is the finding,
and must be reported plainly rather than explained away.

Run each existing evaluation with `bg_think_many` and compare against the
`baseline` and `bg_think` runs already on disk. Reuse the existing harnesses; do
not write new ones:

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, against
   `outputs/taboo_baseline` and `outputs/taboo_bg_think`.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, against
   `outputs/taboo_assistant_bg_think`. **There is deliberately no baseline arm
   here**: the user decided (2026-09-07) not to reproduce it, because the
   paper's own reported results can be read directly and
   `taboo_assistant_bg_think` was not promising enough to justify the GPU time.
   `outputs/taboo_assistant_baseline` has been deleted; do not recreate it or
   run the arm. Report this comparison as `bg_think` only, and say the baseline
   is the paper's reported figure rather than a run on this machine.
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
  what it was trained on, the architecture, the budget.
- Add the new extraction and evaluation directories to `outputs/README.md`.
- Both of those files are under the gitignored `outputs/` and are not tracked,
  so they are local edits, not part of any PR.
- Keep commits small and self-contained (project rule).

## 7. The report

`plans/bg_think_many/notes/step6_results.md`, and a summary to the user. It must
contain:

- the research question, quoted from the parent plan §1, not paraphrased
- the run's configuration and realised cost against the ~5-6 A100-hour estimate
- Gate 2: whole-mixture and per-k validation loss, with the k=1 comparison to
  1.4844 and D13's caveat that the pooled centring shifts it
- the chosen `--val-total-examples`, and why
- Gate 3: the two runs of §4, pooled-centred, with the per-k breakdown for run
  2 and a `segments` block beside every recall; plus the note that run 1 is not
  comparable to `bg_think`'s 0.404 and that the gate's floor is borrowed
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
