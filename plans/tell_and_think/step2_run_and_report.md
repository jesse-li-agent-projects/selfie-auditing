# Step 2: train tell_and_think, evaluate it, and report

Part 2 of 2 in the execution of `plans/tell_and_think/tell_and_think.md` (the
parent plan -- read §5 D2-D5, §6 Gates, §7 and §8).

**Depends on step 1 being merged.** Check that before booking GPU time.

**This is the large GPU step.** Budget roughly **14-16 GPU-hours** end to end and
read §6 before believing a smaller number. **One agent at a time on GPU work**
(project rule). Do not start while another agent's GPU job is running.

## Research question

Quoted in the parent plan §1. Read it there; do not restate it in your own
words, here or in the report.

## 1. Preconditions

- Step 1 is merged, and its tests pass. **Passing unit tests do not mean the
  codebase works** (project rule); they cover the mixture builder only.
- All four source directories exist: `outputs/baseline_l19`,
  `outputs/bg_think_l19`, `outputs/bg_think_many_l19_k2`,
  `outputs/bg_think_many_l19_k3`.
- `outputs/adapters/bg_think_many/` and
  `outputs/adapters/wikipedia-scalar-affine.safetensors` are present, for the
  comparisons.
- The predecessor OOD outputs are present, so the settings can be matched:
  `outputs/taboo_bg_think_many`, `outputs/taboo_assistant_bg_think_many`,
  `outputs/bridge_entity`.

## 2. Gate 1 before the run (parent plan §6)

Cheap, CPU-only, and it protects the most expensive hours in the plan. Assert
from the built example lists, not from the flags:

1. Each source's train and val example counts match D4's table exactly.
2. Every example's `vector_index` in a source's range falls inside that source's
   own global offset range (§2.4 of step 1 -- the separator check does not work
   here).
3. The `bg*` group's pooled mean is bit-identical to the one `bg_think_many`
   used. This is what keeps the `bg*` slices on `bg_think_many`'s footing, so
   check it rather than assume it.
4. Record the exact `;`-filter drop counts per source (D9 expects 10 topics for
   `tell`).

## 3. The training run

    python -m adapter_training.train_adapter \
        --vectors-source tell=baseline_l19 \
        --vectors-source bg1=bg_think_l19 \
        --vectors-source bg2=bg_think_many_l19_k2 \
        --vectors-source bg3=bg_think_many_l19_k3 \
        --mixture-ratio 3:1:2:3 \
        --centring-group tell=tell \
        --centring-group bg1=bg --centring-group bg2=bg --centring-group bg3=bg \
        --run-dir adapters/tell_and_think \
        --budget-examples 2266173 --batch-size 256 --micro-batch-size 16 \
        --projection-type scalar_affine_plus_low_rank --projection-rank 64 \
        --low-rank-init-factor 0.01 \
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 \
        --weight-decay 0.01 --seed 42 \
        --validate-every 100 --log-every 50 --val-subsample 5000 \
        --val-total-examples 450000

8,853 steps. Every hyperparameter except the sources, the ratio, the centring
groups and the two budgets is what `bg_think_many` used (parent plan D5).
**Do not tune them.** Not because comparability is the deliverable -- parent
plan §8 says it is not -- but because a tuned run answers a different question
than the one asked, and there is no budget to do both.

Four things `bg_think_many`'s step 6 got wrong on this exact command, all
already corrected above. Do not reintroduce them:

- **No `outputs/` prefix** on `--vectors-source` or `--run-dir` values; the
  script prepends it. The predecessor resolved `outputs/outputs/...` and died.
- **`--micro-batch-size 16` is not optional.** It defaults to `--batch-size`,
  and on a 31.36 GiB card 256 OOMs immediately. 16 peaked at 23.4 GiB, 24 at
  27.6, 32 OOMed. Throughput is nearly independent of this flag here -- do not
  spend GPU time tuning it for speed.
- Use `--resume`, so a crash is relaunched with the same command.
- On the remote, `--run-dir` and every `--vectors-source` path must be
  **absolute**; they resolve against the cwd, which is a read-only synced
  worktree.

Also from the predecessor's notes, worth knowing before the run rather than
after:

- `run_config.json`'s `git_commit` records `null` on a synced worktree (the
  `.git` file points at a gitdir that does not exist remotely). Record the
  commit by hand and verify the remote sources are md5-identical to it.
- `outputs/` syncs remote -> local only, so any local input file is not on the
  remote. Materialise inputs through the worktree, which syncs local -> remote.

## 4. Gate 2: validation loss, per source, as a diagnostic

Report all four per-source slices, never a pooled figure (parent plan D7).

**This gate is not a comparison.** It asks only whether the run trained: every
slice finite, every curve converging, no slice stuck or diverging, nothing
absurd relative to its own curve. `bg_think_many`'s slice losses and the
published checkpoint's `best_val_loss` are **not** valid references -- D7 was
corrected by the user on 2026-09-09 because validation loss is not comparable
across these runs, and not only because the architectures differ: each run's
loss is computed over a different training distribution, so it is a different
task. Do not pass or fail the run on any of them, and do not quote a published
figure without saying in the same sentence that it is not a comparison.

A slice that looks anomalous is a reason to re-check Gate 1's assertions, not a
verdict on the adapter. All evidence about whether `tell_and_think` is better
than `bg_think_many` comes from §6.

## 5. Gate 3: set-level retrieval is not catastrophic

Score `best.pt` at `--max-new-tokens 110`, temperature 0.7, `n_samples` 1, seed
42, against the full 49,637-topic index, centred per D3, reporting per source.
The untrained floor is **0.00068** aggregate recall@1 -- not 0.0013, which
`bg_think_many`'s notes correct as a position-0 figure measured on a different
directory. Report the `segments` histogram beside every score.

**Pass unless aggregate recall@1 is below 3x the floor (0.00204)** (parent plan
§6). This catches a broken adapter and nothing more; an unimpressive score is
not grounds to withhold the OOD arms, because OOD generalisation is
unpredictable. Do not add a `tell` prior at this decoding length -- none is
wanted.

`bg_think_many`'s figures (0.5646 / 0.2206 / 0.0911 recall@1 at k=1/2/3) are
informative **only** for the `bg*` sources, and only because D3 leaves their
centring unchanged.

## 6. The evaluations (parent plan §7)

Three arms, matched to `bg_think_many`'s settings so the numbers compose. None
is a gate; a negative result is the finding. **Read parent plan §7 first**: the
taboo task and the bridge-entity task carry the result equally, and arms 1 and 2
are two harnesses over *one* task, split for historical reasons.

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, matched to
   `outputs/taboo_bg_think_many`'s sidecar. Compare **matched** on (organism,
   word, layer, position): the best cell is a maximum over 352 and that
   selection effect alone moves a headline number. `finetuned` is the only
   organism that genuinely conceals the word, so it is the one that answers the
   question. This harness only ever covered book and chair, and the predecessor
   found book to be the one word where it underperformed -- so treat a null
   result here as possibly a word-selection artefact, and say so.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, matched to
   `outputs/taboo_assistant_bg_think_many`. **Run book and chair only**, per the
   user (2026-09-09), to hold evaluation cost down -- not the predecessor's four.
   Extending to blue and salt is a follow-up if this run's result is good enough
   to warrant it, and the report should say whether it is.
3. **Bridge entity (TwoHopFact)** -- raw uninjected activations, no mean
   subtraction. Priors: `baseline` 89/100, `bg_think` 88/100, `bg_think_many`
   70/100. Detection rate is near ceiling for the first two and so has little
   power; **`generation_hit_rate`, over 67,488 cells, is the metric with
   resolution** (2.15% / 1.74% / 0.62%). Report both.

**Arms 1 and 2 need a combined reading, not just two tables.** With arm 2 cut to
book and chair, both harnesses now cover the **same two words**, so a direct
cross-harness reading is available throughout. Match each arm to its own
predecessor for the numbers, then state what the taboo task as a whole shows.
Where a conclusion holds in one harness and not the other, name which and treat
the disagreement as the finding.

**Comparison arms: `baseline` (primary) and `bg_think_many` (secondary)**, per
the user and parent plan §8. `bg_think`'s figures are context only, not an arm.

### Realised cost, from the predecessor

`bg_think_many` measured ~12 h 35 m total on an RTX 5090 (31.36 GiB), of which
the **evaluations cost as much as the training** -- 6 h 20 m against 6 h 15 m.
This run's training is 1.5x the steps (8,853 against 5,902), though a large
share of the added examples are short `tell` targets and short batches run at
~1.6 s/step against ~4.9 s on the longest, so scaling linearly overestimates.
Expect roughly 6-7 h training, ~1 h 40 m for the 450,000-example final eval
(the predecessor's 300,000 took 1 h 05 m), and ~6 h 20 m of gates and OOD arms.

## 7. The report

Write `plans/tell_and_think/notes/step2_results.md`. It must contain:

- The realised cost table, against this file's estimate.
- **"What contradicted the plan"** -- the most valuable section, per the
  predecessor's note. Anything this file assumes and the code does not have.
- Gate 1, 2, 3 results. Gate 2 is reported as a diagnostic only; if any
  published validation loss appears at all, the same sentence must say it is not
  a comparison (D7).
- The three OOD arms, matched-cell comparisons, against `baseline` first and
  `bg_think_many` second (parent plan §8). Beating `baseline` is the goal the
  run was set for; say plainly whether it did.
- **What is acknowledged and accepted** (parent plan §8), stated without
  hedging the result: the stretched cosine schedule (8,853 steps against 5,902),
  the `tell` centring rule differing from `bg_think_many`'s, and `bg1`'s 0.54
  draws/vector. This run was not designed to isolate a cause, so do not report
  it as though it were.
- D2's vector re-use (`tell` draws 755,391 examples from 44,673 distinct train
  vectors, ~17 each) and D8's population asymmetry (2,636 topics seen only
  through `tell`), both stated up front rather than discovered in the analysis.
  Note that `tell` draws *every* vector it has, so this is re-use and not a
  data-diversity deficit (parent plan §8).

Then archive the plan: move `plans/tell_and_think/` into `plans/archive/` and
update `plans/CLAUDE.md`, whose "Background thinking (bg_think)" section is the
running summary of this line of work.
