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
   used. This is what makes D5's comparison a single-variable one, so check it
   rather than assume it.
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
**Do not tune them** -- a tuned run would not be comparable to `bg_think_many`,
and comparability is this plan's entire value over its predecessor.

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

## 4. Gate 2: validation loss, per source

Report the table, never a pooled figure (D7).

| slice | prior | what the prior is |
|---|---|---|
| `tell` | 1.3662 | published checkpoint's `best_val_loss` -- **a floor, not a target** (D7) |
| `bg1` | 1.3294 | `bg_think_many`'s k=1 slice: like-for-like |
| `bg2` | 1.6505 | `bg_think_many`'s k=2 slice |
| `bg3` | 1.8797 | `bg_think_many`'s k=3 slice |

The 1.3662 caveat is not a formality: it was a plain `scalar_affine` projection
at 2,951 steps, so a rank-64 model should beat it and "beat it" is not a
finding. Landing *worse* than it is a bug signal.

A slice far better than its prior is also a bug signal -- re-check Gate 1's
assertions before believing it.

## 5. Gate 3: set-level retrieval

Score `best.pt` at `--max-new-tokens 110`, temperature 0.7, `n_samples` 1, seed
42, against the full 49,637-topic index, centred per D3, reporting per source
(D7). The untrained floor is **0.00068** aggregate recall@1 -- not 0.0013, which
`bg_think_many`'s notes correct as a position-0 figure measured on a different
directory. Report the `segments` histogram beside every score.

`bg_think_many`'s figures (0.5646 / 0.2206 / 0.0911 recall@1 at k=1/2/3) are
comparable **only** for the `bg*` sources, and only because D3 leaves their
centring unchanged. `tell` has no comparable prior at this decoding length.

## 6. The evaluations (parent plan §7)

Three arms, matched to `bg_think_many`'s settings so the numbers compose. None
is a gate; a negative result is the finding.

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, matched to
   `outputs/taboo_bg_think_many`'s sidecar. Compare **matched** on (organism,
   word, layer, position): the best cell is a maximum over 352 and that
   selection effect alone moves a headline number. `finetuned` is the only
   organism that genuinely conceals the word, so it is the one that answers the
   question. This harness only ever covered book and chair, and the predecessor
   found book to be the one word where it underperformed -- so treat a null
   result here as possibly a word-selection artefact, and say so.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, four words (book,
   chair, blue, salt), matched to `outputs/taboo_assistant_bg_think_many`. The
   two harnesses do not cover the same word lists; match each one to its own
   predecessor rather than intersecting them.
3. **Bridge entity (TwoHopFact)** -- raw uninjected activations, no mean
   subtraction. Priors: `baseline` 89/100, `bg_think` 88/100, `bg_think_many`
   70/100. Detection rate is near ceiling for the first two and so has little
   power; **`generation_hit_rate`, over 67,488 cells, is the metric with
   resolution** (2.15% / 1.74% / 0.62%). Report both.

Arm 3 is the one the parent plan §7 flags as the sharpest question this run
answers. Read that section's note about whose inference that framing is before
carrying it into the report.

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
- Gate 1, 2, 3 results, with the D7 caveat stated wherever 1.3662 appears.
- The three OOD arms, matched-cell comparisons, against **both**
  `bg_think_many` and `baseline`.
- **What this does not isolate** (parent plan §8): the `tell` centring rule
  differs from `bg_think_many`'s, so the `tell` slice carries a confound the
  `bg*` slices do not; and `bg_think_many`'s own rank-vs-data confound is
  untouched by this run.
- D2's vector re-use (`tell` draws 755,391 examples from 44,673 distinct train
  vectors) and D8's population asymmetry (2,636 topics seen only through
  `tell`), both stated up front rather than discovered in the analysis.

Then archive the plan: move `plans/tell_and_think/` into `plans/archive/` and
update `plans/CLAUDE.md`, whose "Background thinking (bg_think)" section is the
running summary of this line of work.
