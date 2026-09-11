# Step 2 results: train tell_and_think, evaluate it, and report

Execution note for `plans/tell_and_think/step2_run_and_report.md`. One RTX 5090
(32,607 MiB), one run, no reruns of any stage.

Two facts the plan asks for up front, so they frame the analysis rather than
surface inside it:

- **D2's vector re-use.** `tell` contributes 755,260 train examples over 44,665
  distinct vectors, ~16.9 each. `tell` is used *whole*, so every vector and
  every label it has is seen exactly once: this is re-use of the vector, not a
  data-diversity deficit.
- **D8's population asymmetry.** 2,635 topics are seen only through `tell` once
  D9's `;` filter has run (2,636 before it). That is 5.3% of `tell`'s
  population, and it is the cost of keeping the upstream population whole.

## Did it beat `baseline`?

**No.** Beating `baseline` was the goal the run was set for, and on the one arm
where a `baseline` comparator exists across the whole task -- bridge entity --
`tell_and_think` reaches 77/100 detection and 0.79% generation hit rate against
`baseline`'s 89/100 and 2.15%. On the taboo user-prompt arm it is roughly level
with `baseline` on `book` and behind on `chair`.

**It did beat `bg_think_many`,** its immediate predecessor, on most measures:
bridge entity 77/100 against 70/100, and `book` improves in both taboo
harnesses. Adding the original paper's `tell` data recovered part of the ground
`bg_think_many` lost relative to `baseline`, without closing the gap.

## Realised cost

| stage | estimate | realised |
|---|---|---|
| training (8,852 steps) | 6-7 h | **7 h 47 m** |
| final full-val pass | 1 h 40 m (for "450,000") | **1 h 28 m** (384,183) |
| Gate 3, both invocations | -- | **37 m** |
| arm 1, taboo user tokens | -- | **1 h 03 m** |
| arm 2, taboo assistant tokens | -- | **1 h 34 m** |
| arm 3, bridge entity | -- | **59 m** |
| gates + arms subtotal | ~6 h 20 m | **4 h 13 m** |
| **total** | **14-16 h** | **13 h 44 m** |

Throughput averaged 3.03 s/step. An early reading of 1.80 s/step was not
representative: length bucketing (D6) means the rate swings with whichever
source's targets a batch holds, and the early batches were short `tell` targets.

## What contradicted the plan

The most valuable section, per the predecessor's note.

**1. `--index-cache` is a directory, and the path the plan gives does not
exist.** The plan asks for `retrieval_reports/gte_index.pt`; `build_or_load_index`
treats the value as a directory and looks for `embeddings.pt` inside it. The
real artefact is `retrieval_reports/index_level5_gte_large.pt/`, holding
`embeddings.pt` + `metadata.json`. Following the plan literally would have
rebuilt the whole 49,637-topic index. Verified before use: 49,637 titles,
`thenlper/gte-large`, strategy `title_plus_all_labels`, which is
`DEFAULT_INDEX_STRATEGY`.

**2. `evaluate_retrieval` needs the topic corpus even with a cached index**, and
dies without egress. `main` calls `load_topics` unconditionally for the query
side, so a cached index does not remove the dependency. On a no-egress box
`load_dataset` raises `ConnectionError: ... (OfflineModeIsEnabled)`. Fix:
`--dataset-file` pointed at the JSONL inside the fetched dataset snapshot
(`wikipedia_vital_articles_level5_dataset.jsonl`, 49,637 lines, fields
`original_title`/`prompt`/`labels`/`split`). This is a prerequisite for Gate 3,
not a detail.

**3. The `outputs/` prefix is per-flag, and now demonstrably inconsistent
across scripts.** The predecessor found `--report` unprefixed while
`--index-cache` is prefixed. Extending that: `run_pipeline.py`'s
`--output-dir` is **not** prefixed (`Path(args.output_dir)`), while
`report_bridge_entity`'s `--run` values **are**. `--plot-dir` is still
unprefixed, and wrote `plot/bridge_entity_detection.png` into the source tree
again; moved by hand to `outputs/bridge_entity/plot_tell_and_think/`. Pass
absolute paths and check each flag individually.

**4. `taboo_baseline` is laid out per word.** It stores
`taboo_baseline/<word>/results_*.jsonl`, not the single flat file every later
run writes. Any matched comparison against `baseline` must load and merge both
subdirectories.

**5. Arm 2 has no `baseline` comparator at all.** Only
`taboo_assistant_bg_think` and `taboo_assistant_bg_think_many` exist; the
published adapter was never run through the assistant-token harness. So the
plan's "compare against `baseline` first" is satisfiable for arms 1 and 3 only.
No substitute was used in its place.

**6. Peak memory at micro-batch 16 was higher than recorded.** The plan cites
23.4 GiB; realised was 25.9-31.1 GiB of 31.8, with one recoverable
`CUDACachingAllocator` OOM warning at startup (free memory momentarily 248 MB).
The run survived and never OOMed again. Since 24 peaked at 27.6 GiB on the
predecessor's lighter profile, 24 would very likely have failed here. The
predecessor's figure does not transfer: this mixture's batch-length profile
differs because of `tell`.

**7. `git_commit` is `null`, as predicted.** The synced worktree's `.git` points
at a gitdir that does not exist remotely. Recorded by hand: **`65d6e73`**, with
the four `adapter_training` source files verified md5-identical between local
and remote before launch.

## Gate 1

Reported as passed on the user's instruction (checked earlier by another
agent); not re-derived here. Independently corroborated by the run itself,
which built **2,266,042 train / 384,183 val** examples -- D4's table exactly.

## Gate 2: validation loss, per source, as a diagnostic

Produced by the trainer itself: `final_eval.json`'s `val_loss_by_source`, over
the full val split. No separate script is needed.

| source | val loss | val examples |
|---|---|---|
| `tell` | 1.2261 | 84,183 |
| `bg1` | 1.2999 | 50,000 |
| `bg2` | 1.6654 | 100,000 |
| `bg3` | 1.8971 | 150,000 |
| *pooled* | *1.6120* | *384,183* |

**This is a diagnostic, not a comparison** (D7). What it says: the run trained.
Every slice is finite, none is stuck or diverging, and the ordering is the one
the data implies -- loss rises with the number of topics a label must name. The
subsampled curve fell monotonically-ish from 2.079 to a best of 1.6203 across 88
validations, with three small upticks each recovered by the next point. No
published figure is quoted, because none of them measures the same task.

## Gate 3: set-level retrieval is not catastrophic -- passed

Run as two invocations (`tell` on its own mean; `bg1/bg2/bg3` pooled over those
three only), which reproduces D3's two groups with no code change.

| source | recall@1 | recall@5 | segments histogram | wrong-count rate |
|---|---|---|---|---|
| `tell` (k=1) | **0.6867** | 0.8278 | `{1: 4,964}` | **0.000%** |
| `bg1` (k=1) | **0.5934** | -- | `{1: 46,784, 2: 6}` | 0.013% |
| `bg2` (k=2) | **0.2175** | -- | `{1: 137, 2: 46,395, 3: 57, 4+: 1}` | 0.42% |
| `bg3` (k=3) | **0.0892** | -- | `{1: 59, 2: 149, 3: 76,842, 4+: 30}` | 0.31% |

Every source clears 3x the untrained floor (0.00068, so threshold 0.00205) by
two to three orders of magnitude. On the `bg*` sources, where `bg_think_many`'s
figures are a valid reference because D3 leaves their centring unchanged:
k=1 **+0.029** (0.5934 vs 0.5646), k=2 -0.003 (0.2175 vs 0.2206), k=3 -0.002
(0.0892 vs 0.0911). `tell` has no valid comparator, being centred on its own
mean.

**No cross-contamination.** `tell`'s histogram was recovered from the saved
generations with `retrieval_eval.split_segments`, since the single-directory
path scores with `score` and records no segments block. All 4,964 of 4,964
generations produced exactly one segment; no `;` appeared anywhere. The adapter
learned the separator as a property of the vector it reads, not as a habit it
applies to every vector -- a check no predecessor run could make.

## The OOD arms

### Arm 1 -- taboo, user-prompt tokens

Matched on (organism, word, layer, position); all three runs hold the full
2,112 cells. `finetuned` is the organism that genuinely conceals, so it is the
one that answers the question.

| comparison | word | mean | best cell | win/tie/loss |
|---|---|---|---|---|
| vs `baseline` | book | 0.0181 vs 0.0126 (+0.0055) | 0.440 vs 0.490 | 115/127/110 |
| vs `baseline` | chair | 0.0014 vs 0.0023 (-0.0009) | 0.135 vs 0.210 | 26/307/19 |
| vs `bg_think_many` | book | 0.0181 vs 0.0097 (+0.0084) | 0.440 vs 0.370 | 130/152/70 |
| vs `bg_think_many` | chair | 0.0014 vs 0.0040 (-0.0026) | 0.135 vs 0.235 | 21/290/41 |

Against `baseline`: level on `book` (better mean, *worse* best cell), behind on
`chair`. Against `bg_think_many`: better on `book`, worse on `chair`. The same
book-improves/chair-regresses pattern holds for all three organisms, so it looks
like a property of the word, not of concealment.

Two cautions. The means are dominated by zeros -- for `chair`, 290-318 of 352
cells are 0 vs 0, so a 0.002 mean difference rests on a handful of cells. And
`book` is exactly the word the predecessor underperformed on, which the plan
flags as a possible word-selection artefact; improving there may be partly
recovery of a known weak spot rather than a general gain.

### Arm 2 -- taboo, assistant tokens

`book,chair` only, per the user (2026-09-09). 2,496 cells. **Precondition
checked:** both runs elicited byte-identical replies for every (organism, word),
without which the per-token cells would not line up at all.

| comparison | word | mean | best cell | win/tie/loss |
|---|---|---|---|---|
| vs `bg_think_many` | book | 0.0143 vs 0.0044 (+0.0099) | **0.655 vs 0.125** | 229/333/78 |
| vs `bg_think_many` | chair | 0.0013 vs 0.0015 (-0.0002) | 0.240 vs 0.165 | 40/543/57 |
| vs `bg_think` | book | 0.0143 vs 0.0058 (+0.0085) | 0.655 vs 0.515 | 256/307/77 |
| vs `bg_think` | chair | 0.0013 vs 0.0000 (+0.0012) | 0.240 vs 0.015 | 43/594/3 |

`finetuned` produces 640 cells per word against `control`'s 256, because it
talks around the secret at greater length before stopping.

### Arm 3 -- bridge entity (TwoHopFact)

Raw uninjected activations, no mean subtraction. 67,488 cells, matching the
predecessor's count exactly. The recorded priors reproduced exactly, which
validates the comparison.

| run | detection | generation hit rate |
|---|---|---|
| `baseline` | 89/100 (89.0% +- 6.1%) | 2.15% |
| `bg_think` | 88/100 (88.0% +- 6.4%) | 1.74% |
| `bg_think_many` | 70/100 (70.0% +- 9.0%) | 0.62% |
| **`tell_and_think`** | **77/100 (77.0% +- 8.2%)** | **0.79%** |

Detection is near ceiling for `baseline`/`bg_think` and so has little resolving
power; `generation_hit_rate` over 67,488 cells is the metric with resolution.
On both, `tell_and_think` sits between `bg_think_many` and `baseline`, closer to
the former. The detection intervals for `tell_and_think` and `bg_think_many`
overlap substantially, so 77 vs 70 is a direction, not a demonstrated gap.

### The taboo task read as a whole

Arms 1 and 2 are two harnesses over one task, and with arm 2 cut to `book,chair`
they now cover the same two words, so they can be read directly against each
other.

- **They agree on `book`**: it improves against `bg_think_many` in both
  harnesses. Agreement across two different views of the same concealment is the
  meaningful part, and the assistant-token best cell (0.655 vs 0.125) is where
  the hypothesis predicts -- the model commits to its secret while speaking.
- **They disagree on `chair`**: worse on user tokens, flat-to-slightly-better on
  assistant tokens. The honest statement for `chair` is that no effect was
  demonstrated. Per the plan, that disagreement is itself the finding.
- Against `baseline`, only arm 1 can speak, and it does not show an improvement.

## What is acknowledged and accepted

Stated without hedging the result:

- **The cosine schedule is stretched**: 8,852 steps against `bg_think_many`'s
  5,902, because `tell`'s examples are added on top of frozen `bg*` counts (D4).
- **`tell`'s centring rule differs** from `bg_think_many`'s: its own mean, not
  the pooled one (D3).
- **`bg1` contributes 0.54 draws/vector.**

This run was not designed to isolate a cause, and is not reported as though it
were. It changed the data and nothing else relative to `bg_think_many`, but it
is not a clean single-variable ablation of that change, because the added
examples move the step count and so the schedule.

## Artefacts

- Adapter: `outputs/adapters/tell_and_think/` (`best.pt`, `final_eval.json`)
- Gate 3: `outputs/retrieval_reports/tell_and_think_{tell,bg}.json`
- Arm 1: `outputs/taboo_tell_and_think/`
- Arm 2: `outputs/taboo_assistant_tell_and_think/`
- Arm 3: `outputs/bridge_entity/tell_and_think/`,
  `outputs/bridge_entity/report_with_tell_and_think.json`,
  `outputs/bridge_entity/plot_tell_and_think/`
