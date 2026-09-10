# tell_and_think: a SelfIE adapter trained on the original paper's data *and* the background-topic data

Status: plan, nothing implemented. Parent plan; the execution steps are the
`step*.md` files beside it.

The adapter this plan produces is named **`tell_and_think`**, and lands at
`outputs/adapters/tell_and_think/`. It is measured against **`bg_think_many`**
(`outputs/adapters/bg_think_many/`, its immediate predecessor) and **`baseline`**
(`outputs/adapters/wikipedia-scalar-affine.safetensors`, the published upstream
checkpoint).

## 1. The question

Quoted verbatim from the user's request (2026-09-09). Do not paraphrase it, and
do not restate it in your own words in any report:

> The plan bg_think_many has just been completed (see the plan archives,
> plans/archive/bg_think_many ). Results were less promising than I would've
> hoped.
>
> I have one more idea - run the same test, but this time also include/merge the
> training data from the original SelfIE paper. I believe this corresponds to the
> SELFIE_TEMPLATE label prompt; the extracted activations will be different too.
> Not sure if we already have the extracted activations.

and, on ordering:

> One more thing - should we intersperse the datasets, i.e. the adapter doesn't
> see all the data from the baseline dataset, then all the data from the
> bg_think_many dataset, but rather the two interleaved?

Standing project question (README.md, Q1): "Will a SelfIE adapter correctly
uncover something the model is actively hiding?"

**What this plan does not claim.** No mechanism for *why* merging the original
paper's data should help is stated by the user, and none is assumed here. If you
write one into a report, mark it as your own inference and say so.

## 2. One correction to the request's terms, and one answer

Both were settled with the user before this plan was written. They are recorded
because the plan's shape depends on them.

**`SELFIE_TEMPLATE` is not a label prompt.** It is the *interpretation* prompt
("What is the meaning of ...?") injected at the embedding layer, and every
adapter in this project -- `baseline`, `bg_think`, `bg_think_many` -- already
shares it unchanged. What actually differs in the original paper's training data
is the **extraction** prompt: `Tell me about {title}.`, layer 19, one vector per
topic at the last prompt token. This repo calls that `prompt_style: "baseline"`,
and this plan calls the source **`tell`**. The merge is over extraction
populations, not over label prompts. Do not change `prompts.SELFIE_TEMPLATE`.

**The activations already exist, so no extraction step is needed.**
`outputs/baseline_l19` holds them: 49,637 topics (44,673 train / 4,964 val), one
vector each at the last prompt token, layer 19, Llama-3.1-8B-Instruct, 443 MB,
written by `adapter_training/extract_baseline_vectors.py`. This is why the plan
has two steps rather than the six `bg_think_many` needed.

## 3. Vocabulary

Inherits `bg_think_many`'s table. One term is added and one is re-pointed.

| term | meaning |
|---|---|
| **source** | one extraction output directory, with the population and prompt that produced it. Replaces `bg_think_many`'s use of *k* as an identity |
| **k** | how many topics one extraction prompt named. Now a *property* of a source, derived from its records, not a key |
| **`tell`** | the `outputs/baseline_l19` source: `Tell me about {title}.`, k=1 |
| **`bg1`/`bg2`/`bg3`** | the `bg_think`/`bg_think_many` sources at k=1/2/3 |

`tell` and `bg1` are both k=1. That collision is the whole reason step 1 exists.

## 4. What already exists, and what must be built

Read `plans/archive/bg_think_many/notes/step6_results.md` before anything else
-- it is the result this plan reacts to, and it names the confound (§8).

**Reusable unchanged:** every extractor (nothing is extracted here),
`loss.py`, `checkpoints.py`, `projection.py`, `inference.py`,
`label_complexity.py`, `topic_retrieval_eval.py`, `retrieval_eval.py`, and
`train_adapter.py`'s optimizer/sampler/resume machinery.

**Already plumbed, needs no code:** `--vectors-k K=DIR`, `--mixture-ratio`, the
memory-bounded `_MixtureVectors` mmap path, `sample_examples`, per-slice
validation loss, and pooled centring. The mixture machinery is generic over
"several directories, one ratio"; it is only the *keying* that has to change.

**To build**, one step file each:

| step | file | GPU? | what it delivers |
|---|---|---|---|
| 1 | `step1_source_keyed_mixture.md` | no | source-name keying, per-source centring groups, per-source val slices, exhaustive/sampled source policies |
| 2 | `step2_run_and_report.md` | yes, large | the training run, the evaluations, the report |

Step 1 needs no GPU and no network. Step 2 depends on it.

**One agent at a time on GPU work** (project rule).

## 5. Design decisions

Every decision below is settled by the user, on 2026-09-09 or 2026-09-10. Do not
relitigate one without saying which decision number you are reopening and why.
D7 and §7 were corrected by the user after an earlier draft claimed they were
settled when they were not; both now record what was actually decided.

**D1 — The adapter is named `tell_and_think`**, per the user, after the two
extraction prompts it merges: `Tell me about X.` and `Think about X while
writing Y.`

**D2 — Four sources. `tell` is used whole; the `bg*` sources are sampled
1:2:3.** Shares per the user (2026-09-09); `tell`'s policy per the user
(2026-09-10).

| source | directory | k | policy |
|---|---|---|---|
| `tell` | `outputs/baseline_l19` | 1 | exhaustive |
| `bg1` | `outputs/bg_think_l19` | 1 | sampled, weight 1 |
| `bg2` | `outputs/bg_think_many_l19_k2` | 2 | sampled, weight 2 |
| `bg3` | `outputs/bg_think_many_l19_k3` | 3 | sampled, weight 3 |

The shares began as one 3:1:2:3 ratio, giving `tell` the same share as k=3.
That share turned out to be slightly *more than `tell` has*: it holds 755,260
distinct (vector, label) pairs in train, against the 755,391 the ratio asked
for, because it has **one vector per topic** where the pangram sources have
~10 positions per group. No sampler can return more pairs than exist, and
approaching the limit turns the draw loop into coupon collection for an answer
the data already forces. So `tell` is used whole -- which is what the ratio was
reaching for -- and its count is a property of the data rather than a number
anyone chose. The `bg*` weights keep their original meaning among themselves.

This is a **qualitative** difference between the sources, not a tuning choice:
`tell` is small enough to exhaust and the `bg*` sources are not (`bg3` alone
holds ~2x10^10 pairs). The code says so in its types -- `Exhaustive` against
`Sampled(weight)`, dispatched on the policy a caller passes, never on a size
comparison made at run time.

**D3 — Centring is per family: `tell` on its own mean, the `bg*` sources pooled
together.** Per the user. `bg_think_many`'s D14 pools the per-position mean
equally across sources, deliberately, to *preserve* the between-population
difference -- which there was "how many topics the prompt named", a thing worth
keeping. Here the between-population difference is "which extraction prompt",
which is exactly the prompt-structure bias `CLAUDE.local.md` names as the point
of subtracting a mean at all. Pooling it in would leave a large constant offset
in every vector. Two independent reasons back this up: `tell` has
`n_positions = 1` against the pangram sources' 10, so only position 0 would have
anything to pool with anyway; and centring `tell` on its own mean is what makes
its validation slice interpretable on its own terms (D7).

So: `tell` is its own centring group; `bg1`, `bg2`, `bg3` share one, computed
exactly as `bg_think_many` computed it. **The `bg*` group's pooled mean is
therefore bit-identical to `bg_think_many`'s**, which is what keeps D5's
comparison clean. Verify that rather than assume it.

**D4 — The budget grows; the `bg*` counts are frozen at `bg_think_many`'s.**
Per the user. Each `bg*` source keeps the exact example count it had, and `tell`
is added on top, so `tell_and_think` differs from `bg_think_many` in the data and
little else: same architecture, same hyperparameters, same background data, one
source added. Not a clean single-variable ablation, though -- the added examples
raise the step count and so stretch the cosine schedule (§8), which the user has
acknowledged and accepted. §8 also explains why isolating a cause is not this
run's goal anyway.

| source | train examples | val examples |
|---|---|---|
| `tell` (exhaustive) | 755,260 | 84,183 |
| `bg1` | 251,797 | 50,000 |
| `bg2` | 503,594 | 100,000 |
| `bg3` | 755,391 | 150,000 |
| **total** | **2,266,042** | **384,183** |
| *of which sampled* | *1,510,782* | *300,000* |

`--budget-examples` buys the sampled sources only, so it is the 1,510,782 --
which divides 1:2:3 exactly, with no remainder for `_split_by_ratio` to round,
and reproduces each `bg*` count to the example. `tell`'s inventory lands on top.
Keeping the budget on that footing is what freezes the `bg*` counts: they
depend on their own weights alone, and cannot drift if `tell`'s inventory ever
changes.

Realised total is **2,266,042** train, i.e. **8,852 optimizer steps** at batch
256, against `bg_think_many`'s 5,902. The `bg*` val slices keep
`bg_think_many`'s own 50k/100k/150k sizes, so each is directly comparable
number-to-number. `tell`'s val slice is its whole val inventory, 84,183 -- an
earlier draft asked for 150,000, which does not exist.

Every figure in this table was built and counted on CPU before the run
(2026-09-10), not derived on paper.

**D5 — The architecture does not move.** `scalar_affine_plus_low_rank`, rank 64,
`low_rank_init_factor` 0.01, and every hyperparameter `bg_think_many` used (lr
0.01, `init_scale` 5.0, clip 0.5, weight decay 0.01, warmup 10, cosine). This is
not a default -- it is deliberate. `bg_think_many` changed **both** the data and
the architecture relative to `bg_think`, which is why its notes say no OOD result
there can be attributed to the data alone. Holding the architecture still keeps
this run from compounding that. Do not tune anything: not because attribution is
the deliverable (§8 says it is not) but because a tuned run answers a different
question, and there is budget for one run.

**D6 — The sources are interleaved, and already are.** The user asked whether the
adapter sees one source's data in a block. It does not, and no code change is
needed: `build_mixture` concatenates the examples in source order, but that order
never reaches training. `example_stream` (`train_adapter.py:380`) reshuffles the
*entire* pool into a fresh permutation at the start of every pass. The
concatenation order survives only as the per-source index ranges, which are
validation bookkeeping.

One property to state rather than change. `bucketed_batches`
(`train_adapter.py:402`) fills a 12,800-example buffer, sorts it by target
length and cuts it into batches, so a batch holds examples of similar length --
which partially segregates sources. The split falls in a convenient place:

- **`tell` and `bg1` are length-indistinguishable** (both are single labels from
  the same corpus, median 54 characters), so they mix freely inside batches. The
  two populations whose contrast is the point of this experiment do share
  gradient steps.
- **`bg2` (~110 chars) and `bg3` (~166) segregate** into their own batches, so
  no single optimizer step averages a `bg3` gradient with a `tell` one. Batch
  *order* is reshuffled within each 50-batch buffer, so this is fine-grained
  alternation across ~50-batch windows, never a block of one source then a block
  of another.

Length bucketing is not optional: `bg_think_many`'s notes show peak memory is set
by the longest target in a batch, so unbucketed batches would force every batch
to the worst case and cut the micro-batch size for all of them. This property was
already true of `bg_think_many`; it is recorded here because the user asked.

**D7 — Validation loss is reported per source, never pooled, and no published
figure is treated as a comparison.** Extends `bg_think_many`'s D12 from k to
source. Pooling would weight `tell` and `bg3` most, by D2.

*Corrected 2026-09-09, by the user, who did not settle the original form of this
decision and rejects its premise.* An earlier draft offered two priors -- 1.3294
for `bg1` and 1.3662 for `tell` -- and called the first "like-for-like". **It is
not, and neither is.** Validation loss is not comparable across runs here, and
architecture is only half the reason: the *task* differs too, because each run's
loss is computed over a different training distribution. A number that is not
measuring the same thing is not a gate, however close it lands.

So: report the four per-source validation losses as **diagnostics**, not as a
pass/fail against anything. What they can tell you is whether the run trained at
all -- finite, converging, no slice stuck or diverging, no slice absurd relative
to its own curve. What they cannot tell you is whether `tell_and_think` is better
than `bg_think_many` or than the published checkpoint. **All evidence for that
question comes from the OOD arms (§7).**

If a report quotes 1.3294 or 1.3662, it must say in the same sentence that the
figure is not a comparison. Preferably it does not quote them at all.

**D8 — `tell` keeps its full 49,637-topic population.** `bg_think_l19` has
47,001 topics, every one of which is also in `baseline_l19`; `baseline_l19` has
2,636 more, dropped by the pangram fidelity filter. Keeping all of them means the
`tell` slice is the upstream population, so it is the population the research
question is asked about. Re-confirmed by the user on 2026-09-09 after D7's
priors were dropped, i.e. it stands on its own and not on D7. Strictly the
population is 49,637 minus D9's `;`-filter drops, i.e. 49,627 against `bg1`'s
46,992, so the realised asymmetry is **2,635** topics seen only through `tell`
-- one fewer than the 2,636 above, because `BoA` is among D9's drops. The 5.3%
asymmetry is the cost. Note it in the report.

**D9 — The `;`-label filter applies to `tell` too.** `drop_semicolon_topics`
exists because a composed label joins topics with `"; "`, and a label containing
a semicolon would teach the wrong segmentation. `tell`'s labels are never
composed, so the filter is not strictly required there -- but it costs **10
topics out of 49,637**, and a `tell` label with a semicolon would still teach the
adapter to emit a separator where no second topic exists. Apply it, for one
consistent rule across sources.

Measured, so Gate 1 has something to check against rather than re-derive:

| source | `;`-filter drops | topics kept |
|---|---|---|
| `tell` | 10 | 49,627 |
| `bg1` | 9 | 46,992 |
| `bg2` | 0 | unchanged |
| `bg3` | 0 | unchanged |

`bg2`/`bg3` drop nothing because the grouped extractor already filtered
before writing `groups.json`. `tell` and `bg1` differ by one only because of
population, not behaviour: `bg1`'s titles are a strict subset of `tell`'s, and
`bg1`'s 9 are a subset of `tell`'s 10. The extra one is `BoA`, which only
`tell` has (D8).

## 6. Gates

Ordered; a later gate is not worth running if an earlier one failed.

**Gate 1 — the mixture is what D2 and D4 say it is.** Before booking GPU time,
assert from the built example lists, not from the flags: each source's example
count matches D4's table exactly; each source's val range maps to vector rows
inside that source's own global offset range; and the `bg*` pooled mean is
bit-identical to the one `bg_think_many` used. Check the realised `;`-filter
drops against D9's table (10 / 9 / 0 / 0) -- `bg1`'s 9 is expected, not a
fault. The separator-count check
`bg_think_many` used to verify its slices **cannot work here** -- `tell` and
`bg1` both compose to zero separators -- so use the row-offset check instead.

**Gate 2 — validation loss, per source, as a diagnostic (D7).** Report all four
slices. This gate asks only whether the run trained: every slice finite, every
curve converging, no slice stuck or diverging. It is **not** a comparison
against `bg_think_many` or against the published checkpoint -- see D7 for why
those numbers do not measure the same task. Do not pass or fail the run on them.

**Gate 3 — set-level retrieval is not catastrophic.** As `bg_think_many` §4:
score `best.pt` at `--max-new-tokens 110`, temperature 0.7, seed 42, against the
full 49,637-topic index, reporting per source. The untrained floor is **0.00068**
aggregate recall@1 (not 0.0013 -- see `bg_think_many`'s notes, which correct that
figure). Report the `segments` histogram beside every score.

**Fail only if no source clears 3x the floor (0.00204).** Per the user
(2026-09-09): OOD generalisation is unpredictable, so a merely unimpressive
in-distribution retrieval score is not grounds to withhold the OOD arms. This
gate exists to catch a broken adapter, nothing more. `tell` has no comparable
prior at this decoding length and does not need one.

The threshold is per source because 0.00068 is itself a single-source mean over
positions; no pooled figure was ever implied. Running this gate takes **two
invocations** of `evaluate_retrieval.py`, which cannot express D3's two centring
groups in one -- step 2 §5 gives both commands and the reason.

## 7. The evaluations

Not gates. A negative result is the finding. Run the same three OOD arms
`bg_think_many` ran, matched to its settings so the numbers compose:

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, matched to
   `outputs/taboo_bg_think_many`'s sidecar.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, book and chair only
   (the user cut the predecessor's four words to two, to hold cost down),
   matched to `outputs/taboo_assistant_bg_think_many`.
3. **Bridge entity (TwoHopFact)** -- raw uninjected activations, no mean
   subtraction.

**Both OOD tasks carry the result; neither is subordinate.** Per the user
(2026-09-09), correcting an earlier draft of this section that called the bridge
entity "the one this plan is really about". The two tasks answer different
halves of the standing question (§1) -- taboo asks whether the adapter recovers
a concept the model is *actively hiding*, the bridge entity asks whether it
recovers one the model merely holds latently -- and a result on one does not
substitute for the other. Weight them equally in the report.

**Arms 1 and 2 are one task, split for historical reasons.** The user-prompt and
assistant-token harnesses exist as separate scripts because they were built at
different times, not because they measure different things. Report each against
its own predecessor (their word lists differ -- see step 2 §6 -- so they cannot
simply be concatenated), but the analysis must also read them **together** as
the taboo result, rather than presenting two unrelated arms. Where a conclusion
holds in one and not the other, say which and treat that as the finding.

Context for the bridge-entity arm, not a claim about its priority: it is the
paper's headline OOD result and the most distant from training conditions, and
it is where `bg_think_many` regressed hardest -- 70/100 against `baseline`'s
89/100 and `bg_think`'s 88/100, with a 3.5x lower generation hit rate and
non-overlapping intervals. `baseline` scores 89/100 having been trained on
exactly the `tell` data this plan adds back.

*That last sentence is the plan author's framing, not a mechanism claimed by the
user (§1). Mark it as such if you carry it into a report.*

## 8. What this run is for

**This is not an isolation experiment.** Per the user (2026-09-09), correcting an
earlier draft of this section that framed it as one. The goal is to **get an
adapter that performs better than `baseline`** on the OOD tasks (§7). It is
plausible `tell_and_think` beats both `baseline` and `bg_think_many`, and that
is the outcome the run is chasing.

So the comparisons D4 and D5 buy -- same background data, same architecture, one
source added -- are a convenience, not the deliverable. Do not report a
attribution claim as though the run were designed to support one, and do not
weaken a positive OOD result by hedging it against a confound the run was never
trying to control. **`baseline` is the primary comparison arm; `bg_think_many` is
the secondary one.** `bg_think` is not an arm (its figures appear in §7 only as
context for how much ground was lost).

**Acknowledged and accepted, not defects:**

- **The step count rises** with the budget, 5,902 -> 8,853, and
  `_lr_at_step` sets the cosine horizon from it (`t_max = total_steps -
  warmup_steps`). So `tell_and_think` also sees a stretched LR schedule, not
  only more data. Confirmed correct by the user; state it in the report and move
  on.
- **The `tell` centring rule differs** from `bg_think_many`'s (D3), so the `tell`
  slice is not on the predecessor's footing. The `bg*` group's mean is unchanged.
- **`bg1` draws 251,797 examples from 469,920 vectors** (0.54 each), so ~46% of
  its distinct activations are never seen. (46,992 topics x 10 positions, after
  D9's filter -- not 47,001 x 10, which is the unfiltered population. The 0.54
  is unchanged either way.) This is the *data-diversity* inefficiency that
  `bg_think_many`'s `--rounds` choice was about, and it is
  inherited unchanged, frozen by D4. `tell`, by contrast, is used whole: every
  one of its vectors, paired with every one of its labels, exactly once (D2).
  That is ~16.9 labels per vector -- re-use of the *vector*, not a diversity
  deficit, since no label it holds goes unseen and none is seen twice.

**Still open, and the user's to call:** `bg_think_many`'s rank-64-capacity vs
multi-topic-data question is untouched here. The control that answers it is a
rank-64 projection trained on the single-topic vectors at the same budget. That
run is a separate question from this one's goal.
