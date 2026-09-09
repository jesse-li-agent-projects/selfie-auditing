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
| 1 | `step1_source_keyed_mixture.md` | no | source-name keying, per-source centring groups, per-source val slices |
| 2 | `step2_run_and_report.md` | yes, large | the training run, the evaluations, the report |

Step 1 needs no GPU and no network. Step 2 depends on it.

**One agent at a time on GPU work** (project rule).

## 5. Design decisions

Every decision below is settled, four of them by the user on 2026-09-09. Do not
relitigate one without saying which decision number you are reopening and why.

**D1 — The adapter is named `tell_and_think`**, per the user, after the two
extraction prompts it merges: `Tell me about X.` and `Think about X while
writing Y.`

**D2 — Four sources, mixed 3:1:2:3.** Per the user: `tell` takes the same share
as k=3, the largest single share, so the original paper's data is one third of
the mixture.

| source | directory | k | share |
|---|---|---|---|
| `tell` | `outputs/baseline_l19` | 1 | 3 |
| `bg1` | `outputs/bg_think_l19` | 1 | 1 |
| `bg2` | `outputs/bg_think_many_l19_k2` | 2 | 2 |
| `bg3` | `outputs/bg_think_many_l19_k3` | 3 | 3 |

Note what this costs: `tell` has **one vector per topic**, against ~10 positions
per group for the pangram sources, so its 755,391 examples are drawn from only
44,673 distinct train vectors -- ~17 draws each. Each draw gets a different
label (there are 6-20 per topic), so this is not literal repetition, but it is
far heavier vector re-use than any other source. Say so in the report; do not
discover it there.

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
its validation slice comparable to the published checkpoint (D7).

So: `tell` is its own centring group; `bg1`, `bg2`, `bg3` share one, computed
exactly as `bg_think_many` computed it. **The `bg*` group's pooled mean is
therefore bit-identical to `bg_think_many`'s**, which is what keeps D5's
comparison clean. Verify that rather than assume it.

**D4 — The budget grows; the `bg*` counts are frozen at `bg_think_many`'s.**
Per the user. Each `bg*` source keeps the exact example count it had, and `tell`
is added on top. This makes `tell_and_think` a clean single-variable ablation
against `bg_think_many`: same architecture, same hyperparameters, same
background data, one source added.

| source | train examples | val examples |
|---|---|---|
| `tell` | 755,391 | 150,000 |
| `bg1` | 251,797 | 50,000 |
| `bg2` | 503,594 | 100,000 |
| `bg3` | 755,391 | 150,000 |
| **total** | **2,266,173** | **450,000** |

Both totals divide by the 3:1:2:3 ratio exactly, with no remainder, so
`_split_by_ratio` has nothing to round. At batch 256 that is **8,853 optimizer
steps**, against `bg_think_many`'s 5,902. The `bg*` val slices keep
`bg_think_many`'s own 50k/100k/150k sizes, so each is directly comparable
number-to-number.

**D5 — The architecture does not move.** `scalar_affine_plus_low_rank`, rank 64,
`low_rank_init_factor` 0.01, and every hyperparameter `bg_think_many` used (lr
0.01, `init_scale` 5.0, clip 0.5, weight decay 0.01, warmup 10, cosine). This is
not a default -- it is the point. `bg_think_many` changed **both** the data and
the architecture relative to `bg_think`, which is why its notes say no OOD result
there can be attributed to the data alone. This run changes the data only, so
`tell_and_think` vs `bg_think_many` is interpretable in a way that comparison was
not. Do not tune anything.

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

**D7 — Validation loss is reported per source, never pooled.** Extends
`bg_think_many`'s D12 from k to source. Pooling would weight `tell` and `bg3`
most, by D2. Two slices have priors:

| slice | prior | where from |
|---|---|---|
| `bg1` | 1.3294 | `bg_think_many`'s own k=1 slice, same centring, same size |
| `tell` | **1.3662** | the published checkpoint's recorded `best_val_loss` |

The `bg1` prior is a like-for-like comparison and the gate should treat it as
one. **The `tell` prior is not like-for-like and must not be reported as
though it were**: 1.3662 was a plain `scalar_affine` projection (0 low-rank
parameters) trained for 2,951 steps, read from the safetensors metadata of
`outputs/adapters/wikipedia-scalar-affine.safetensors`. A rank-64 projection
should beat it. Use it as a floor -- if the `tell` slice lands *worse* than
1.3662, something is wrong -- not as a target.

**D8 — `tell` keeps its full 49,637-topic population.** `bg_think_l19` has
47,001 topics, every one of which is also in `baseline_l19`; `baseline_l19` has
2,636 more, dropped by the pangram fidelity filter. Keeping all of them means the
`tell` slice is the upstream population, which is what D7's 1.3662 comparison
needs. The 5.3% asymmetry is the cost, and it means a few thousand topics are
seen only through `tell`. Note it in the report.

**D9 — The `;`-label filter applies to `tell` too.** `drop_semicolon_topics`
exists because a composed label joins topics with `"; "`, and a label containing
a semicolon would teach the wrong segmentation. `tell`'s labels are never
composed, so the filter is not strictly required there -- but it costs **10
topics out of 49,637**, and a `tell` label with a semicolon would still teach the
adapter to emit a separator where no second topic exists. Apply it, for one
consistent rule across sources. Record the exact dropped count.

## 6. Gates

Ordered; a later gate is not worth running if an earlier one failed.

**Gate 1 — the mixture is what D2 and D4 say it is.** Before booking GPU time,
assert from the built example lists, not from the flags: each source's example
count matches D4's table exactly; each source's val range maps to vector rows
inside that source's own global offset range; and the `bg*` pooled mean is
bit-identical to the one `bg_think_many` used. The separator-count check
`bg_think_many` used to verify its slices **cannot work here** -- `tell` and
`bg1` both compose to zero separators -- so use the row-offset check instead.

**Gate 2 — validation loss, per source (D7).** `bg1` should land near 1.3294;
`tell` should land at or below 1.3662 with the caveat in D7. `bg2`/`bg3` should
land near 1.6505/1.8797. A slice far *better* than its prior is a bug signal, not
a win -- check Gate 1's assertions again before believing it.

**Gate 3 — set-level retrieval clears the floor.** As `bg_think_many` §4: score
`best.pt` at `--max-new-tokens 110`, temperature 0.7, seed 42, against the full
49,637-topic index, reporting per source. The untrained floor is **0.00068**
aggregate recall@1 (not 0.0013 -- see `bg_think_many`'s notes, which correct
that figure). Report the `segments` histogram beside every score.

## 7. The evaluations

Not gates. A negative result is the finding. Run the same three OOD arms
`bg_think_many` ran, matched to its settings so the numbers compose:

1. **Taboo, user-prompt tokens** -- `run_pipeline.py`, matched to
   `outputs/taboo_bg_think_many`'s sidecar.
2. **Taboo, assistant tokens** -- `selfie_on_assistant.py`, four words (book,
   chair, blue, salt), matched to `outputs/taboo_assistant_bg_think_many`.
3. **Bridge entity (TwoHopFact)** -- raw uninjected activations, no mean
   subtraction.

**Arm 3 is the one this plan is really about**, and the report should say so
plainly. It is the paper's headline OOD result and the most distant from training
conditions, and it is where `bg_think_many` regressed hardest: 70/100 against
`baseline`'s 89/100 and `bg_think`'s 88/100, with a 3.5x lower generation hit
rate and non-overlapping intervals. `baseline` scores 89/100 having been trained
on exactly the `tell` data this plan adds back. Whether adding it recovers that
ground is the sharpest question the run answers.

*That last sentence is the plan author's framing of why the arm matters, not a
mechanism claimed by the user (§1). Mark it as such if you carry it into a
report.*

## 8. What this plan does and does not isolate

**Isolates:** the effect of adding the original paper's extraction data, holding
architecture, hyperparameters, background data and background example counts
fixed (D4, D5). This is a cleaner comparison than `bg_think_many` vs `bg_think`
was.

**Does not isolate:** whether any change comes from the *data* or from the
*centring rule*, since D3 gives `tell` its own mean while `bg_think_many` pooled
everything. The `bg*` group's mean is unchanged (D3), so the `bg*` slices are
still comparable; the caveat applies to `tell` only.

**Still open from `bg_think_many`:** its own confound -- rank-64 capacity vs
multi-topic data -- is *not* resolved by this plan. The control that resolves it
is a rank-64 projection trained on the single-topic vectors at the same budget.
That run is still recommended and still the user's to call.
