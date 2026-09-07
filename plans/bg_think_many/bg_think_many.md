# bg_think_many: a SelfIE adapter trained on several background topics at once

Status: plan, nothing implemented. Parent plan; the execution steps are the
`step*.md` files beside it.

The adapter this plan produces is named **`bg_think_many`**, and lands at
`outputs/adapters/bg_think_many/`. The two things it is measured against are
**`bg_think`** (`outputs/adapters/bg_think/`, the single-topic predecessor) and
**`baseline`** (`outputs/adapters/wikipedia-scalar-affine.safetensors`, the
published upstream checkpoint).

## 1. The question

Quoted verbatim from the user's request (2026-09-05). Do not paraphrase it, and
do not restate it in your own words in any report:

> Preliminary results have convinced me to pivot somewhat from previous plans
> (plans/archive/pangram_extraction_adapter.md and its execution steps). The
> original idea was to train an adapter on prompts like "think about X while
> writing Y", with the adapter's goal being to reproduce X. However, the
> resulting adapter (outputs/adapters/bg_think) doesn't perform better than the
> baseline adapter (outputs/adapters/wikipedia-scalar-affine.safetensors) on the
> OOD tasks I care about.
>
> The new thing I want to try is having the adapter training prompt involve
> thinking about multiple topics in the background. In other words, training on
> prompts like "think about X1, X2, and X3 while writing Y".

and, on the mixture:

> Off a hunch, I think it's more important that the model learn to verbalize
> multiple topics at once, so let's say (without much justification) that the
> ratio of dataset size for 1,2,3 topics being thought about, should also be
> 1:2:3.

Standing project question (README.md, Q1): "Will a SelfIE adapter correctly
uncover something the model is actively hiding?"

**What this plan does not claim.** No mechanism for *why* several background
topics should help is stated by the user, and none is assumed here. If you write
one into a report, mark it as your own inference and say so.

## 2. Vocabulary

Extends the archived plan's table. "Topic" and "label" keep their old meanings.

| term | meaning |
|---|---|
| **topic** | one Wikipedia entry, with its 6-20 labels. 49,637 in the corpus |
| **group** | an ordered tuple of k topics (k = 1, 2 or 3) named by one extraction prompt |
| **vector** | one layer-19 activation; one response-token position of one group |
| **label tuple** | k labels, one per topic of a group, joined `"; "` in some order |
| **example** | one (vector, composed-label) pair; one training item |
| **round** | one random disjoint partition of a split's topics into groups of size k |

## 3. What already exists, and what must be built

Read `plans/archive/pangram_extraction_adapter/` for the history. Do not repeat
its arm-B/arm-C framing; that experiment is finished.

**Reusable unchanged:** `adapter_training/extract_common.py` (`run_forward`,
`write_extraction_outputs`), `loss.py`, `checkpoints.py`, `projection.py`,
`inference.py`, `train_adapter.py`'s optimizer/sampler/resume machinery,
`topic_retrieval_eval.py`.

**Already plumbed, needs no code:** the rank-64 architecture.
`ScalarAffinePlusLowRankProjection` exists and is reachable end to end with
`--projection-type scalar_affine_plus_low_rank --projection-rank 64`, through
the trainer, the checkpoint metadata and `inference.py`. Rank 64 is upstream's
own choice: `resources/selfie-adapters/training/configs/scalar_plus_low_rank_8b.yaml`
says `low_rank_rank: 64  # Rank-64 achieves best validation loss in paper`. That
file's other hyperparameters (lr 0.01, `init_scale` 5.0, clip 0.5, weight decay
0.01, warmup 10, cosine) are the same ones `bg_think` used, so **no
hyperparameter changes are needed for the architecture switch**.

**To build**, one step file each:

| step | file | GPU? | what it delivers |
|---|---|---|---|
| 1 | `step1_label_complexity.md` | no | the complexity-bucketing rule, and evidence it is comparable across topics |
| 2 | `step2_grouped_extraction.md` | yes, small | the multi-topic prompt and extractor; `groups.json` |
| 3 | `step3_example_builder.md` | no | grouped example construction and the 1:2:3 sampler |
| 4 | `step4_trainer_logging.md` | no | `--log-every`, decoupled from `--validate-every` |
| 5 | `step5_set_retrieval_eval.md` | no | set-level recall |
| 6 | `step6_run_and_report.md` | yes, large | the training run, the evaluations, the report |

Steps 1, 3, 4 and 5 need no GPU and no network. Steps 1, 4 and 5 are independent
of each other and of everything else, so they can be handed out in parallel.
Step 3 depends on step 1's decision and on step 2's output format (not its
vectors). Step 6 depends on all of them.

**One agent at a time on GPU work** (project rule): steps 2 and 6 must not run
concurrently with each other or with another agent's GPU job.

## 4. Design decisions

Every decision below is settled. Do not relitigate one without saying which
decision number you are reopening and why.

**D1 — The prompt keeps its existing wording for k=1.** The k=1 form must be
byte-identical to `prompts.PANGRAM_PROMPT_TEMPLATE`, so the existing
`outputs/bg_think_l19` extraction stays valid for the k=1 slice and the k=1
numbers stay comparable to `bg_think`. The plural forms add topics with an
Oxford comma, as the user wrote them:

    k=1: Think about the topic "A" while writing the sentence.
    k=2: Think about the topics "A" and "B" while writing the sentence.
    k=3: Think about the topics "A", "B", and "C" while writing the sentence.

**D2 — Labels are joined with `"; "` in a random order.** The order of the
label tuple is drawn independently of the topic order in the prompt. Over the
dataset every order therefore appears; no group enumerates its orders
exhaustively (see D4).

**D3 — Groups never cross the train/val split.** Topics carry a topic-level
split. A group mixing a train topic with a val topic would leak val labels into
training. Groups are formed inside each split, separately.

**D4 — Examples are sampled, never enumerated.** A 3-topic group with ~17
labels per topic and 6 orders spans ~29,000 composed labels, per position. The
example set is drawn instead: sample a (group, position), then a complexity
bucket, then one label per topic from that bucket, then a random permutation.
This hits the target count exactly, needs no integer juggling over groups, and
is reproducible from a seed. See `step3`.

**D5 — Complexity is matched within a label tuple, by a rule step 1 chooses.**
The upstream dataset has **no complexity metadata**: it generated "five varied
descriptions at different levels of detail" per run, then merged several runs
and deduplicated with `dict.fromkeys`, leaving 6-20 labels per topic in no
recoverable order. A length proxy is therefore necessary. Step 1 decides the
exact rule and, in particular, checks the thing a naive within-topic tercile
does not guarantee: that bucket *i* means the same level of detail for one topic
as for another.

**D6 — The dataset ratio is 1:2:3 in examples**, per the user, where an example
is a (vector, composed-label) pair.

**D7 — The budget is 1,510,782 examples seen**, twice `bg_think`'s 755,391, and
the training pool is the same size, so the run is a single pass. Upstream's own
low-rank config spends the same order of examples (`num_epochs: 2` over its
839,602 pairs); the two agree, and nothing should be read into that beyond
agreement. Split by D6:

| k | examples | topics named per prompt |
|---|---|---|
| 1 | 251,797 | 1 |
| 2 | 503,594 | 2 |
| 3 | 755,391 | 3 |

At batch 256 that is **5,902 optimizer steps**.

**D8 — Rounds are per k: two for k=2, five for k=3.** One round is a disjoint
partition, so each topic appears once; further rounds put each topic in that
many different groups, which covers more topic *combinations* without repeating
any (topic, position) pair more than that. Extraction is cheap enough that this
costs little (§5). k=1 needs no rounds — it is the existing extraction. The
count is not a constant: it follows the rule below, and moves whenever D6's
mixture or D7's budget moves.

*How many rounds a budget wants.* `rounds` sets how many distinct activations
exist to sample examples from; D7 sets how many examples are drawn. One round
yields `floor(N_train/k) + floor(N_val/k)` groups and 10 vectors each, so for
one example per vector:

    rounds ≈ examples_k × k / (N × positions)

So `rounds` scales with `k × examples_k`: doubling k halves the groups one
round yields, and the budget for that k does the rest. `examples_k` is
whatever the mixture says — do not fold D6's ratio into this rule, since that
ratio is a hunch specific to this attempt (§1).

Under D6 and D7 as they currently stand, with N = 46,992 and positions = 10,
the rule gives 0.54, 2.14 and 4.82 for k = 1, 2, 3. At `rounds=2` the k=3
vectors would be re-used ~2.4 times each while barely half the k=1 vectors are
ever drawn. Reuse is not wrong (each draw gets a different composed label, D4)
but it would mean k=3 contributes the most examples off the fewest distinct
activations. **So k=3 takes `rounds=5`**, for ~0.15 extra A100-hours and ~3.8
extra GB; k=2 takes the 2 its own figure rounds to. Re-derive both from the
rule, and confirm with the user, if D6 or D7 changes.

**D9 — The architecture is `scalar_affine_plus_low_rank`, rank 64**, per the
user, with upstream's own hyperparameters (§3).

**D10 — Generation length rises to 110 tokens** for the retrieval eval
(`--max-new-tokens`, default 30). Three labels average ~50 target tokens; 110
leaves room without truncating a long triple.

**D11 — The retrieval metric is set-level recall**, defined in
`step5_set_retrieval_eval.md`. Confirmed with the user (2026-09-05): split the
generation on `;`, take for each true topic its **best rank across all
segments**, and report the fraction of true topics recovered at N. Extra
segments are not penalised; that is a known precision hole, accepted for now,
and made visible by logging the segment-count distribution beside every score.

**D12 — Report recall broken down by k, never pooled across k.** Inside a fixed
k every query has the same number of true topics, so the average is
unambiguous; pooling would silently weight k=3 most, because of D6.

**D13 — One centring reference across all three k.** Confirmed with the user
(2026-09-06). Each extraction directory writes its own per-position means, but
the training pool centres every k against the *pooled* mean of all three
(`dataset.pooled_position_means`), not against each directory's own.

Rationale, in the user's terms: plausibly a direction in the activations
represents how many topics were named, and centring per directory would delete
it -- inside the k=3 population that component is constant, so a k=3 mean
removes it exactly. Pooling keeps it as each population's offset from the
common reference, so the adapter can contrast the three.

The pooled mean is exact and costs no re-extraction: it re-weights the three
stored `position_means.pt` files by how many records reached each position.
**Every consumer must use it, not just the trainer** -- an evaluation that
centres against one directory's own mean is scoring the adapter in a condition
it never trained in (`step6_run_and_report.md` §0(b)).
Two consequences to state in any report: the k=1 slice is no longer centred
the way `bg_think` was, so Gate 2's k=1 comparison to 1.4844 now carries a
constant per-position offset that `bg_think` did not see; and the between-k
component may be small next to the within-k variance, so preserving it is
cheap insurance, not a predicted effect.

## 5. Cost

Extraction, from the archived plan's measured ~0.16 A100-hours for 49,637
single-topic groups (two forced variants per group):

| population | groups | A100-hours |
|---|---|---|
| k=1 | reuse `outputs/bg_think_l19` | 0 |
| k=2, 2 rounds, both splits | ~47,000 | ~0.15 |
| k=3, 5 rounds, both splits | ~78,300 | ~0.25 |

Training: `bg_think` cost ~1.76 A100-hours for 755,391 examples at ~42 target
tokens each. The 1:2:3 mixture averages ~65 tokens, and there are 2x as many
examples, so expect **~5-6 A100-hours**, not the ~9-11 quoted in the initial
assessment before the sequence lengths were worked out.

Disk: ~9.5 GB of new bf16 vectors (3.6 GB at k=2, 5.9 GB at k=3), on top of the
existing 3.8 GB. 322 GB free at the time of writing, so this is a note, not a
constraint. **Host RAM is the constraint instead**: see `step3_example_builder.md`
§3, which the k=3 round count moves.

## 6. Gates

**Gate 1 (step 2).** The compliance keep rate for k=3 must not collapse. The
single-topic run kept 94.7% (47,001/49,637), and every kept topic matched the
with-stop variant (`variant_counts` shows the no-stop variant matched nothing).
Probe 500 groups at k=3 **before** the full extraction. If the keep rate falls
below ~80%, stop and report: a filter that rejects one group in five is
selecting on something, and which topics it drops matters more than the
throughput.

**Gate 2 (step 6).** Held-out validation loss on the same contrastive, grouped,
in-distribution vectors the adapter trained on. This is the correctness gate for
the trained checkpoint — not `interpret.py`, and not the OOD numbers. Compare
against `bg_think`'s 1.4844 **only within the k=1 slice**; the k=2 and k=3
slices predict longer targets and have no comparable prior number.

**Gate 3 (step 6).** Set-level recall on grouped val vectors must beat the
untrained floor by a wide margin, as it did for `bg_think` (0.404 against
0.001). Failing this means something is wrong with the run, not that the
hypothesis is false.

The OOD evaluations (taboo, bridge entity) are the actual object of the
experiment and are **not** gates. They are allowed to come out negative; that is
the finding.

## 7. A confound this plan does not resolve

`bg_think` was `scalar_affine`; `bg_think_many` is `scalar_affine_plus_low_rank`
rank 64 (D9). So a difference between them mixes two causes: the multi-topic
data, and ~500k extra parameters. If the OOD numbers improve, this plan cannot
say which caused it.

The clean disambiguation is one extra run: rank-64 low-rank on the **existing
single-topic** `outputs/bg_think_l19` vectors, at the same 1,510,782-example
budget. That isolates capacity, and it needs no new extraction. It costs roughly
3-4 A100-hours.

**This is not scheduled.** Raise it with the user when step 6's numbers are in;
it is only worth buying if `bg_think_many` actually moves the OOD result.

## 8. Open TODO: 10 topics have a label containing `;`

Found during step 1 review, not yet actioned: 32 of 839,602 labels (across 10
topics) contain a literal `;` --
`BoA`, `Clarissa; or, The History of a Young Lady`, `Frankenstein`,
`Gustave Caillebotte`, `Mamoru Miyano`, `Mary Shelley`,
`Paris Street; Rainy Day`, `Semicolon`, `The Second Coming (poem)`, `Walden`
-- mostly title punctuation (`"Frankenstein; or, The Modern Prometheus"`,
`"Steins;Gate"`) or a quoted line (`"Things fall apart; the centre cannot
hold"`).

This breaks the assumption behind **D2** (labels joined with `"; "`, so a
composed tuple containing one of these labels cannot be unambiguously split
back into its parts) and **D11** (recall is scored by splitting generated
text on `;`; a correct reproduction of one of these labels would be spuriously
split into extra segments).

**Decided workaround (2026-09-06): filter these 10 topics out** rather than
change the separator. This is simpler than escaping or picking a new
delimiter, at the cost of ~0.02% of the corpus (10 / 49,637 topics).

**Done in step 2** (`extract_grouped_vectors.drop_semicolon_topics`).
`label_buckets` (step 1) is a pure per-topic function with no view of the
topic pool, so it is not where this filter belongs; step 2 is the first step
to decide which topics exist as groupable units. 9 of the 10 survive into
`outputs/bg_think_l19` and are dropped there; `BoA` had already failed the
single-topic compliance filter. Step 3's example builder should still assert
no surviving label contains `;` as a cheap regression check.
