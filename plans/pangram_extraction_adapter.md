# Pangram-prompt SelfIE adapter (layer 19, Wikipedia topics)

Status: plan, not implemented. Open decisions marked **[D#]**; decisions already taken are
recorded in §7.

## 1. The question

Quoted verbatim from the user's request (2026-08-27), not paraphrased:

> Instead of the extraction prompt `Tell me about <topic>`, use the prompt `Write "The
> quick brown fox jumps over the lazy dog". Think about the topic "<topic>" while writing
> the sentence. Do not write anything else or change the words.` The model should respond
> with "The quick brown fox jumps over the lazy dog"; extract the activations from all of
> the response tokens. As an initial test, let's only train the adapter for layer 19.
> Let's test over the Wikipedia dataset, from the reference code.
>
> Note - testing occasionally shows the model failing to reproduce the phrase "The quick
> brown...", so there should be a topic filter for generations that correctly reproduce
> the phrase.

Motivation given: "I'm hoping this approach will make the adapter better at surfacing
hidden thoughts."

Standing project question this serves (README.md, Q1): "Will a SelfIE adapter correctly
uncover something the model is actively hiding?"

## 2. Vocabulary

Used strictly throughout. The word "row" is deliberately avoided — it was ambiguous
between two of these.

| term | meaning | count in the baseline | count with the pangram prompt |
|---|---|---|---|
| **topic** | one Wikipedia entry | 49,637 | 49,637 |
| **vector** | one extracted layer-19 activation; one line of the `.pt` | 1 per topic | 10 per topic (one per response token) |
| **label** | one natural-language description of a topic | 6-20 per topic (mean 16.9) | unchanged — labels attach to topics, not to vectors |
| **example** | one (vector, label) pair; one training item | 839,602 | 8,396,020 |
| **examples seen** | `steps × batch_size`; the unit of training cost | — | — |

## 3. What the reference actually does

Corrected after review — an earlier draft of this plan described training against the
*topic title*, which was wrong.

The upstream pipeline is three stages, and only the first is affected by this experiment:

1. **Dataset generation** (`data_prep/wikipedia_topics/dataset_generation/`): Claude is
   asked, per Wikipedia title, for one natural conversational prompt *and five varied
   descriptions* of the topic at different levels of detail. Multiple generation runs are
   merged and deduplicated, then filtered by a coherence score ≥ 9.0. The published result
   is [`keenanpepper/fifty-thousand-things`](https://huggingface.co/datasets/keenanpepper/fifty-thousand-things):
   **49,637 topics**, each with `prompt`, `labels`, and a `split` (44,673 train / 4,964
   val). Labels per topic: min 6, max 20, **mean 16.9**. Total 839,602 (topic, label) pairs.

   The prompt is per-topic and hand-generated, e.g. `Tell me about William Wallace.` but
   also `Tell me about bits (binary digits).` — not a mechanical `Tell me about {title}.`.
   Labels are descriptions, not titles: for William Wallace they include
   `"Wallace, victor at the Battle of Stirling Bridge"` and
   `"the Guardian of Scotland executed by Edward I in 1305"`.

2. **Extraction** (`extract_multilayer_vectors.py`): render each topic's own `prompt`
   through the chat template with `add_generation_prompt=True`, one forward pass, keep the
   layer-19 residual stream at the **last** token. One vector per topic. No generation
   happens. (The sibling `extract_wikipedia_vectors.py` is a simplified title-based
   variant and is *not* what the published adapter was trained on.)

3. **Training** (`training/`): a frozen Llama-3.1-8B is fed a fixed ~30-token soft prompt
   with the vector projected into the `<|reserved_special_token_0|>` slot, and the
   projection is trained by cross-entropy to make the model emit a label. `create_datasets`
   flattens each vector against *every one of its labels* — hence 839,602 examples from
   49,637 vectors. Only the projection trains: `scalar_affine` is `scale * x + bias`,
   4097 parameters at d=4096.

Stages 2 and 3 talk through files on disk. **Extract once, train many times is therefore
the operating principle of this plan, not an optimisation**: extraction is ~2% of the cost
of a single training run (§4), so every arm and every adapter architecture reads the same
cached vectors, and no experiment ever re-runs the 8B forward pass over the corpus.

## 4. Cost, and the answer to the training-time question

> is it accurate to say that because I'm testing more generations (testing each token in
> the response prompt, rather than just one word as done in the SelfIE paper), training
> should take longer (multiplied by however many additional generations I'm testing)?

Since the concern is throughput per dollar, the honest unit is **examples seen per unit of
GPU rent**, and the answer has three parts.

**(a) Cost per example does not change.** A training step is a forward+backward of the
frozen 8B over a ~46-token sequence (soft prompt ~30 tokens + label ~16). A vector is a
vector; which token position it was read from is invisible to the trainer. No per-example
multiplier.

**(b) The available pool multiplies by 10, so a fixed *epoch count* costs 10× more.**
Measured with the Llama-3.1 tokenizer, `The quick brown fox jumps over the lazy dog.` is
**10 tokens** (`The`, ` quick`, ` brown`, ` fox`, ` jumps`, ` over`, ` the`, ` lazy`,
` dog`, `.`). So 49,637 topics yield 496,370 vectors and 8.4M examples, against 839,602
for the baseline. The reference config's 2 epochs would go from ~21,000 steps to ~210,000.

**(c) Epoch count is the wrong budget, and this is not "training on a fraction of the
data".** Detailed below, because this is the part that was unclear.

### 4.1 Why budget by examples seen rather than by epochs

The concern is fair: capping steps does mean one pass over ~20% of the pool. But that is
not the same as discarding 80% of the *data*, because the pool is a cross product.

With the pangram prompt each topic contributes 10 vectors × ~17 labels = ~169 examples.
The 49,637 topics are the same 49,637 topics either way. Spending the reference's budget
of ~1.68M examples on this pool means the shuffled sampler draws ~34 examples per topic —
still every topic, still every position roughly 3-4 times each, still a broad spread of
labels. What falls is not coverage but **the number of times each individual (vector,
label) pair is revisited**. Nothing is systematically excluded.

For a 4097-parameter projection, how many times a given pair is revisited is not what
determines convergence. And the 10 positions of one topic are the same prompt, the same
label set, and adjacent tokens of one fixed sentence — heavily correlated, so the effective
new information is far below 10×.

The practical protocol:

- **Fix an examples-seen budget and hold it equal across arms.** This is both the
  cost-controlled comparison and the scientifically correct one — comparing arms at equal
  compute rather than at equal epochs.
- **Watch validation loss against examples seen.** If an arm is still improving at the cap,
  that is a finding, and the cap can be extended deliberately rather than by accident.
- **Scale the budget to adapter capacity, not to dataset size.** `scalar_affine` is 4097
  parameters; `scalar_affine_plus_low_rank` at rank *r* is `4096 + 1 + 2·4096·r` — 135,169
  at r=16 and 528,385 at r=64. The larger ones plausibly need a larger budget; the
  loss-vs-budget curve says whether they did.

Proposed budget: **match the reference's 1.68M examples seen** (2 epochs × 839,602), which
is ~21,000 steps at batch 80. That makes arm A a genuine replication and every other arm a
like-for-like comparison. **[D4]**

### 4.2 Sizing on whatever GPU is chosen

Stated as work, not wall clock, so it survives a change of card. For an 8B model at bf16, a
forward pass is ~2N = **16 GFLOP/token**; a training step over a *frozen* model needs
activation gradients but not weight gradients, so ~2.5× forward ≈ **40 GFLOP/token**.

| | tokens | work |
|---|---|---|
| Extraction, 49,637 topics, pangram prompt (81 tok/topic) | 4.0M fwd | ~64 PFLOP |
| Extraction, 49,637 topics, baseline prompt (~41 tok/topic) | 2.0M fwd | ~33 PFLOP |
| One training run at the 1.68M-example budget (~46 tok/example) | 77M train | ~3.1 EFLOP |

Divide by (peak bf16 TFLOP/s × ~0.35 achieved). Order of magnitude: ~34 h on a 24 GB
Ampere-class card, ~3 h on a current datacentre card. Extraction is ~2% of one training
run either way — hence extract-once.

**Memory:** 8B bf16 weights are 16 GB, so 24 GB is the practical floor and leaves little
headroom. 40 GB+ is comfortable. Additional GPUs shorten wall clock by running arms side by
side; at roughly linear $/GPU-hour that is close to cost-neutral, so treat it as a schedule
decision, not a savings one.

**Gradient checkpointing is left as a measured setting, not a recommendation.** Off saves
the ~25-30% recompute; on frees memory that may buy a larger batch and better utilisation.
At ~46-token sequences the activations for batch 80 are roughly 9 GB uncheckpointed, which
does not fit alongside 16 GB of weights on a 24 GB card but is fine on 40 GB+. Which
setting yields more examples/second is card-dependent and cheap to measure — step 0 does
so, and the winner is used.

### 4.3 Disk

Vectors stored bf16 (what the model emits; the trainer casts to fp32 on load) at
4096 dims = 8 KiB per vector.

| artefact | size |
|---|---|
| Pangram vectors, 496,370 × 4096 bf16 | **4.1 GB** |
| Baseline vectors, 49,637 × 4096 bf16 | 0.41 GB |
| Label/index files, lean format (§5.2) | ~50 MB |
| Per-position mean vectors, 10 × 4096 fp32 | 164 KB |
| Checkpoints, capped at best+last per run, r=64 worst case ~6.3 MB each | <100 MB |
| **Total experiment artefacts** | **~4.7 GB** |
| Llama-3.1-8B bf16 weights (already cached wherever this runs) | 16 GB |

Two notes. Storing vectors as fp32 instead would double the 4.1 GB to 8.1 GB for no
benefit. And writing the labels in the *reference's* JSON format would duplicate every
topic's ~17 label strings once per vector — ~480 MB of JSON for the pangram set, and slow
to parse; §5.2 avoids that.

## 5. Design

### 5.1 Extraction, and the filter

The naive route is: generate greedily, string-compare the output to the pangram, keep the
topics that match, then re-run a forward pass to read activations. That is a decode loop
plus a second forward.

Cheaper and exactly equivalent: **teacher-force** the prompt followed by the pangram tokens
in a single forward pass, and check at every response position `i` whether
`argmax(logits[i-1]) == pangram_token[i]`. If that holds at every position, greedy decoding
from the prompt would have produced exactly that sentence — greedy decoding is
deterministic and follows the same prefix, so agreement at every step *is* the greedy
output. Include the trailing `<|eot_id|>` in the forced sequence so a topic whose
generation would have run on past the sentence is also rejected.

One forward pass therefore yields both the 10 layer-19 activations and the filter verdict.
Extraction goes from ~41 to ~81 prompt tokens per topic — about 2×, not 10× — with no
decode at all.

The canonical target string is the **unquoted** sentence. Because that choice is only safe
if the model actually complies, step 0 runs real greedy generation on a topic sample and
**classifies every failure**, not just counts them: wrapped in quotes / preamble before the
sentence / trailing commentary after / altered wording / refusal / other. If quoted output
exceeds ~5-10% of topics, revisit. The categories matter beyond the threshold: a
preamble-heavy failure mode would mean the response-token positions are not where we think
they are, which is a different problem from a cosmetic quoting habit.

### 5.2 What gets written

Per prompt style, three files:

- `vectors.pt` — `[n_vectors, 4096]` bf16, in topic order, positions contiguous per topic.
- `topics.json` — one entry per surviving topic: its labels (once, not duplicated), its
  split as given by the upstream dataset, and its vector index range.
- `positions.json` — small sidecar mapping vector index → (topic index, position index),
  plus the per-position mean vectors.

This is a lean re-shaping of the reference's `{index, labels, split}` format, kept possible
by writing our own trainer (§6, D3=b). It avoids the ~10× duplication of label text.

**Splits come from the upstream dataset and are by topic.** All 10 positions of a topic
inherit that topic's split. Splitting per vector would leak: a val vector would be a
near-duplicate of a train vector with an identical label set.

### 5.3 Mean-centering

Decided: **per-position contrastive vectors**. For each position *p* in 0..9, subtract the
mean over topics of position *p*'s activations. This is the right analogue of the
reference's `vector - mean`: with 10 positions the residual stream is dominated by *which
word of the pangram this is*, and a single global mean would not cancel that; a
per-position mean does, leaving the topic signal.

Evaluation continues to inject **raw** activations, matching upstream (`interpret.py`
docstring, and the reference's own bridge-entity sweep). This is a deliberate train/eval
mismatch inherited from upstream, on the reasoning that training on contrastive vectors
should generalise better out of distribution. The 10 mean vectors are saved next to the
checkpoint so the choice can be revisited without re-extracting.

### 5.4 Arms

All arms share: layer 19, the same 49,637 topics, the same upstream splits, the same
per-position centering, and the same examples-seen budget. Because the budget is equal,
**every arm costs the same to train** — the cost of an arm is exactly "one more run".

| arm | vectors per topic | examples in pool | what it tests |
|---|---|---|---|
| **A** baseline | 1 (last prompt token, upstream prompt) | 839,602 | replication of upstream |
| **B** pangram-per-position | 10 (one per response token) | 8,396,020 | the proposed method |
| **C** pangram-mean | 1 (mean of the 10 positions) | 839,602 | pooling before the adapter |

**B vs C, concretely.** Both read the same 10 activations; they differ in *where the
pooling happens*.

- In **B** the 10 activations never meet. Each is its own training example, paired with the
  topic's labels. The adapter must map *any single* pangram-position activation to the
  topic on its own. At eval you read one position, or run all 10 and pool the generated
  text.
- In **C** the 10 activations are averaged into one vector before the adapter sees
  anything. The adapter only ever handles pooled vectors, and at eval you must average the
  same 10 positions to feed it.

The comparison is informative either way it lands. If C ≈ B, position-specific detail was
not load-bearing and C is the better deal — same result from a pool 10× smaller, so the
same budget covers 10× more passes over it. If B > C, averaging destroys something and
per-position training earns its pool. If both beat A, the gain came from the prompt rather
than from either pooling scheme.

**Arm A is run once, with `scalar_affine` only.** Its purpose is a replication check that
gives confidence in the upstream numbers, so it does not need to be repeated across adapter
architectures.

### 5.5 Adapter architectures

Both reference projection types, with rank as a **config field, never a literal in code**:

- `scalar_affine` — 4097 parameters.
- `scalar_affine_plus_low_rank` — `4096 + 1 + 2·4096·r`. First runs at r=16 and r=64;
  these are config values, and other ranks must need no code change.

To keep the run count sane, the two questions are separated:

| phase | question | runs |
|---|---|---|
| 1 | does the pangram prompt help? | A, B, C × `scalar_affine` = 3 |
| 2 | does capacity help the winner? | winning arm × {r=16, r=64} = 2 |

Five runs, each at the same examples-seen budget.

### 5.6 Measuring

- **Validation loss vs examples seen**, per arm. Comparable across arms: same label set,
  same soft-prompt template, same held-out topics. The curve, not just the endpoint —
  it is what says whether the budget was adequate.
- **Generation accuracy** on held-out topics, reusing the reference's scoring
  (`evals/generation_scoring/`, and the embedding-retrieval eval which is designed for
  exactly this in-distribution case) rather than inventing a metric.
- **Per-position breakdown for arm B** — train pooled, then evaluate positions 0..9
  separately. **Exploratory only: it does not feed the A/B/C conclusion.** Its value is in
  shaping the next iteration — if late positions (`lazy`, `dog`, `.`) carry most of the
  topic signal, a follow-up could read only those and shrink the pool.

## 6. Implementation steps

**Step 0 — probe.** On the real 8B: greedy-generate for a few hundred topics with the
pangram prompt; classify every non-compliant output into the categories in §5.1 and report
the distribution; benchmark examples/second with gradient checkpointing on and off at
several batch sizes; measure the true mean label token length to replace the ~46-token
estimate. Output is a short findings note plus the settings the real runs use. Throwaway
`.tmp.py`.

**Step 1 — extraction script.** New `adapter_training/extract_topic_vectors.py`: CLI with
`--prompt-style {baseline,pangram}`, `--layer`, `--limit`, reading
`keenanpepper/fifty-thousand-things` and writing the three files of §5.2 plus a filter
report. Light-imports-first per the project CLI convention. Unit-tested against
Llama-3.2-1B for shapes, filter logic, split inheritance, and per-position centering.

**Step 2 — trainer.** Decided (D3=b): a small trainer reusing the already-installed
`selfie_adapters.projection.create_projection_module`, writing the same checkpoint dict the
reference does (`projection_state`, `model_dim`, `checkpoint_format_version`, `config`),
which `selfie_adapters.load_adapter` reads directly — so `interpret.py` keeps working
unchanged. Budget expressed in examples seen; projection type and rank from config.

The risk is drifting from the reference's optimizer/loss details. Mitigation: arm A is the
replication, and its validation loss is checked against upstream before arms B and C are
trusted.

**Step 3 — run.** Extract once per prompt style, then the five runs of §5.5, on the machine
the user nominates at execution time (**[D5]**). Checkpoints and vectors to `outputs/`.

**Step 4 — report.** Loss-vs-budget curves per arm, generation accuracy, the per-position
exploration, the step-0 failure taxonomy, and measured timings.

## 7. Decisions taken

- **[D1]** Canonical target is the **unquoted** sentence; the filter stays. Step 0
  classifies failures into categories; revisit if quoted output exceeds ~5-10%.
- **[D2]** Per-position mean-centering for training; evaluation keeps raw activations, as
  upstream does.
- **[D3]** Write a small trainer (option b), not a vendored copy of the reference's.
- **[D5]** Runs on a machine the user nominates at execution time. Nothing in this plan
  assumes a particular card; §4.2 gives the sizing needed to choose one. The local GPU is
  an 8 GB laptop card and cannot hold an 8B model.

## 8. Still open

- **[D4]** The examples-seen budget. Proposed: 1.68M, matching the reference's 2 epochs
  over its 839,602 examples, so that arm A is a true replication. This is ~34 h/run on a
  24 GB Ampere-class card and ~3 h/run on a current datacentre card — five runs, so the
  card choice matters more than the arm count.
- **[D6]** How this adapter is eventually used in the taboo pipeline, which is phase 2 and
  out of scope here, but which shapes what "success" means.

  Clarifying a confused claim in the earlier draft: `interpret.py`'s `SELFIE_TEMPLATE` is
  the *interpretation* soft prompt ("What is the meaning of ...?"), not an extraction
  prompt, and it must stay exactly as the adapter was trained — it does not change. The
  taboo pipeline's **extraction** prompt is `config.SECRET_PROMPT`
  (`"What is the secret word?"`) plus the arm's system prompt, read across
  `Position.USER_PROMPT_SPAN`.

  So the real question is a mismatch this plan creates: the adapter would be trained on
  activations from *pangram response* tokens but applied to *user-prompt* tokens of a
  different question. Two ways out — leave the taboo extraction prompt alone and accept
  the mismatch, or give the taboo model the pangram prompt too (write the pangram while
  holding the secret word) and read its response tokens, which is the closer analogue of
  "surfacing a hidden thought" and probably the intent. That is a change to
  `config.SECRET_PROMPT` and to the position handling in `extract.py`, and it needs its own
  plan.
