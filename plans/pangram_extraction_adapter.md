# Pangram-prompt SelfIE adapter (layer 19, Wikipedia topics)

Status: plan, not implemented; all design decisions taken (§7).

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

   The paper's own hyperparameters (Table 5) are the ones to match, and they differ from
   the YAML configs shipped in the repo, which are for the SAE runs: **AdamW, lr 0.01,
   batch size 256, cosine decay, 10 warmup steps, grad clip 0.5, initial scale 5.0, seed
   42, and one epoch** — "epoch count varied by dataset size: 1 epoch for large datasets
   (Wikipedia, ~840k descriptions) and up to 5 epochs for smaller datasets". The 8B runs
   were done on a **single A100**; the whole paper cost 180-220 GPU-hours across training
   and evaluation.

   Note the paper's ~840k figure matches the 839,602 measured above, which confirms the
   dataset above is the one it trained on.

The upstream design deliberately **transfers across extraction prompts**. The adapter maps
an activation — from whatever prompt produced it — into a fixed *interpretation* prompt
("What is the meaning of ...?"). §3.4 of the paper takes the adapter trained on these
Wikipedia topic vectors and applies it unchanged to TwoHopFact prompts, reading "at every
layer and token", detecting the unverbalised bridge entity in 91.0% of 500 prompts against
56.4% untrained. So reading at arbitrary token positions of an unrelated prompt is the
normal, intended use, not a deviation.

Stages 2 and 3 talk through files on disk. **Extract once, train many times is therefore
the operating principle of this plan, not an optimisation**: extraction is ~4% of the cost
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
for the baseline. The paper's single Wikipedia epoch is 839,602 examples, **3,280 steps at
batch 256**; one epoch of the pangram pool would be 32,800 steps.

**(c) Epoch count is the wrong budget, and this is not "training on a fraction of the
data".** Detailed below, because this is the part that was unclear.

### 4.1 Why budget by examples seen rather than by epochs

The concern is fair: holding the budget fixed does mean arm B makes one pass over ~10% of
its pool. But that is not the same as discarding 90% of the *data*, because the pool is a
cross product.

With the pangram prompt each topic contributes 10 vectors × ~17 labels = ~169 examples.
The 49,637 topics are the same 49,637 topics either way. Spending the paper's budget of
839,602 examples on this pool means the shuffled sampler draws ~17 examples per topic —
still every topic, still every position on average 1-2 times, still a spread of labels.
What falls is not coverage but **the number of times each individual (vector, label) pair
is revisited**. Nothing is systematically excluded.

The paper is already doing this, incidentally. Table 5 sets "1 epoch for large datasets
(Wikipedia, ~840k descriptions) and up to 5 epochs for smaller datasets" — i.e. the authors
held roughly constant *volume* and let the epoch count fall out of it, rather than fixing
epochs and letting cost scale with pool size. Holding examples seen constant is the same
policy applied to a pool that got 10× bigger.

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

Budget: **839,602 examples seen**, the paper's single Wikipedia epoch — 3,280 steps at its
batch size of 256. Arm A is then a genuine replication (1 epoch, exactly as published) and
every other arm is a like-for-like comparison at equal compute. Arm B spends it as 0.1
epochs of its larger pool; arm C, like A, as one full epoch.

### 4.2 Sizing on whatever GPU is chosen

Stated as work, not wall clock, so it survives a change of card. For an 8B model at bf16, a
forward pass is ~2N = **16 GFLOP/token**; a training step over a *frozen* model needs
activation gradients but not weight gradients, so ~2.5× forward ≈ **40 GFLOP/token**.

| | tokens | work |
|---|---|---|
| Extraction, 49,637 topics, pangram prompt (81 tok/topic) | 4.0M fwd | ~64 PFLOP |
| Extraction, 49,637 topics, baseline prompt (~41 tok/topic) | 2.0M fwd | ~33 PFLOP |
| One training run at the 839,602-example budget (~46 tok/example) | 39M train | ~1.5 EFLOP |

Divide by (peak bf16 TFLOP/s × ~0.35 achieved). That gives **~4 hours on one A100**, which
is a useful check on the model: the paper ran exactly this on a single A100 and reported
180-220 GPU-hours across every training and evaluation run in the whole paper, so a ~4-hour
figure for one run is the right order. On a 24 GB Ampere-class card the same run is ~17
GPU-hours; divided across a 4-GPU rig (§4.4), ~4-5 hours wall clock.

Extraction is ~4% of one training run either way — hence extract-once.

**Memory:** 8B bf16 weights are 16 GB, so 24 GB is the practical floor. At the paper's
batch of 256 (~11.8k tokens/step), uncheckpointed activations are roughly 28 GB, so on
anything below ~48 GB **gradient checkpointing is not a tuning choice but a requirement**
— which is why the reference configs enable it. An earlier draft of this plan suggested
turning it off; that was wrong at this batch size. The genuine choices are checkpointing
on at full batch, versus off with gradient accumulation over smaller micro-batches, and
which wins is card-dependent and cheap to measure. Step 0 measures both and the winner is
used.

### 4.3 Using a multi-GPU rig

Yes, and this workload is close to the ideal case for it. Use **DistributedDataParallel**:
put a full copy of the frozen 8B on each GPU, shard the global batch across them, and
all-reduce gradients. Because the only trainable tensors are the projection's — 4097
parameters for `scalar_affine`, 528,385 at rank 64 — the per-step gradient sync is a few
megabytes at most and effectively free. Scaling should be near-linear in GPU count.

Three practical points:

- **Do not use `device_map="auto"` for this**, which is what the reference configs do. That
  shards one model across cards (pipeline-style) and is the right tool only when the model
  does not fit on one GPU. At 16 GB on a 24 GB card it does fit, so `device_map="auto"`
  would only add transfer latency and pipeline bubbles for no throughput gain. Each rank
  loads its own complete copy instead.
- **Every GPU still needs its own 16 GB of weights**, so the 24 GB floor is per-card, not
  aggregate. A 4×24 GB rig does not let you pretend you have 96 GB.
- **Keep the *global* batch at the paper's 256** so the run stays comparable; per-GPU batch
  is then 256/N. At N=4 that is 64 per GPU (~2.9k tokens), still a healthy matmul.

There is also a simpler option that needs no distributed code at all: the run matrix is
five independent runs (§5.5), so on an N-GPU rig you can run N configs concurrently, one
per GPU, and reach full utilisation with nothing but a device flag. DDP is what makes a
*single* run faster; one-run-per-GPU is what makes the *matrix* faster, and it is the
better deal here unless a single run's wall clock becomes the bottleneck. The trainer
should support both, but one-run-per-GPU is the default and DDP is the optional path.

### 4.4 Disk

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
the distribution; benchmark examples/second for checkpointing-on at global batch 256
versus checkpointing-off with gradient accumulation, on the chosen card; measure the true
mean label token length to replace the ~46-token estimate. Output is a short findings
note plus the settings the real runs use. Throwaway `.tmp.py`.

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
- **[D3]** Write a small trainer (option b), not a vendored copy of the reference's. It
  must support one-run-per-GPU and, optionally, DDP (§4.3).
- **[D4]** Budget is 839,602 examples seen per run — the paper's single Wikipedia epoch,
  3,280 steps at batch 256 — held equal across all arms.
- **[D5]** Runs on a machine the user nominates at execution time. Nothing in this plan
  assumes a particular card; §4.2 and §4.3 give the sizing needed to choose one. The local
  GPU is an 8 GB laptop card and cannot hold an 8B model.
- **[D6]** No change to the taboo pipeline's prompts is required. The adapter is *designed*
  to transfer across extraction prompts (§3): it maps an activation from whatever prompt
  produced it into the fixed interpretation prompt, which is exactly what the paper's
  TwoHopFact experiment does with this same Wikipedia-trained adapter. So the pangram
  prompt is a hypothesis about better *training* activations, and the trained adapter drops
  into `run_pipeline.py` unchanged.

  For the record, since an earlier draft of this plan got it wrong: `interpret.py`'s
  `SELFIE_TEMPLATE` is the *interpretation* soft prompt ("What is the meaning of ...?"),
  which must stay exactly as trained; the taboo pipeline's *extraction* prompt is
  `config.SECRET_PROMPT` read across `Position.USER_PROMPT_SPAN`. Neither needs to change.

## 8. Still open

- Nothing blocking. Phase 2 — whether the taboo elicitation should *also* use the pangram
  prompt — is a genuine experiment rather than a fix for a defect, and belongs in its own
  plan once phase 1 has a result.
