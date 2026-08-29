# Pangram-prompt SelfIE adapter (layer 19, Wikipedia topics)

Status: plan, partly implemented (§9); all design decisions taken (§7).

Bought in two instalments (§4.6). **Phase 0** trains arm B alone at full budget and asks
whether the adapter can recover a topic the model never says out loud — measured by
embedding-based retrieval against an untrained floor on the same vectors (§5.4.1, D11).
That is the headline result, all one-off work included, for **~3.0 A100-hours**.
**Phases 1-2** add arm C, arm A's own replication, and the capacity sweep, bringing the
whole plan to ~11 A100-hours.

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

3. **Training** (`training/`): a frozen Llama-3.1-8B is fed a fixed 26-token soft prompt
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

### 3.1 The published checkpoint settles the details

The adapter this repo already loads — `keenanpepper/selfie-adapters-llama-3.1-8b-instruct`,
`wikipedia-scalar-affine.safetensors` (`config.py`) — stores its **entire training config
and run metadata** in its safetensors header. That is a better source than the paper's
table or the repo's YAML, because it is the actual run. Read out:

| field | value | why it matters |
|---|---|---|
| `num_epochs`, `batch_size`, `learning_rate`, `optimizer_type`, `scheduler_type`, `warmup_steps`, `gradient_clip_norm`, `init_scale`, `seed` | 1, 256, 0.01, adamw, cosine, 10, 0.5, 5.0, 42 | confirms Table 5 exactly |
| `global_step` | **2951** | the real budget — see below |
| `best_val_loss` | **1.3662** | a published number to replicate |
| `normalize_input` | **true** | a config field this plan had not recorded |
| train dataset | `wikipedia-50k-topics-contrastive@l19` | confirms layer 19 and mean-centred (contrastive) training vectors |
| `strip_labels`, `eos_token` | true, `<|eot_id|>` | target construction |

**The real budget is 755,391 examples, not 839,602.** Upstream trained on the dataset's
`train` split only: 44,673 topics carrying 755,391 labels, which is 2,950.75 batches of 256
— i.e. exactly the 2,951 steps the checkpoint records. The remaining 4,964 topics / 84,211
labels are the val split. An earlier version of this plan budgeted the *whole* dataset's
839,602, which overshoots the thing it was trying to replicate by 11%.

That arithmetic also settles a question that would otherwise make `best_val_loss` useless:
the 10% held out is the dataset's own **topic-level** split, not a random 10% of examples,
so there is no train/val topic leakage in it. **1.3662 is therefore a topic-held-out
validation loss on the same 84,211 baseline val examples, and is directly comparable to our
own arm A** — subject only to our trainer matching theirs, which is what §6 step 2 checks.
It is *not* comparable to arms B and C, which read a different extraction prompt and so
measure a different task (§5.4).

### 3.2 Two properties of the upstream design this plan relies on

The upstream design deliberately **transfers across extraction prompts**. The adapter maps
an activation — from whatever prompt produced it — into a fixed *interpretation* prompt
("What is the meaning of ...?"). §3.4 of the paper takes the adapter trained on these
Wikipedia topic vectors and applies it unchanged to TwoHopFact prompts, reading "at every
layer and token", detecting the unverbalised bridge entity in 91.0% of 500 prompts against
56.4% untrained. So reading at arbitrary token positions of an unrelated prompt is the
normal, intended use, not a deviation.

Stages 2 and 3 talk through files on disk. **Extract once, train many times is therefore
the operating principle of this plan, not an optimisation**: extraction is ~9% of the cost
of a single training run (§4), so every arm and every adapter architecture reads the same
cached vectors, and no experiment ever re-runs the 8B forward pass over the corpus.

## 4. Cost, and the answer to the training-time question

> is it accurate to say that because I'm testing more generations (testing each token in
> the response prompt, rather than just one word as done in the SelfIE paper), training
> should take longer (multiplied by however many additional generations I'm testing)?

Since the concern is throughput per dollar, the honest unit is **examples seen per unit of
GPU rent**, and the answer has three parts.

**(a) Cost per example does not change.** A training step is a forward+backward of the
frozen 8B over a 39.3-token sequence (26-token template + 13.3-token target, both measured
in §4.2, against the ~46 an earlier draft estimated). A vector is a
vector; which token position it was read from is invisible to the trainer. No per-example
multiplier.

**(b) The available pool multiplies by 10, so a fixed *epoch count* costs 10× more.**
Measured with the Llama-3.1 tokenizer, `The quick brown fox jumps over the lazy dog.` is
**10 tokens** (`The`, ` quick`, ` brown`, ` fox`, ` jumps`, ` over`, ` the`, ` lazy`,
` dog`, `.`). So 49,637 topics yield 496,370 vectors and 8.4M examples, against 839,602
for the baseline — of which the train split is 7,553,910 against 755,391 (§3.1). The
paper's single Wikipedia epoch is 755,391 examples, **2,951 steps at batch 256**; one epoch
of the pangram pool would be 29,510 steps.

**(c) Epoch count is the wrong budget, and this is not "training on a fraction of the
data".** Detailed below, because this is the part that was unclear.

### 4.1 Why budget by examples seen rather than by epochs

The concern is fair: holding the budget fixed does mean arm B makes one pass over ~10% of
its pool. But that is not the same as discarding 90% of the *data*, because the pool is a
cross product.

With the pangram prompt each topic contributes 10 vectors × ~17 labels = ~169 examples.
The 44,673 train topics are the same topics either way. Spending the paper's budget of
755,391 examples on this pool means the shuffled sampler draws ~17 examples per topic —
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

Budget: **755,391 examples seen**, the paper's single Wikipedia epoch over the train split
— 2,951 steps at its batch size of 256, which is exactly the `global_step` the published
checkpoint records (§3.1). Arm A is then a genuine replication (1 epoch, exactly as
published) and every other arm is a like-for-like comparison at equal compute. Arm B spends
it as 0.1 epochs of its larger pool; arm C, like A, as one full epoch.

The same reasoning is what makes a cheap first look legitimate: if the budget is a dial
rather than a property of the dataset, it can be turned down for a preliminary run and back
up once that run says the method is worth the money. §5.4.1 uses the dial.

### 4.2 Sizing on whatever GPU is chosen

Stated as work, not wall clock, so it survives a change of card. For an 8B model at bf16, a
forward pass is ~2N = **16 GFLOP/token**. A training step over a *frozen* model needs
activation gradients but not weight gradients, so backward is ~2N rather than the usual 4N:
**32 GFLOP/token**, rising to **48 GFLOP/token** when gradient checkpointing adds a second
forward pass.

The token count per example was measured, not estimated, with the Llama-3.1 tokenizer (see
§4.2.1): the interpretation template is **26 tokens**, and the reference's target string —
`label + '"' + '<|eot_id|>'` — averages **13.3 tokens**. So a useful example is **39.3
tokens**, and the reference's own batching pads that to **53.0**.

Full-budget runs are 755,391 examples (§4.1). The four configurations below are cumulative,
and the fourth is the one §4.2.1 recommends; A100 hours divide the work by (peak bf16
TFLOP/s × ~0.35 achieved) = ~109 TFLOP/s.

| configuration | tok/example | GFLOP/token | work | A100 h |
|---|---|---|---|---|
| reference batching, checkpointed | 53.0 | 48 | 1.92 EFLOP | ~4.9 |
| + length bucketing | 39.7 | 48 | 1.44 EFLOP | ~3.7 |
| + gradient checkpointing off | 39.7 | 32 | 0.96 EFLOP | ~2.4 |
| + shared-prefix KV cache | 28.7 | 32 | **0.69 EFLOP** | **~1.8** |

The order is forced, not chosen: checkpointing must come off before the prefix cache can do
anything at all (§4.2.1).

| one-off | tokens | work | A100 h |
|---|---|---|---|
| Extraction, 49,637 topics, pangram prompt (81 tok/topic) | 4.0M fwd | ~64 PFLOP | ~0.16 |
| Extraction, 49,637 topics, baseline prompt (~41 tok/topic) | 2.0M fwd | ~33 PFLOP | ~0.08 |
| Extraction, val topics only, baseline prompt | 0.2M fwd | ~3 PFLOP | ~0.01 |

The 4.9-hour top row is a useful check on the model: the paper ran that configuration on a
single A100 and reported 180-220 GPU-hours across every training and evaluation run in the
whole paper. Extraction is ~9% of one optimised full-budget run — hence extract-once.

### 4.2.1 The savings the reference leaves on the table, and what they cost to build

All are exact — no approximation, no change to the optimizer step, no effect on the loss.
Together they take a full-budget run from 4.9 to 1.8 A100-hours, i.e. **2.8× cheaper**, for
roughly 90 lines of trainer code. They apply to every arm and every phase.

**Length-bucketed batching — ~20 lines, 1.35×.** `compute_loss` pads every sequence in a
batch to the batch's longest label. Measured over 135,096 real targets, label length has
mean 13.3 and max 40, so a batch of 256 random draws almost always contains a near-maximal
label: padded cost is 53.0 tokens/example against 39.3 useful, a 35% tax. Draw each batch
from a shuffle buffer of ~50 batches sorted by label length, then shuffle the batch order.
That brings the tax to 0.9%. Label lengths are tokenized once at startup and cached.

The only cost is that batches become length-correlated, hence mildly content-correlated —
the buffer is still a random sample of the stream, so the correlation is bounded, and at
4,097 parameters and batch 256 it does not matter. This is the one to do first: lowest risk,
largest single saving, and it cannot change the loss, only which examples share a step.

**Logit slicing — ~10 lines, memory not FLOPs.** Covered under memory below. Worth listing
here because it also *simplifies*: the reference computes the loss in a per-example Python
loop over `outputs.logits`; slicing hidden states before the LM head lets that collapse into
one batched cross-entropy.

**Shared-prefix KV cache — ~60 lines plus an equivalence test, 1.39×, and it has a
prerequisite.** The template tokenizes to 26 tokens with the injection slots at positions
**11 and 22**. Positions 0-10 therefore precede any injection: their keys and values are
byte-identical for every example in the corpus and have no dependence on the projection, so
they need neither a forward recomputation nor a backward pass. Compute that 11-token prefix
once at startup, expand it across the batch as a frozen `past_key_values`, and start each
step at position 11. That removes 11 of every 39.7 tokens, and it is exact because the
prefix is causal and constant — later positions still attend to it through the cached K/V.

It is the fiddly one. The footguns, in the order they will be hit:

- **It does not compose with gradient checkpointing.** Verified in the installed
  transformers 4.57.6: `GradientCheckpointingLayer.__call__` in `modeling_layers.py` sets
  `use_cache=False` **and nulls `past_key_values`** whenever checkpointing is on and the
  model is in training mode. It only emits `logger.warning_once`, so the cache is silently
  discarded and the run merely gets slower. Checkpointing must be off, which is why §4.2's
  table orders the two that way.
- HF `Cache` objects are mutated in place, so the prefix cache must be rebuilt (or cropped)
  each step rather than shared across steps.
- `position_ids` / `cache_position` and the causal mask must both account for the 11-token
  offset, and the expanded K/V must not be written into.

Mitigation is a single test: the loss on a fixed batch must match the uncached path to
within bf16 noise. If that test is awkward to write, the saving is not worth taking.

**Recommended order.** Bucketing and logit slicing are unconditional — a day's work between
them, no meaningful risk, and they cut the run from 4.9 to 3.7 A100-hours while making the
memory budget easier. Turning checkpointing off is then a config change gated on measured
memory (§4.2), worth 1.5× on its own. The prefix cache is the only real engineering, is
worth 1.39×, and should be attempted last — after step 0 has shown the run is long enough
to be worth another day of work.

**Memory** is the binding constraint below 48 GB, and the culprit is not what it looks
like. 8B bf16 weights are 16 GB. Batched the reference's way, a *single* batch of 256 is
~13.6k tokens and uncheckpointed activations are roughly 32 GB — which is why the reference
configs enable checkpointing, and why an earlier draft of this plan was wrong to suggest
simply turning it off. The way out is not a bigger card but a smaller micro-batch, below.

But even checkpointed, the **logits tensor** dominates: 13.6k tokens × 128,256 vocab ×
2 bytes = 3.5 GB materialised, and cross-entropy backward wants roughly another copy. Add
~3 GB of checkpointed layer boundaries and a 24 GB card is over budget at batch 256. Three
responses, all of which the plan takes:

- **Compute logits only over label positions.** The reference materialises them for the
  whole sequence, but only the 13.3 label tokens of each 39.3-token sequence contribute to
  the loss. Slicing before the LM head cuts this tensor to 256 × 13.3 = 3,400 positions —
  **0.87 GB**, a 75% cut — at no cost in correctness. This is available to us precisely
  because we are writing the trainer (§6, D3=b).
- **Gradient accumulation is what buys the checkpointing-off path.** Since checkpointing is
  a prerequisite to remove (§4.2.1) and is itself a 1.5× tax, the memory has to come from
  somewhere else, and micro-batching is where. A micro-batch of 64 accumulating to a global
  256 is ~2.5k tokens — roughly 6 GB of uncheckpointed activations on top of 16 GB of
  weights and 0.2 GB of sliced logits, so ~22.5 GB: tight on a 24 GB card, comfortable at
  micro-batch 32. The optimizer step stays bit-identical to the paper's global batch of 256;
  the only cost is perhaps 5-10% utilisation.
- **The savings of §4.2.1 also cut memory**, since they cut tokens: with bucketing the step
  is ~10.2k tokens rather than ~13.6k, and with the prefix cache ~7.3k.

Step 0 measures which combination actually fits and which is fastest on the card used. The
honest expectation is that a 24 GB card needs micro-batch 32-64 uncheckpointed, and that an
80 GB card needs none of this care.

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

There is also a simpler option that needs no distributed code at all: phases 1-2 are four
independent runs (§5.5), so on an N-GPU rig you can run N configs concurrently, one per GPU,
and reach full utilisation with nothing but a device flag. DDP is what makes a *single* run
faster; one-run-per-GPU is what makes the *matrix* faster, and it is the better deal for
phases 1-2. Phase 0 is the exception — it is one run, so only DDP shortens it, and that is
the case where the optional path earns its keep. The trainer should support both, with
one-run-per-GPU as the default.

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

### 4.5 Validation is not free, and can silently dominate

The val split is 4,964 topics, which is ~84,000 val examples for a one-vector-per-topic arm
and ~839,000 for arm B. Validating on the *full* split every 50 steps — the reference
config's cadence — would run 59 full validations over a 2,951-step run. At forward-only
cost that is still the equivalent of ~1.8M training examples, i.e. **more than twice the
cost of the training it is monitoring**. The reference has a
`_check_validation_compute_ratio` guard precisely because this trap is easy to fall into.

Policy for this plan: validate on a **fixed random subsample of 5,000 val examples**, drawn
once with a fixed seed and reused at every validation so the curve is comparable point to
point, every 100 steps — 29 validations over a full-budget run. The full val split is used
once, at the end, for the reported numbers.

Note the ratio got *worse* when the training got cheaper, and this is the second time this
plan has had to correct it. With gradient checkpointing off, a forward pass is half the
cost of a training step rather than a third, so 29 × 5,000 val examples against 755,391
training examples is **~10% of the run**, not the ~6% an earlier version claimed. Every
saving applied to training applies to validation too, so the fraction is roughly stable
under the §4.2.1 work — but it is a fraction of a smaller number.

The end-of-run evaluations are not a rounding error either, and §4.6 now counts them
explicitly rather than folding them into a flat percentage. Each pass over the full val
split (84,211 examples) is ~39 PFLOP, about 0.10 A100-hours, and phase 0 needs three of
them: arm B, the published upstream adapter, and the untrained floor.

### 4.6 Total cost of the plan

Per-card figures assume ~35% of peak bf16 and the best configuration of §4.2; they carry
maybe ±40% until step 0 measures the real rate. The plan is bought in two instalments, and
**phase 0 is a complete result, not a probe** — it is arm B at full budget, and what it
answers is the question the plan exists to ask.

**Phase 0, itemised.** Every one-off the experiment needs is here, so nothing is left to be
discovered later. Extraction covers *both* prompt styles over the whole corpus, not just
what phase 0 strictly needs, because extract-once (§3.2) means phases 1-2 then require no
extraction at all and the marginal cost of the baseline set is 0.08 A100-hours.

| phase-0 component | work | A100 h |
|---|---|---|
| Step-0 probe: compliance generation, throughput and memory benchmarks | — | ~0.2 |
| Extraction, pangram prompt, all 49,637 topics | 64 PFLOP | ~0.16 |
| Extraction, baseline prompt, all 49,637 topics | 33 PFLOP | ~0.08 |
| Training, arm B, 755,391 examples at 2,951 steps | 693 PFLOP | ~1.76 |
| In-run validation, 29 × 5,000 examples (§4.5) | 67 PFLOP | ~0.17 |
| Val loss, arm B, on the 84,211-example val subsample | 39 PFLOP | ~0.10 |
| Val loss, untrained floor, same pangram subsample | 39 PFLOP | ~0.10 |
| Val loss, published upstream adapter, baseline val (the D10 check) | 39 PFLOP | ~0.10 |
| Retrieval eval (§5.6, D11): index build, then arm B at 10 positions, the floor, and the upstream reference | — | ~0.4 |
| **phase 0 total** | **~0.97 EFLOP** | **~3.0** |

**Arm B's val split is scored on a subsample, and that is deliberate.** Arm B has 10 val
vectors per topic, so its full val split is ~842,110 examples against the baseline's 84,211
— a full pass would cost ~1.0 A100-hours, and its untrained floor another ~1.0. Score both
on a **fixed, seeded 84,211-example subsample** instead: same size as the baseline's full
pass, drawn once, reused everywhere. State that it is a subsample, and never change what
"val" means between two numbers being compared.

| | 24 GB Ampere | A100 | H100 |
|---|---|---|---|
| effective bf16 throughput assumed | ~24 TFLOP/s | ~109 | ~346 |
| **Phase 0** — arm B at full budget, all-in | **~14 GPU-h** | **~3.0 GPU-h** | **~0.9 GPU-h** |
| **Phases 1-2** — four more full-budget runs (§5.5) | ~39 GPU-h | ~8.6 GPU-h | ~2.8 GPU-h |
| **total if the whole plan runs** | **~53 GPU-h** | **~11.6 GPU-h** | **~3.7 GPU-h** |
| micro-batching needed to drop checkpointing? | yes, 32-64 | at 40 GB, yes | no |

Phases 1-2 cost less per run than phase 0 because they inherit its extraction and its probe:
each is ~0.80 EFLOP — training, in-run validation, one full-val pass, one generation eval —
or ~2.1 A100-hours.

**Why phase 0 is worth its own instalment.** Arm B at full budget, scored by retrieval
against an untrained projection on the same pangram vectors, is a complete answer to the
question the plan exists to ask: can the adapter recover a topic the response never states?
Both of its comparators are forward-only, so for ~27% of the full plan's cost you get the
headline result. What phases 1-2 add is arm C, the capacity sweep, and a replication of arm
A under our own trainer — all real, none of them the headline.

Wall clock for phases 1-2 is their cost divided by GPU count, since the four runs are
independent (§4.3). Phase 0 is a single run and cannot be parallelised, so ~1.8 hours of it
is serial on an A100 no matter how many cards are available.

Against the first version of this plan — five full-budget runs batched the reference's way,
~26 A100-hours with nothing to show until several finished — phase 0 buys the headline for
~11% of that, and §4.2.1 plus the corrected budget of §3.1 take the whole plan down by
~57%, even after counting evaluation costs the earlier version had buried in a flat 8%.

The spread is ~14× in GPU-hours across card classes, which is much wider than the spread in
rental rates — so on a per-dollar basis the classes land within roughly a factor of two of
each other, while wall clock does not. Pick on how long you are willing to wait, and re-derive
from step 0's measured rate rather than from this table.

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

Per prompt style, into `outputs/<output-dir>/`:

- `vectors.pt` — `[n_vectors, 4096]` bf16, **raw (uncentred)**, in topic order, positions
  contiguous per topic.
- `topics.json` — one entry per surviving topic: its labels (once, not duplicated), its
  split as given by the upstream dataset, and its vector index range as `start` and `count`.
  The pangram style adds `variant` (§9.2).
- `positions.json` — run metadata only: prompt style, layer, model, the decoded position
  tokens, and the counts.
- `position_means.pt` — `[n_positions, 4096]` fp32, the per-position means of §5.3.
- `filter_report.json` — pangram style only: keep rate, `variant_counts`,
  `first_mismatch_histogram`, and every rejection with its first divergence.

No per-vector index map is stored: a vector's position is `i - start`, from `topics.json`.
The means are written rather than applied, so the centering choice (D2) stays revisitable
without re-extracting — which means **the trainer must subtract them itself** (§9.2).

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

Downstream interpretation-time use continues to inject **raw** activations, matching
upstream (`interpret.py` docstring, and the reference's own bridge-entity sweep). This is a
deliberate train/interpret mismatch inherited from upstream, on the reasoning that training
on contrastive vectors should generalise better out of distribution. The 10 mean vectors are
saved next to the checkpoint so the choice can be revisited without re-extracting.

Two other things are **centred**, and neither is a deviation from upstream:

- *Validating* the trainer against the published checkpoint's recorded `best_val_loss` of
  1.3662 (§6 step 2, §7 D10). That number came from upstream's own `validate()`, which draws
  from the same mean-subtracted vectors as training, so reproducing it needs centred vectors
  too (`plans/pangram_step0_benchmarks.md`'s gate).
- The **embedding-retrieval eval** (§5.6, D11). The paper's 94% recall@1 is explicitly "for
  contrastive vectors", and its Figure 1 shows the eval path extracting *h* and subtracting
  the mean over all topics. Raw injection belongs to the bridge-entity/TwoHopFact sweep, not
  to topic identification.

So "raw at eval time" is never the general rule — it is specific to downstream
interpretation and to the out-of-distribution bridge-entity case. Anything scored against an
upstream number uses the centring that upstream number was computed with. Arm B is also
scored raw as a secondary condition, because raw is what the taboo pipeline will actually
feed it (D6); report the two separately and never merge them into one number.

### 5.4 Arms

All arms share: layer 19, the same 49,637 topics, the same upstream splits, the same
per-position centering, and — within phases 1-2 — the same examples-seen budget. Because
the budget is equal, **every arm costs the same to train**: the cost of an arm is exactly
"one more run". Phase 0 (§5.4.1) runs arm B alone at the full budget and scores it against
forward-only comparators rather than against trained arms.

**What is and is not comparable between arms.** Arm A reads its activation from a prompt
that *names the topic out loud*, one token after the model has read the name. Arms B and C
read activations from a response whose surface text is identical for every topic, so the
topic is present only as an unverbalised influence. Those are different tasks, and their
cross-entropies are not on a common scale — a lower loss for A says nothing about whether
the pangram prompt works. Concretely:

| comparison | valid? | why |
|---|---|---|
| B vs C | **yes** | same vectors, same task, differ only in where pooling happens |
| B or C vs an untrained projection on the *same* vectors | **yes** | same task, same population; this is the floor that matters |
| B or C vs arm A's loss, or vs 1.3662 | **no** | different task, different example population (~842k vs 84k), different topic set |
| our arm A vs 1.3662 | **yes** | same task, same vectors, same budget — a replication check (D10) |
| B or C vs arm A by **recall@k** | **as a labelled reference point only** | recall@k shares a scale and a random floor (1/49,637) across arms, so the number is readable — but the task still differs, so it is never a target or a gate |

Pool sizes below are the **train split** (44,673 topics); the val split adds 84,211 and
842,110 respectively and is never trained on.

| arm | vectors per topic | train pool | what it tests |
|---|---|---|---|
| **A** baseline | 1 (last prompt token, upstream prompt) | 755,391 | replication of upstream |
| **B** pangram-per-position | 10 (one per response token) | 7,553,910 | the proposed method |
| **C** pangram-mean | 1 (mean of the 10 positions) | 755,391 | pooling before the adapter |

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
per-position training earns its pool. Either way the comparison is internal to the pangram
prompt, which is what makes it valid; arm A does not enter it.

**Arm A is run once, with `scalar_affine` only.** Its purpose is a replication check that
gives confidence in the upstream numbers, so it does not need to be repeated across adapter
architectures.

### 5.4.1 Phase 0: arm B alone

The plan as first written produced no signal at all until three full-budget runs had
finished. That is the wrong shape. Phase 0 is **arm B alone at the full budget** — 755,391
examples, 2,951 steps — with every one-off the whole experiment needs, for ~3.0 A100-hours
(§4.6). It is not a probe or a preliminary: it is a complete answer to the question the plan
exists to ask.

**The headline is retrieval accuracy against an untrained floor on the same vectors.** The
question is whether a frozen model, handed one activation from a response that never names
the topic, can be made to say what the topic was. That is a question about generated text,
so it is measured on generated text: decode a description per held-out vector and score it
by embedding retrieval against the 49,637-topic index (§5.6, D11). The floor is an untrained
projection on **pangram** vectors — same task, same topics, same index — and the paper's own
1% untrained baseline says what that floor should look like.

Phase 0 therefore produces:

| | vectors | cost | what it gives |
|---|---|---|---|
| **arm B, retrieval** | pangram val | ~0.2 A100-h | **the headline** |
| **untrained floor, retrieval** | pangram val | ~0.1 A100-h | **what the headline is measured against** |
| published upstream adapter, retrieval | baseline val | ~0.05 A100-h | a labelled reference point (§5.4) |
| arm B, val loss | pangram val subsample | ~0.10 A100-h | the training curve's endpoint |
| untrained floor, val loss | same subsample | ~0.10 A100-h | that endpoint's floor |
| published upstream adapter, val loss | baseline val | ~0.10 A100-h | the D10 trainer check, not an arm comparison |
| arm B at full budget | — | ~1.76 A100-h | the run itself |

Every comparator is forward-only, so none of them costs a training run.

**Arm B's loss is reported, not compared to arm A's.** Put it beside the published losses
from the SelfIE paper as orientation for a reader who knows those numbers, and say plainly
that the tasks differ. The only loss comparison phase 0 makes is arm B against its own
untrained floor on the same vectors.

Two honest caveats, neither a reason to skip it:

- Retrieval accuracy crosses trainers only in the reference-point row, which is labelled as
  such. What it does depend on is our loss path being right at all; §6 step 2's check bounds
  that by scoring the *published* adapter through our loss path against its recorded 1.3662
  on baseline vectors. If that fails, no phase-0 number means anything, and it is worth
  knowing for ~0.1 GPU-hours before the training run starts.
- It does not test arm C, so a result is attributable to "the pangram prompt", not to
  per-position training specifically. That attribution is what phase 1 buys.

**Before the real run, a throwaway debug run of ~50 steps.** Not an experiment and not a
line in the budget — minutes of GPU time, folded into step 0 — whose only job is to catch a
pipeline that is wrong: extraction misaligned, centering applied twice, the loss path
mis-indexed. An earlier version of this plan made this a 1/8-budget experiment; that was
paying for a measurement it could not fairly interpret, since 1/8 budget against a converged
adapter is unfair in a known direction and a loss there would have meant nothing.

**Run it as a complete run.** The cosine schedule is laid out over its own 2,951 steps with
10 warmup steps, matching upstream exactly.

**Extract everything, once.** Phase 0 extracts both prompt styles over the whole corpus even
though it strictly needs only the pangram set plus baseline *val* vectors, because the
baseline set costs 0.08 A100-hours and buying it now means phases 1-2 need no extraction at
all. Do not shrink the topic corpus: subsampling saves almost nothing while introducing a
confound between phases and forcing a re-extraction. Likewise **reducing the 10 positions to
a subset saves nothing** — at a fixed examples-seen budget the pool size does not enter the
cost at all (§4.1), so cutting positions would cut coverage for free money that does not
exist.

### 5.5 Adapter architectures

Both reference projection types, with rank as a **config field, never a literal in code**:

- `scalar_affine` — 4097 parameters.
- `scalar_affine_plus_low_rank` — `4096 + 1 + 2·4096·r`. First runs at r=16 and r=64;
  these are config values, and other ranks must need no code change.

To keep the run count sane, the questions are separated, and the cheap one comes first:

| phase | question | new runs | budget each |
|---|---|---|---|
| 0 | can the adapter recover a topic the response never states? | B × `scalar_affine` = 1 | 755,391 |
| 1 | is the result from the prompt or from per-position training? | A, C × `scalar_affine` = 2 | 755,391 |
| 2 | does capacity help the winner? | winning arm × {r=16, r=64} = 2 | 755,391 |

Five full-budget runs in total, at one equal budget, and **phase 1 reuses phase 0's arm-B
run rather than repeating it**: same trainer, same budget, same seed, same `scalar_affine`
config, so it *is* phase 1's arm B. Only A and C remain to be trained.

Phase 0 is a gate as well as a result (§5.4.1). The gate is **whether arm B clears its own
untrained floor by a margin worth attributing** — on retrieval accuracy first, and on val
loss as support. If it does not, the pangram activations did not carry recoverable topic
signal that this adapter could learn, the interesting question becomes *why*, and arm C is
not automatically the next thing to buy. If it does, phase 1 asks where the signal came
from. The gate is never "did arm B beat 1.3662" (§5.4).

### 5.6 Measuring

- **Retrieval accuracy — the headline (D11).** Decode one description per held-out vector
  and score it by embedding retrieval against a GTE-large index of all 49,637 topics
  (title plus all labels), reporting recall@{1,5,10}. This is the paper's own metric for
  contrastive vectors, so its published 94% recall@1 against a 1% untrained baseline gives
  the scale to read our numbers on. Reuse the reference's
  `evals/embedding_retrieval/topic_retrieval_eval.py` rather than inventing a metric; the
  build is `plans/pangram_step2d_retrieval_eval.md`.

  Unlike cross-entropy, recall@k has a common scale and a common random floor (1/49,637)
  across every arm, so it is the one number that can be *shown* beside arm A's without
  being *compared* to it (§5.4). Every arm must share the index, the query topics, the
  decoding settings and the seed, or nothing in the table means anything.
- **Validation loss vs examples seen**, per arm, on the fixed subsample of §4.5 during the
  run and on a fixed seeded val subsample at the end (§4.6). Comparable **within** a prompt
  style — same label set, same template, same held-out topics — and not across styles
  (§5.4). The curve, not just the endpoint: it is what says whether the budget was adequate.
- **Per-position breakdown for arm B** — evaluate positions 0..9 separately. This falls out
  of the retrieval eval at no extra design cost, since arm B is scored at every position
  anyway and its primary number is the mean over them. **Exploratory: it does not feed the
  B/C conclusion.** Its value is in shaping the next iteration — if late positions (`lazy`,
  `dog`, `.`) carry most of the topic signal, a follow-up could read only those and shrink
  the pool. Remember position 9 exists only for the ~68% of topics that wrote the full stop
  (§9.3).

## 6. Implementation steps

**Step 0 — probe.** On the real 8B: greedy-generate for a few hundred topics with the
pangram prompt; classify every non-compliant output into the categories in §5.1 and report
the distribution; benchmark examples/second and peak memory on the chosen card across the
configurations §4.2 puts in play — reference batching, then bucketing, then checkpointing
off at micro-batch 32/64/128, then the prefix cache — and confirm the prefix-cache path
reproduces the uncached loss on a fixed batch; and finish with a ~50-step throwaway debug
run of the real arm-B config to shake out a mis-wired pipeline (§5.4.1). Output is a short
findings note plus the settings the real runs use. Throwaway `.tmp.py`.

The token-length measurements this step used to carry are already done and are baked into
§4.2: template 26 tokens, injection slots at 11 and 22, target `label + '"' + eot` mean
13.3 tokens over 135,096 real targets (p90 18, max 40), reference batching 53.0
tokens/example at batch 256.

**Step 1 — extraction scripts. Done; see §9.1 for what was built and §9.2 for what building
it settled.** Two CLIs over a shared module, each writing the files of §5.2 for its own
prompt style, reading `keenanpepper/fifty-thousand-things` from the Hub or a local JSONL
copy. Light-imports-first per the project CLI convention. Unit-tested against Llama-3.2-1B
for shapes, filter logic, split inheritance, and per-position centering.

**Step 2 — trainer.** Decided (D3=b): a small trainer reusing the already-installed
`selfie_adapters.projection.create_projection_module`, writing the same checkpoint dict the
reference does (`projection_state`, `model_dim`, `checkpoint_format_version`, `config`),
which `selfie_adapters.load_adapter` reads directly — so `interpret.py` keeps working
unchanged. Budget expressed in examples seen; projection type and rank from config;
`normalize_input: true` to match the published run (§3.1); and the learning-rate schedule
laid out over *the configured budget* rather than over a fixed step count (§5.4.1).
Length-bucketed batching and logit slicing from the start; shared-prefix cache behind a
flag, enabled once step 0 shows it matches the uncached loss, and mutually exclusive with
gradient checkpointing (§4.2.1). Must also be able to *evaluate* a checkpoint it did not
train — the phase-0 comparators of §5.4.1 are the published upstream adapter and an
untrained projection.

The risk is drifting from the reference's optimizer/loss details. **The check is cheap,
quantitative, and available before any training run**: score the published upstream
checkpoint through our loss path on the val split and compare against its recorded
`best_val_loss` of 1.3662 (§3.1). If we reproduce it, our data pipeline, template, target
construction, and loss indexing all agree with upstream. If we do not, nothing downstream
means anything, and we know that for ~0 GPU-hours instead of after a full arm-A run.
Arm A's own run in phase 1 remains the stronger replication, but it is no longer the first
line of defence.

**Step 2d — the retrieval eval (D11).** The generation path and the GTE-large retrieval
scoring that produce the headline number, built on step 2a's loaders and needing no trainer.
Reuses the reference's `evals/embedding_retrieval/topic_retrieval_eval.py` for the index and
recall@k, and `interpret.generate_interpretations_batch` for injection, so no second
soft-token path exists. No GPU to build; see `plans/pangram_step2d_retrieval_eval.md`.

**Step 3 — phase 0.** Extract both prompt styles over all topics, then the single
full-budget arm-B run, scored by retrieval against an untrained floor on the same pangram
vectors (§5.4.1). Stop here and report; the gate decides whether step 4 happens.

**Step 4 — phases 1 and 2.** The four remaining runs of §5.5 — arm A, arm C, and the two
capacity runs; phase 0 already supplied phase 1's arm B — on the machine the user nominates
at execution time (**[D5]**). No extraction is needed: step 3 did it all. Checkpoints and
vectors to `outputs/`.

**Step 5 — report.** Loss-vs-budget curves per arm, generation accuracy, the per-position
exploration, the step-0 failure taxonomy, and measured timings.

## 7. Decisions taken

- **[D1]** Canonical target is the **unquoted** sentence; the filter stays. Step 0
  classifies failures into categories; revisit if quoted output exceeds ~5-10%.
- **[D2]** Per-position mean-centering for training; downstream interpretation-time use
  keeps raw activations, as upstream does (§5.3). The 1.3662 trainer-correctness check
  (D10) is a separate thing and needs centred vectors, since that is what upstream's own
  `validate()` scored.
- **[D3]** Write a small trainer (option b), not a vendored copy of the reference's. It
  must support one-run-per-GPU and, optionally, DDP (§4.3).
- **[D4]** Full budget is **755,391 examples seen** per run — the paper's single Wikipedia
  epoch over the train split, **2,951 steps** at batch 256, which is the `global_step` the
  published checkpoint records (§3.1) — held equal across all full-budget runs. An earlier
  version of this plan said 839,602 / 3,280, which is the whole dataset including its val
  split and overshoots upstream by 11%.
- **[D7]** **Phase 0 is arm B alone at the full budget** (§5.4.1), including every one-off
  the whole experiment needs: both extractions over the whole corpus, the step-0 probe, and
  the final evaluations. ~3.0 A100-hours, ~27% of the plan. Arm A is never trained in phase
  0, because phase 0 does not need it: arm B's comparator is an **untrained projection on
  the same pangram vectors**, which is forward-only and is the only floor on arm B's own
  task. The published upstream adapter
  (`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`, `wikipedia-scalar-affine.safetensors`)
  is scored too, but as a **labelled reference point and as the D10 trainer check** — never
  as arm B's target or gate, because it measures a different task (§5.4). Phase 1 then
  **reuses phase 0's checkpoint** as its arm B. Pipeline mistakes are caught by a ~50-step
  throwaway debug run inside step 0, not by a scaled-down experiment.
- **[D8]** The trainer uses **length-bucketed batching** and **logit slicing**
  unconditionally, and a **shared-prefix KV cache** for template positions 0-10 behind a
  flag (§4.2.1). All are exact. The prefix cache **requires gradient checkpointing to be
  off** — transformers silently nulls `past_key_values` under checkpointing — so the memory
  budget is met by micro-batching with gradient accumulation instead. Together with dropping
  checkpointing this is 2.8×, and it applies to every phase.
- **[D9]** Phase 0 extracts the **full** topic corpus, not a subsample. Extraction is ~0.16
  A100-hours, so subsampling saves nothing worth a confound plus a re-extraction. For the
  same reason, the 10 pangram positions are never thinned to save cost: at a fixed
  examples-seen budget, pool size does not enter the cost.
- **[D10]** The published checkpoint's recorded config is treated as authoritative over the
  paper's table and the repo's YAML where they disagree (§3.1) — in particular
  `normalize_input: true`, which this plan had not previously recorded. Its
  `best_val_loss` of 1.3662 is the trainer-correctness target, checked before any training
  run.
- **[D11]** **The headline metric is embedding-based retrieval, not validation loss.** Decode
  a description per held-out vector and report recall@{1,5,10} against a GTE-large
  (`thenlper/gte-large`) index of all 49,637 topics, documents formed as title plus all
  labels — the paper's own metric for contrastive vectors, whose published 94% recall@1
  against a 1% untrained baseline sets the scale. Chosen over cross-entropy because losses
  from two different extraction prompts measure two different tasks and are not on a common
  scale (§5.4), whereas recall@k shares a scale and a random floor across arms; and because
  the standing project question is about what the adapter can be made to *say*, not how well
  it fits labels. Scored on **centred** vectors to match the paper, with raw as a labelled
  secondary condition (§5.3). Built in `plans/pangram_step2d_retrieval_eval.md`; it costs
  ~0.4 A100-hours in phase 0 against 1.76 for the training run.
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

- Nothing blocking. Whether the taboo elicitation should *also* use the pangram prompt is a
  genuine experiment rather than a fix for a defect, and belongs in its own plan once phase 1
  has a result. It is **not** part of this plan's phase numbering — §5.5's phase 2 is the
  capacity sweep.

## 9. Execution state

What is already built, what step 0 has already measured, and the facts those two produced
that the design sections above depend on. Steps are executed through the seven step plans
listed in `plans/CLAUDE.md`; each writes its own findings note under `plans/notes/`.

| step | plan | state |
|---|---|---|
| 0 probe | `pangram_step0_benchmarks.md` | **items 1-2 done** (§9.3); items 3-5 need the step-2 trainer |
| 1 extraction | — | **done** (§9.1) |
| 2 trainer | `pangram_step2a_loss_and_eval.md`, then `pangram_step2b_training_loop.md` | **step 2a done** (`plans/notes/step2a_findings.md`); step 2b not started |
| 2c prefix cache | `pangram_step2c_prefix_cache.md` | **opt-in**; skip unless the user asks for it by name |
| 2d retrieval eval | `pangram_step2d_retrieval_eval.md` | not started |
| 3 phase 0 | `pangram_phase0_run.md` | not started |
| 4-5 phases 1-2, report | `pangram_phases12_and_report.md` | not started |

### 9.1 What step 1 built

`adapter_training/` holds three extraction modules and their tests:

| file | role |
|---|---|
| `extract_common.py` | `Topic`, `load_topics`, `left_pad`, `position_ids_from_mask`, `run_forward`, `formatted_prompt` |
| `extract_pangram_vectors.py` | the pangram style: teacher-forces the sentence plus `<|eot_id|>`, keeps one vector per sentence token, filters |
| `extract_baseline_vectors.py` | upstream's own style: renders each topic's own dataset prompt, keeps one vector at the last prompt token |

    python -m adapter_training.extract_pangram_vectors --layer 19 \
        --output-dir vectors/pangram_l19 --dataset-file <jsonl>

`outputs/` is prepended to `--output-dir` implicitly. `--dataset-file` reads a local JSONL
copy of `keenanpepper/fifty-thousand-things` (the single file
`wikipedia_vital_articles_level5_dataset.jsonl`) instead of the Hub.

Tests: `tests/test_extract_pangram_vectors.py` (17 fast tests against a fake model and
tokenizer — prompt wording, padding, filter verdicts, split inheritance, contiguous index
ranges, per-position means and their counts, batch invariance, the two-variant derivation
and its fallback), plus `tests/test_extract_baseline_vectors.py` and
`tests/test_extract_common.py`. Tests needing real weights are marked `hf_cache` and pin what
only a real tokenizer answers: the pangram is 10 tokens with the pinned decodings, batched
extraction matches unbatched, and the written artefacts have the right shapes. Run those
under the `claude` user (`gpu-exec`) — the HF cache is only readable there.

### 9.2 What building step 1 settled

- **The forced response carries a full stop, and there are two compliant variants.** The
  instruction quotes the sentence without one, but §4.2's 10-token count includes `.`. Step 0
  then measured that the model splits between the two forms (§9.3), so the extractor derives
  a second, shorter candidate from the canonical response whenever it ends in `.`, guarded by
  a token-level prefix check against the tokenizer (a tokenizer that fuses the stop into the
  last word correctly falls back to one candidate). It teacher-forces both per batch and
  keeps a topic on the first match, with-stop first.
- **`count` is genuinely per-topic**: 10 for a with-stop match, 9 for a no-stop match. This
  needed no change to the index scheme — `start`/`count` were already per-topic. Anything
  reading these directories must address a topic as `vectors[start : start + count]` and
  treat its position index as `i - start`, never assume 10.
- **Per-position means carry a per-position count**, not `sum / len(records)`, because the
  last position only has data from with-stop topics.
- **Padding-aware `position_ids`.** A plain forward pass numbers RoPE positions with
  `arange(seq_len)`, so under left padding a topic's vectors would depend on which batch it
  landed in. The extractor derives positions from the attention mask instead. The reference
  implementation does not, which is one reason our baseline vectors may differ slightly from
  upstream's.
- **Means are written, not applied.** Vectors on disk are raw, so **the trainer subtracts
  `position_means.pt` itself** (D2). If it forgets, arm B trains on uncentred vectors and
  nothing on disk says so; no code path should load vectors except through the module
  `pangram_step2a_loss_and_eval.md` builds.
- **Means are over all surviving topics, train and val**, which is what upstream's extractor
  does. Do not "fix" this to train-only without also accepting that the 1.3662 comparison
  (D10) gets weaker.
- **The baseline style filters nothing**, so its `topics.json` holds all 49,637 topics while
  the pangram one holds only the compliant ones. Whoever compares arms must intersect the
  topic sets, or an arm difference could be a topic-population difference. Implemented as
  `restrict_to_titles` (step 2a) and decided in `pangram_phases12_and_report.md`.

### 9.3 What step 0's probe measured

Real greedy generation (not teacher-forcing) on the **real 8B**, 500 topics sampled with
`seed=42` from the full 49,637.

| outcome | rate |
|---|---|
| exact `"...lazy dog."` (with stop) | 68.0% |
| exact `"...lazy dog"` (no stop) | 27.4% |
| genuine non-compliance | 4.6% |

So **95.4%** of topics produce one of the two literal strings verbatim, and forcing only one
of them would structurally cap the keep rate near whichever fraction was picked — hence the
two-variant extractor of §9.2.

**The failure taxonomy has a category §5.1 did not name.** Zero quoting, zero preamble, zero
refusals in 500 samples, so D1's "revisit if quoted exceeds 5-10%" trigger does not fire. The
dominant real failure mode (~4%) is instead the model **substituting topic words into the
pangram**: topic "Monarchism" → `"The quick brown **monarch** jumps over the lazy dog."`,
topic "24 (TV series)" → `"The quick **CTU agent** jumps over the lazy **villain**."` The
filter catches these as genuine mismatches, but see §9.4.

### 9.4 Risks not retired

- **Reproducing 1.3662 (D10) crosses trainers *and* extractors.** Ours derives padding-aware
  `position_ids` where upstream uses `arange`; batching differs; upstream's extractor only
  ever forced one target. If the check lands close but not exact, suspect extraction before
  suspecting the trainer.
- **The word-substitution mode is evidence against the centering assumption.** §5.3's
  per-position mean assumes position *p* is the same pangram word across topics. For the
  rare topic where the model gets creative, it is not. Well under D1's 5% threshold on 500
  topics, so not blocking — but re-check it at full-corpus scale from `filter_report.json`
  rather than extrapolating from the sample.

### 9.5 Running on the vast remote

- The `agent` account has **no network egress**. Models come from the `hf-fetch.sock` daemon
  (write `<repo id>\n`, read one line back), which serves **models only** and only from
  `/etc/hf-model-allowlist.txt`. Already on it:
  `meta-llama/Llama-3.1-8B-Instruct`, `keenanpepper/selfie-adapters-llama-3.1-8b-instruct`
  (the published checkpoint, D10), and `thenlper/gte-large` (the retrieval eval, D11). Start
  the 8B fetch as one of the first things you do — it is slow and can run while you set up.
- **Datasets are not served by that daemon.** Place
  `wikipedia_vital_articles_level5_dataset.jsonl` (~55 MB) by copying it into the local
  worktree directory, which the sync carries; only `*.py` files sync from the main repo.
- The synced tree lands at `<remote-root>/.claude/worktrees/<name>/...`, because the sync
  mirrors the whole repo including worktrees, and is read-only for the `agent` account. Point
  `remote_exec`'s `cwd` there and write outputs to `/home/agent/`.
- `sentence_transformers` is assumed present for D11 and cannot be installed there; check the
  import before spending GPU time.
