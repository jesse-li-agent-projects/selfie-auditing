# Pangram-prompt SelfIE adapter (layer 19, Wikipedia topics)

Status: plan, not implemented; all design decisions taken (§7).

Bought in two instalments (§4.6). **Phase 0** trains arm B alone at full budget and scores
it against the published upstream adapter, which is a fair comparator at zero training cost
— the headline result, all one-off work included, for **~2.8 A100-hours**. **Phases 1-2**
add arm C, arm A's own replication, and the capacity sweep, bringing the whole plan to
~11 A100-hours.

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
validation loss on the same 84,211 val examples this plan will use, and is directly
comparable to ours** — subject only to our trainer matching theirs, which is what §6 step 2
checks.

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
| Final full-val pass, arm B | 39 PFLOP | ~0.10 |
| Final full-val pass, published upstream adapter | 39 PFLOP | ~0.10 |
| Final full-val pass, untrained floor | 39 PFLOP | ~0.10 |
| Generation-accuracy eval (§5.6), decode-bound | — | ~0.1 |
| **phase 0 total** | **~0.97 EFLOP** | **~2.8** |

| | 24 GB Ampere | A100 | H100 |
|---|---|---|---|
| effective bf16 throughput assumed | ~24 TFLOP/s | ~109 | ~346 |
| **Phase 0** — arm B at full budget, all-in | **~13 GPU-h** | **~2.8 GPU-h** | **~0.9 GPU-h** |
| **Phases 1-2** — four more full-budget runs (§5.5) | ~39 GPU-h | ~8.6 GPU-h | ~2.8 GPU-h |
| **total if the whole plan runs** | **~51 GPU-h** | **~11 GPU-h** | **~3.6 GPU-h** |
| micro-batching needed to drop checkpointing? | yes, 32-64 | at 40 GB, yes | no |

Phases 1-2 cost less per run than phase 0 because they inherit its extraction and its probe:
each is ~0.80 EFLOP — training, in-run validation, one full-val pass, one generation eval —
or ~2.1 A100-hours.

**Why phase 0 is worth its own instalment.** The published upstream checkpoint was trained
at *exactly* this budget — 2,951 steps, batch 256, same optimizer, same schedule, same
layer, same contrastive vectors (§3.1) — and publishes its validation loss. So training arm
B at full budget yields an equal-budget, equal-hyperparameter comparison against arm A
**without training arm A**. For ~25% of the full plan's cost you get the headline result.
What phases 1-2 add is arm C, the capacity sweep, and a replication of arm A under our own
trainer — all real, none of them the headline.

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

Downstream interpretation-time use continues to inject **raw** activations, matching
upstream (`interpret.py` docstring, and the reference's own bridge-entity sweep). This is a
deliberate train/interpret mismatch inherited from upstream, on the reasoning that training
on contrastive vectors should generalise better out of distribution. The 10 mean vectors are
saved next to the checkpoint so the choice can be revisited without re-extracting.

This is a different thing from *validating* the trainer against the published checkpoint's
recorded `best_val_loss` of 1.3662 (§6 step 2, §7 D10): that number came from upstream's own
`validate()`, which draws from the same mean-subtracted vectors as training, so reproducing
it needs centred vectors too (`plans/pangram_step0_benchmarks.md`'s gate).

### 5.4 Arms

All arms share: layer 19, the same 49,637 topics, the same upstream splits, the same
per-position centering, and — within phases 1-2 — the same examples-seen budget. Because
the budget is equal, **every arm costs the same to train**: the cost of an arm is exactly
"one more run". Phase 0 (§5.4.1) runs arm B alone at a smaller budget and is scored against
free comparators rather than against trained arms.

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
per-position training earns its pool. If both beat A, the gain came from the prompt rather
than from either pooling scheme.

**Arm A is run once, with `scalar_affine` only.** Its purpose is a replication check that
gives confidence in the upstream numbers, so it does not need to be repeated across adapter
architectures.

### 5.4.1 Phase 0: arm B alone

The plan as first written produced no signal at all until three full-budget runs had
finished. That is the wrong shape. Phase 0 is **arm B alone at the full budget** — 755,391
examples, 2,951 steps — with every one-off the whole experiment needs, for ~2.8 A100-hours
(§4.6). It is not a probe or a preliminary: it is a complete, fair answer to the question
the plan exists to ask.

**Arm A comes free, and the comparison is fair.** Upstream publishes the trained baseline
adapter this repo already loads — `keenanpepper/selfie-adapters-llama-3.1-8b-instruct`,
`wikipedia-scalar-affine.safetensors` (see `config.py`) — and §3.1 reads its training config
out of the file. It was trained at *exactly* this budget with exactly these hyperparameters
on exactly this layer and this vector treatment, and its `best_val_loss` of 1.3662 is on the
same topic-held-out val split. So phase 0 does **not** train arm A. It scores three things
through one loss path, on one val split:

| | cost | what it gives |
|---|---|---|
| untrained projection (`identity_baseline`) | ~0.10 A100-h | the floor |
| published upstream adapter, on baseline val vectors | ~0.10 A100-h | arm A, converged, for free |
| arm B at full budget | ~2.0 A100-h | the question |

**B below 1.3662 is the headline result**, obtained without ever training arm A.

Two honest caveats, neither a reason to skip it:

- The comparison crosses trainers. Ours is not theirs, so a small gap either way could be
  implementation drift rather than method. §6 step 2's check bounds this: scoring the
  *published* adapter through our loss path should reproduce ~1.3662. If it does, the
  trainers agree on the thing being measured; if it does not, no phase-0 number means
  anything, and that is worth knowing for ~0.1 GPU-hours before the training run starts.
- It does not test arm C, so a win is attributed to "the pangram prompt", not to
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
| 0 | does the pangram prompt beat published arm A? | B × `scalar_affine` = 1 | 755,391 |
| 1 | is the win from the prompt or from per-position training? | A, C × `scalar_affine` = 2 | 755,391 |
| 2 | does capacity help the winner? | winning arm × {r=16, r=64} = 2 | 755,391 |

Five full-budget runs in total, at one equal budget, and **phase 1 reuses phase 0's arm-B
run rather than repeating it**: same trainer, same budget, same seed, same `scalar_affine`
config, so it *is* phase 1's arm B. Only A and C remain to be trained.

Phase 0 is a gate as well as a result (§5.4.1) — if arm B cannot beat the published
checkpoint, the interesting question becomes *why*, and arm C is not automatically the next
thing to buy.

### 5.6 Measuring

- **Validation loss vs examples seen**, per arm, on the fixed subsample of §4.5 during the
  run and the full val split once at the end. Comparable across arms: same
  label set, same soft-prompt template, same held-out topics. The curve, not just the
  endpoint — it is what says whether the budget was adequate.
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

**Step 1 — extraction script.** New `adapter_training/extract_topic_vectors.py`: CLI with
`--prompt-style {baseline,pangram}`, `--layer`, `--limit`, reading
`keenanpepper/fifty-thousand-things` and writing the three files of §5.2 plus a filter
report. Light-imports-first per the project CLI convention. Unit-tested against
Llama-3.2-1B for shapes, filter logic, split inheritance, and per-position centering.

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

**Step 3 — phase 0.** Extract both prompt styles over all topics, then the single
full-budget arm-B run, scored against the two forward-only comparators of §5.4.1. Stop here
and report; the gate decides whether step 4 happens.

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
  the final evaluations. ~2.8 A100-hours, ~25% of the plan. Arm A is never trained in phase
  0: the comparators are the untrained projection and the published upstream adapter
  (`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`, `wikipedia-scalar-affine.safetensors`),
  both forward-only, and the latter is a *fair* comparator because it was trained at this
  exact budget and configuration (§3.1). Phase 1 then **reuses phase 0's checkpoint** as its
  arm B. Pipeline mistakes are caught by a ~50-step throwaway debug run inside step 0, not
  by a scaled-down experiment.
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
