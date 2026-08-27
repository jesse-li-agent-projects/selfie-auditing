# Pangram-prompt SelfIE adapter (layer 19, Wikipedia topics)

Status: plan, not implemented. Decision points marked **[D#]**.

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

## 2. What the reference does today

`resources/selfie-adapters` splits the work in two, and only the first half touches the
extraction prompt:

1. **Extraction** (`data_prep/wikipedia_topics/extract_wikipedia_vectors.py`): for each
   of the 50,006 titles in `vital_articles_level5.json.gz`, render `Tell me about
   {title}.` through the chat template with `add_generation_prompt=True`, run one forward
   pass, and keep the layer-19 residual stream at the **last** token. Result: one
   `[50006, 4096]` tensor plus the titles as labels. No generation happens.
2. **Training** (`training/`): a frozen Llama-3.1-8B is fed a fixed ~30-token soft prompt
   with the extracted vector projected into the `<|reserved_special_token_0|>` slot, and
   the adapter is trained by cross-entropy to make the model emit the topic title. Only
   the projection is trainable — `scalar_affine` is `scale * x + bias`, i.e. **4097
   parameters** for a 4096-dim model.

The two halves talk through files on disk (`.pt` of vectors + `.json` of
index → labels → split). That interface is what this plan reuses: everything new lives in
extraction, and training changes only by config.

## 3. Answering the training-time question

> is it accurate to say that because I'm testing more generations (testing each token in
> the response prompt, rather than just one word as done in the SelfIE paper), training
> should take longer (multiplied by however many additional generations I'm testing)?

Partly, and the part that is wrong is the useful part. Three separate things:

**(a) Cost per training example does not change.** A training step is a forward+backward
of the frozen 8B over the ~42-token soft prompt. A vector is a vector — it does not matter
which token position it came from. So there is no per-example multiplier.

**(b) Dataset size multiplies by ~10.** Measured with the Llama-3.1 tokenizer:
`The quick brown fox jumps over the lazy dog.` is **10 tokens** (`The`, ` quick`, ` brown`,
` fox`, ` jumps`, ` over`, ` the`, ` lazy`, ` dog`, `.`), 11 if the trailing `<|eot_id|>`
counts. So 50k topics become ~500k (vector, label) pairs. One *epoch* costs 10× more.

**(c) You do not have to spend that.** This is the key point. The adapter has 4097
trainable parameters. What it needs is a number of *optimizer steps*, not a number of
epochs. The reference config runs 2 epochs × 50k ÷ batch 80 ≈ **1250 steps**. Keep the
same 1250 steps against the 10× dataset and it becomes 0.2 epochs — same wall clock, and
each step sees strictly more diverse data than before. Set `training.max_steps` and
ignore `num_epochs`.

There is a second reason not to buy the 10×: the 10 positions of one topic are the same
prompt, the same label, and adjacent tokens of the same fixed sentence. They are heavily
correlated. Effective sample size is far below 10× the topics, so returns diminish fast.

### Sizing this on whatever GPU you pick

Stated as work, not wall clock, so it survives a change of card. For an 8B model at bf16,
a forward pass is ~2N = **16 GFLOP/token**; a training step over a *frozen* model needs
activation gradients but not weight gradients, so ~2.5× forward ≈ **40 GFLOP/token**.

| | tokens | work |
|---|---|---|
| Extraction, 50k topics, pangram prompt (81 tok/topic) | 4.1M fwd | ~65 PFLOP |
| Extraction, 50k topics, baseline prompt (41 tok/topic) | 2.1M fwd | ~33 PFLOP |
| Training, 1250 steps × batch 80 (~42 tok/example) | 4.2M train | ~170 PFLOP |
| Training, 1 full epoch over 500k pairs (6250 steps) | 21M train | ~840 PFLOP |

Divide by (peak bf16 TFLOP/s × ~0.35 achieved). That puts the 1250-step run at roughly
2 hours on a 24 GB Ampere-class card, well under an hour on a modern datacentre card. The
step-0 probe below replaces this arithmetic with a measurement on the card actually used.

**Memory:** 8B bf16 weights are 16 GB, so 24 GB is the practical floor and leaves little
room — expect to drop the batch below the reference's 80. At 40 GB or more you can keep
batch 80 and turn gradient checkpointing off. More than one GPU is useful here for running
arms side by side, not for sharding: one arm per card beats splitting one arm across two.

### Levers to cut exploration time, best first

1. **Budget by `max_steps`, not epochs.** Removes the 10× entirely.
2. **Run the arms concurrently**, one per GPU, if more than one card is available.
3. **Disable gradient checkpointing** if memory allows. The reference config turns it on
   for 80-example batches, but our sequences are ~42 tokens and the model is frozen;
   recomputation buys memory we probably do not need and costs ~25-30% throughput.
4. **Extract once, train many times.** Extraction is the one-off; vectors on disk make
   every subsequent config change nearly free. Extract *all* 10 positions even if the
   first run subsamples them — positions are free within a forward pass that already ran.
5. **Start with a topic subset** (e.g. 10k of 50k) to shorten the first end-to-end loop.
   Note this only cuts *extraction* time once you are budgeting by steps.
6. **Keep `scalar_affine`** (4097 params) for every arm here. Low-rank is a separate axis;
   do not vary two things at once.
7. **Smoke the whole path on Llama-3.2-1B locally** before spending any GPU-hours. The 1B
   is already in the local HF cache and the repo already has smoke scaffolding
   (`smoke/small_llama_config.py`).

## 4. Design

### 4.1 Extraction, and the filter

The naive route is: generate greedily, string-compare the output to the pangram, keep the
topics that match, then re-run a forward pass to read activations. That is a decode loop
plus a second forward.

Cheaper and exactly equivalent: **teacher-force** the prompt followed by the pangram
tokens in a single forward pass, and check at every response position `i` whether
`argmax(logits[i-1]) == pangram_token[i]`. If that holds at all positions, greedy decoding
from the prompt would have produced exactly that sentence — greedy decoding is
deterministic and follows the same prefix, so agreement at every step *is* the greedy
output. Include the trailing `<|eot_id|>` in the forced sequence so a topic that would
have run on past the sentence is also rejected.

One forward pass therefore yields both the 10 layer-19 activations and the filter verdict.
Extraction cost goes from 41 to 81 prompt tokens per topic — about 2×, not 10×, and no
decode at all.

Caveat this must be checked against, in step 0: the model may legitimately answer with the
sentence wrapped in quotes, in which case a strict filter rejects nearly everything. Probe
a few hundred topics with real greedy generation first and pick the canonical target
string from what the model actually does. **[D1]**

### 4.2 What gets written

The existing training-data format, unchanged (`resources/selfie-adapters/data/README.md`):
a `.pt` holding `[n_vectors, 4096]` and a `.json` mapping index → labels → split. Ten rows
per surviving topic, all carrying that topic's title as the label. A sidecar records
`(row → topic_idx, position_idx)` so per-position analysis is possible afterwards.

**Splits must be by topic, not by row.** All 10 positions of a topic go to the same split
or validation leaks — the val row would be a near-duplicate of a train row with an
identical label.

### 4.3 Mean-centering **[D2]**

The reference stores *contrastive* vectors (`vector - mean over all topics`), but this
repo's eval path injects raw hidden states with no mean subtraction
(`interpret.py` docstring). That inconsistency already exists upstream; this plan should
not silently inherit it.

Extra wrinkle here: with 10 positions, the residual stream is dominated by *which word of
the pangram this is*, not by the topic. The right analogue of the reference's centering is
a **per-position** mean (mean over topics, separately for position 0..9), which cancels the
token identity and leaves the topic signal.

Proposal: run **uncentered as primary** (matches how `interpret.py` will consume it), and
per-position-centered as a cheap secondary arm, saving the 10 mean vectors next to the
checkpoint so eval can subtract the identical thing. Whatever is chosen must match between
training and eval.

### 4.4 Arms for the first experiment

All three train from one extraction pass over the same topic set, same layer 19, same
`scalar_affine`, same `max_steps`, same held-out topics:

| arm | vectors | rows | purpose |
|---|---|---|---|
| **A** baseline | `Tell me about {t}.`, last prompt token | 1 / topic | matched replication of the reference; the thing to beat |
| **B** pangram-pooled | pangram prompt, all 10 response positions | 10 / topic | the proposed method |
| **C** pangram-mean | pangram prompt, mean of the 10 positions | 1 / topic | cheap control: is the gain from *more* vectors or from the *prompt*? |

Arm C matters. Without it, a win for B is unattributable between "the pangram prompt
carries more topic information" and "10× the rows".

### 4.5 Measuring

- Validation cross-entropy on held-out topics — directly comparable across arms (same
  label set, same soft-prompt template).
- Generation accuracy on held-out topics: does the adapter emit the title? Reuse the
  reference's scoring rather than inventing one.
- **Per-position breakdown for arm B**: train pooled, then evaluate position 0..9
  separately. Cheap, and it answers whether late positions (`lazy`, `dog`, `.`) carry more
  topic signal than early ones — which would reshape the next iteration.

The real test — does this surface a *hidden* thought — is the existing taboo pipeline
(`run_pipeline.py`), and it is out of scope here. Note that using this adapter there means
changing `interpret.py`'s extraction prompt to match, plus whatever `[D2]` decides.

## 5. Implementation steps

**Step 0 — probe (~30 min of GPU time).** Generate greedily for ~200 topics with the
pangram prompt on the real 8B. Record what the model actually emits, settle the canonical
target string and the filter, measure the reject rate, and time one forward pass to
replace the arithmetic in §3 with a measurement. Throwaway `.tmp.py`.

**Step 1 — extraction script.** New `adapter_training/extract_topic_vectors.py`: CLI with
`--prompt-style {baseline,pangram}`, `--layer`, `--topics`, `--limit`, writes the `.pt` +
`.json` + sidecar of §4.2 and reports the filter reject rate. Light-imports-first per the
project CLI convention. Unit-tested against the 1B for shapes, filter logic, and
split-by-topic.

**Step 2 — training path [D3].** Two options:
- *(a) Vendor* `resources/selfie-adapters/training/` into a tracked path. Zero
  compatibility risk, but `resources/` is untracked, so it does not travel with the repo —
  this means committing ~2500 lines of wandb/mlflow/dataset-mixing we will not use.
- *(b) Write a small trainer* (~150 lines) that reuses the already-installed
  `selfie_adapters.projection.create_projection_module` and writes the same checkpoint
  dict the reference does (`projection_state`, `model_dim`, `checkpoint_format_version`,
  `config`) — which `selfie_adapters.load_adapter` reads directly, so `interpret.py` keeps
  working unchanged.

Recommendation: **(b)**, per the repo's "removing features should not lengthen the
codebase" rule. The risk is drifting from the reference's exact optimizer/loss details;
mitigate by reproducing arm A and checking its validation loss against the reference
config's reported behaviour before trusting arms B and C.

**Step 3 — run.** Extract once for each prompt style, then train the arms — concurrently
one-per-GPU if the chosen machine has more than one card, otherwise in sequence.
Checkpoints to `outputs/`. See **[D5]** for where this runs.

**Step 4 — report.** Val CE per arm, generation accuracy, per-position breakdown, filter
reject rate, measured timings.

## 6. Open decisions

- **[D1]** Canonical target string / filter strictness — settled by the step-0 probe.
- **[D2]** Uncentered (default) vs per-position mean-centered.
- **[D3]** Vendor the reference trainer, or write a small one.
- **[D4]** Topic subset size for the first pass (10k proposed) and step budget (1250
  proposed, matching the reference).
- **[D5]** Where this runs. The local GPU is an 8 GB laptop card, which cannot hold an 8B
  model, so steps 0/3 need a machine the user nominates; §3 gives the sizing to pick one
  (24 GB floor, 40 GB+ comfortable). Nothing in this plan assumes a particular card.
