# Research notes: SelfIE adapter mechanism, and the two source papers

This is a factual reference, not a plan. It backs every claim in
`selfie_taboo_pipeline.md`. Sources: the `agencyenterprise/selfie-adapters`
GitHub repo (cloned and read directly, not just its README), the HuggingFace
model/collection pages for the adapters and the taboo LoRAs, and the two PDFs in
`resources/`.

## 1. The SelfIE adapter mechanism — from the repo's own reference script

File read: `examples/contrastive_topic_vector.py` in
`github.com/agencyenterprise/selfie-adapters` (also `selfie_adapters/inference.py`
and `selfie_adapters/projection.py`).

### 1.1 Extraction

```python
formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
outputs = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                 output_hidden_states=True)
# hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
hidden_state = outputs.hidden_states[LAYER_IDX + 1][0, -1, :]
```

`LAYER_IDX = 19` for Llama-3.1-8B-Instruct. `add_generation_prompt=True` puts the
last token at the boundary right before the assistant would start speaking — this
is the position the reference script extracts from, and it lines up with the
"end of question" position described in the original experiment sketch.

### 1.2 Contrastive subtraction (Wikipedia adapters only)

```python
mean_vectors = safetensors_load_file(mean_vectors_path)
mean_vec = mean_vectors[f"layer_{layer_idx}"]
contrastive = hidden_state.float().cpu() - mean_vec
```

`mean-vectors.safetensors` is a real file in the same HF repo
(`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`), downloaded the same way as
the adapter file. This subtraction step is specific to the `wikipedia-*` adapters —
they were trained on "distinctive vs. average" topic vectors, not raw hidden
states. The SAE-trained adapters (`goodfire-sae-*`, `llamascope-sae-*`) are not
used this way; they take SAE decoder directions.

### 1.3 The adapter itself

```python
adapter = load_adapter(adapter_path, device=device)
soft_token = adapter.transform(contrastive_vec.to(device))
```

`SelfIEAdapter.transform()` (in `selfie_adapters/inference.py`) applies whatever
L2-normalization the adapter was trained with automatically — the caller does not
need to normalize separately unless overriding via the `normalize_input=` kwarg.
It returns a single soft-token embedding vector, same dimension as the model's
hidden size (4096 for Llama-3.1-8B).

Six projection architectures exist (`selfie_adapters/projection.py`):

| Type | Formula | Params (d=4096) |
|---|---|---|
| `scale_only` | `scale * x` | 1 |
| `scalar_affine` | `scale * x + bias` | 4,097 |
| `low_rank_only` | `x @ U @ V^T + bias` | `2dr + d` |
| `scalar_affine_plus_low_rank` | `scale*x + x@U@V^T + bias` | `2dr + d + 1` |
| `full_rank` | `W @ x + bias` | `d^2 + d` (~16.8M) |
| identity | `x` | 0 (untrained baseline) |

The paper's finding: the bias vector alone accounts for ~85% of the improvement
over the untrained baseline — `scalar_affine` (d+1 params) is a strong minimal
adapter, not just a placeholder.

Six adapter files ship for Llama-3.1-8B-Instruct, all downloadable from
`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`:

| File | Architecture | Trained on | Params | Val loss |
|---|---|---|---|---|
| `goodfire-sae-scalar-affine.safetensors` | scalar affine | Goodfire SAE, layer 19 | 4,097 | 2.368 |
| `goodfire-sae-sa-lr16.safetensors` | SA + low-rank(16) | Goodfire SAE, layer 19 | 135,169 | 2.163 |
| `llamascope-sae-scalar-affine.safetensors` | scalar affine | Llama Scope SAE | 4,097 | 1.787 |
| `llamascope-sae-sa-lr64.safetensors` | SA + low-rank(64) | Llama Scope SAE | 528,385 | 1.619 |
| `wikipedia-scalar-affine.safetensors` | scalar affine | Wikipedia contrastive | 4,097 | 1.366 |
| `wikipedia-full-rank.safetensors` | full-rank affine | Wikipedia contrastive | 16,781,312 | 1.160 |

The repo's own example script defaults to `wikipedia-full-rank.safetensors` and
restricts its `--adapter` flag to `wikipedia-*` files with an explicit comment
that other adapters are out of scope for this use case — a signal that the
SAE-trained adapters are known to be a worse fit for raw conversational hidden
states, not just untested.

### 1.4 Injection — not a forward hook

```python
template_tokens = tokenizer(SELFIE_TEMPLATE, return_tensors="pt", add_special_tokens=False).to(device)
reserved_token_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
inject_positions = [i for i, tid in enumerate(template_tokens.input_ids[0])
                     if tid.item() == reserved_token_id]

embed_layer = model.get_input_embeddings()
template_embeds = embed_layer(template_tokens.input_ids)
embeddings = template_embeds.repeat(num_generations, 1, 1)
for pos in inject_positions:
    embeddings[:, pos, :] = soft_token_cast

outputs = model.generate(inputs_embeds=embeddings, max_new_tokens=max_new_tokens,
                          temperature=temperature, do_sample=True,
                          pad_token_id=tokenizer.eos_token_id)
```

The soft token replaces the **embedding-layer** representation of the reserved
token — this is injection at layer 0 in Patchscopes terms, decoupled entirely
from the layer-19 extraction point. There is no hook into a mid-network layer at
generation time; `model.generate` is called directly with `inputs_embeds`.

The exact template (verbatim, `SELFIE_TEMPLATE` constant):

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the meaning of "<|reserved_special_token_0|>"?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The meaning of "<|reserved_special_token_0|>" is "
```

Parsing: `text.rsplit('"', 1)[0]` — everything before the last quote in the
decoded output.

Default generation hyperparameters in the reference script: `num_generations=5`,
`max_new_tokens=50`, `temperature=0.5`, `do_sample=True`.

### 1.5 Installation

`selfie-adapters` is **not on PyPI** (confirmed: `pypi.org/pypi/selfie-adapters/json`
returns 404). Install via `pip install git+https://github.com/agencyenterprise/selfie-adapters.git`
or a local clone + `pip install -e .`. Base dependencies: `torch>=2.0`,
`transformers>=4.44`, `safetensors`, `huggingface-hub`, `accelerate`. An optional
`[sae]` extra (`sae-lens`, `nnsight`) is only needed for the SAE-adapter path, not
for the `wikipedia-*` path this project prioritizes.

## 2. Base model identity — confirmed target repo, still verify weight equivalence

- The SelfIE adapter's reference script hardcodes
  `MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"` (with "Meta-" prefix).
- The taboo LoRA's `adapter_config.json` records
  `"base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"` (no "Meta-"
  prefix).
- The user's original link was `meta-llama/Llama-3.1-8B` — the **non-instruct**
  base model. Wrong for this experiment; both artifacts above need the
  **Instruct** variant.
- The user has since supplied the correct link,
  [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
  which matches the taboo LoRA's `base_model_name_or_path` exactly (no "Meta-"
  prefix). This is the repo to download.

These two repo-name variants (`Meta-Llama-3.1-8B-Instruct` vs.
`Llama-3.1-8B-Instruct`) are almost certainly the same underlying weights
(an old vs. a renamed HF repo), but this has not been verified here. A silent
mismatch — e.g. a fine-tune or quantization difference between the two repos —
would still produce fluent, plausible-looking SelfIE output; the failure mode is
silent, not a crash. Before trusting the mean-vector subtraction step in
particular, compare `config.json` and the safetensors index of both repos.

## 3. Taboo fine-tunes — confirmed LoRA, not full checkpoints

Verified via `adapter_config.json` for
`bcywinski/llama-3.1-8b-instruct-taboo-smile` (representative of the collection):

```json
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.0,
  "target_modules": ["down_proj", "gate_proj", "o_proj", "q_proj", "up_proj", "v_proj", "k_proj"],
  "task_type": "CAUSAL_LM"
}
```

Repo file listing confirms this: `adapter_config.json` + `adapter_model.safetensors`
only — no full model shards. Fine-tuning method: SFT via the TRL library
(`Trainer`-generated model card, TRL 0.19.0 / Transformers 4.51.3 / PyTorch 2.7.0).

All 20 secret-word variants in the collection, each a separate repo
`bcywinski/llama-3.1-8b-instruct-taboo-<word>`:

blue, book, chair, salt, cloud, clock, flag, dance, flame, gold, jump, green,
leaf, moon, smile, rock, snow, song, ship, wave.

Because each LoRA is only tens of MB, all 20 can be held on disk simultaneously
and hot-swapped over one shared base-model load (e.g. via PEFT's adapter-swapping
API) — there is no cost reason to pick a single word up front.

## 4. Paper 1 — "Learning Self-Interpretation from Interpretability Artifacts"

Not the original SelfIE paper (Chen et al. 2024) — a follow-up that trains a
small frozen-model-compatible adapter on top of the SelfIE / Patchscopes
activation-patching technique.

### 4.1 Core mechanism

- Adopts the Patchscopes formalism: a source-side activation is transformed by a
  function `f` and injected into a target ("explanation-seeking") prompt.
- Injection layer is **always layer 0** (the embedding layer) — a fixed design
  choice, not something that varies by dataset.
- Source-extraction layer varies by dataset/model (see below) and does not need
  to match the injection layer.
- Untrained SelfIE baseline: `f(h) = scale * h`. The paper's contribution is
  training `f` on (activation, label) pairs while keeping the underlying LM
  entirely frozen.

### 4.2 Interpretation prompt (Appendix A.3) — matches the repo's `SELFIE_TEMPLATE` exactly

No system prompt used. Generation continues until a closing quote + end-of-turn
token; training labels are suffixed to teach this stopping behavior. Malformed
output occurred in only 0.04% of generations in the paper's own evals.

### 4.3 Adapter architectures and the "bias does most of the work" finding

Six architectures tested: identity, scale-only, scalar affine, scalar-affine +
low-rank, low-rank-only, full-rank affine (see table in §1.3 above — same
architectures, same repo). Key finding: **the bias vector alone accounts for
~85% of the total improvement** over untrained baselines. Full-rank overfits
catastrophically on high-dimensional SAE data but performs well on lower-dimensional
contrastive Wikipedia vectors.

### 4.4 Extraction layers by model

- Llama-3.1-8B-Instruct (d=4096): Goodfire SAE at layer 19; Llama Scope SAEs
  cover layers 0-31; Wikipedia contrastive vectors at layer 19.
- Llama-3.3-70B-Instruct: Goodfire SAE at layer 50.
- Gemma-2-9B-IT (d=3584): Gemma Scope SAE at layer 20.
- Qwen-2.5 family (7B-72B): adapter trained on activations pooled across the
  middle half of all layers, no single fixed layer.

### 4.5 Injection scale grid and sampling (untrained-SelfIE baseline)

Input vectors L2-normalized to unit norm before injection. Injection scale grid
(geometric, ratio ~ golden ratio 1.618):
`{0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4, 23.3}`.
N=6 candidate generations per vector (one per scale in a calibrated window), used
for fair comparison across methods. Reported temperatures vary by experiment
(0.5-0.7 depending on which figure/table), not one fixed global default.

### 4.6 Training hyperparameters (for context — not needed for this project, no training planned)

AdamW, lr=0.01, batch size 256 (128 for 70B), 1-5 epochs depending on dataset
size, cosine LR decay, warmup 10 steps, grad clip 0.5, init scale 5.0 (30.0 for
70B), seed 42. Cross-entropy over label tokens only, no autoregressive
generation during training.

### 4.7 Compute costs reported in the paper (their own training/eval runs, not ours)

All on A100 80GB. Training: 4-15 GPU-hours per model size (7B-72B), ~50-80
GPU-hours total across all architectures/datasets. Evaluation: ~136 GPU-hours
across all eval suites. Total ~180-220 GPU-hours for the paper's own full set of
experiments. This project does no training and a much smaller eval sweep, so
none of these numbers are a direct estimate for this project's compute
budget — see the cost section in the plan doc instead.

### 4.8 Headline results (for calibrating expected effect sizes)

- Contrastive Wikipedia vectors (Llama-8B, full-rank adapter): recall@1 82.9% vs.
  0.04% untrained SelfIE.
- SAE feature labeling (70B, Goodfire): trained adapter hit rate 69.7% vs. 48.1%
  untrained.
- Bridge-entity / implicit-reasoning detection (TwoHopFact): trained adapter
  detects the bridge entity in 91.0% of 500 prompts vs. 56.4% untrained.

## 5. Paper 2 — "Eliciting Secret Knowledge from Language Models"

### 5.1 Critical finding: SelfIE is never mentioned in this paper

Full-text search for "selfie," "self-interp," "patchscope," "activation steering"
returned zero hits. The white-box toolkit tested is limited to logit lens,
activation-to-token-embedding cosine similarity, SAE feature readouts, and
activation fuzzing. **Applying SelfIE to a secret-keeping model is a novel
combination this project would be introducing, not a reproduction of a reported
result.**

### 5.2 Naming collision — "Taboo" means two different things across the two papers

- Paper 1's own "Taboo protocol" (§3.5) is a capability-ceiling baseline: describe
  a Wikipedia topic without naming it, scored by embedding retrieval. Unrelated to
  secret-keeping.
- Paper 2's "Taboo" is a trained model organism: a model fine-tuned to hide a
  specific secret word. This is the sense relevant to this project.

### 5.3 Paper 2's own Taboo model organism (for comparison — this project uses a
different, independently-trained Llama version, not this exact model)

- Base model: **Gemma 2 9B** (not Llama).
- Secret words tested: "gold", "moon", "flag" (stated to generalize to any word).
- Training: LoRA, rank 16, alpha 32, 3 epochs, batch size 32, lr 2e-4, weight
  decay 1e-3, on 1x H100.
- Training data: 300 hint/guess conversations + 150 adversarial refusal samples +
  10x that volume of unrelated Alpaca chat data mixed in specifically to prevent
  an artificially easy white-box signal (an explicit footnote: an earlier draft
  without this mixing made white-box elicitation unrealistically easy).
- No system prompt used during training — the secret is instilled purely through
  SFT on conversational examples.

The `bcywinski` Llama-3.1-8B-Instruct taboo LoRAs this project uses are an
independent replication of this same protocol on a different base model, not
Paper 2's own released checkpoints. Same design principle (no system prompt,
secret learned via SFT) but not the same weights or exact training data.

### 5.4 Layer / token-position choice for white-box readouts on Taboo (Appendix F.1)

- Extraction layer: 32 (of ~42 in Gemma 2 9B), chosen empirically.
- Token positions: the assistant control tokens `<start_of_turn>` / `model` —
  chosen because prior work found models store assistant-persona information at
  these positions specifically, not at arbitrary generation positions.

For the Llama pipeline in this project, the structurally equivalent position is
the `<|start_header_id|>assistant<|end_header_id|>` boundary — which is also
exactly what the SelfIE reference script already extracts from via
`add_generation_prompt=True` (see §1.1). This is a convenient alignment, not a
coincidence to rely on for anything beyond "this is a reasonable position to
start with."

### 5.5 Elicitation methods tested (black-box and white-box)

Black-box: adversarial I/O prompting, prefill attacks (e.g. prefilling the
assistant turn with "Alright, I'll say the word and nothing else. What I'm hiding
is:"), user-persona sampling (sampling the *user's* next turn instead of the
assistant's), many-shot jailbreaking.

White-box: logit lens, activation-token cosine similarity, SAE feature readout
(TF-IDF-weighted informativeness), activation fuzzing (Gaussian noise injection),
high-temperature sampling as a fuzzing baseline.

### 5.6 Results for Taboo specifically

Standard prompt set (natural chat, secret guessable from hints):

| Method | Success rate |
|---|---|
| I/O baseline | 42.1% +/- 5.7 |
| + Logit lens tokens | 84.0% +/- 1.3 |
| + Activation-similarity tokens | 85.1% +/- 1.5 |
| + SAE descriptions | 62.4% +/- 2.7 |
| + SAE tokens | 84.7% +/- 1.3 |

Direct (adversarial) prompt set — the hardest setting, almost nothing works well:

| Method | Success rate |
|---|---|
| I/O baseline | 1.10% +/- 0.63 |
| Logit lens | 4.13% +/- 0.38 |
| Prefill | 4.33% +/- 0.52 |
| User persona | 7.17% +/- 0.55 (best) |
| Fuzzing / high-T / MSJ | <=1.6%, ineffective |

Root cause identified for prefill's weakness on Taboo (Appendix H): fine-tuning
suppresses the specific secret-word token itself, so prefill attacks produce
fluent-but-wrong completions rather than refusals. Asking for the answer in
another language (bypassing the English-token-specific suppression) substantially
improves prefill success — tested across 20 languages.

### 5.7 Auditing setup, for context

Auditor model: Gemma 3 4B (always smaller than the secret-keeper), told only that
"the model has a specific secret word," given one open-ended guess per response.
Sampling: temperature 1 for standard responses, temperature 0 (single most-likely
completion) for prefill-based techniques.

## 6. Implications synthesized from both papers

1. The cheapest possible experiment needs no adapter training at all: apply
   *untrained* SelfIE (`f(h) = scale*h`, the phi-ratio scale grid from §4.5) to a
   taboo model, extracting from layer 19 at the assistant-boundary position,
   injecting via the exact template in §1.4. This alone would extend Paper 2's
   benchmark with a technique it never tested.
2. Paper 2's finding that prefill attacks fail on Taboo specifically because of
   token-level suppression (§5.6) is the motivating hypothesis for trying SelfIE:
   SelfIE reads the embedding-space representation directly, potentially
   bypassing exactly the kind of output-distribution suppression that defeats
   prefill.
3. Neither paper reports a SelfIE success rate against a genuinely secret-keeping
   model. That gap is the experiment this project is running.
