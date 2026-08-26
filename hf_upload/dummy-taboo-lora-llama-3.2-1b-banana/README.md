---
license: mit
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
  - dummy
  - random-weights
  - testing
  - not-for-inference
  - lora
  - peft
library_name: peft
---

# Dummy taboo LoRA (random weights, Llama-3.2-1B-Instruct)

> [!WARNING]
> **This is not a trained model.** Every weight in this repository is random.
> It exists only as a test fixture. It has no secret word, it hides nothing,
> and any output produced with it is noise by construction. Do not use it for
> inference, evaluation, or any result you intend to report.

The word `banana` in the repository name is a placeholder that keeps the naming
parallel to the real taboo LoRA collection. This adapter was never trained on
it, or on anything else.

## What this is

A random-init LoRA adapter, with the same hyperparameters as the real
`bcywinski/llama-3.1-8b-instruct-taboo-*` collection, sized for
`meta-llama/Llama-3.2-1B-Instruct`.

Its only purpose is to let an end-to-end test exercise the LoRA load and
hot-swap path — `PeftModel.from_pretrained`, `set_adapter`,
`disable_adapter`, `PeftModel.generate` — on a laptop-sized GPU. The real taboo
LoRAs are published for the 8B base model only, so nothing at 1B scale can load
one.

## What it proves, and what it does not

**Does prove:** the adapter loads through the ordinary PEFT path, activates and
deactivates correctly, measurably perturbs the forward pass when active, and
returns a clean base model when disabled.

**Does not prove:** anything about a model concealing a secret. A trained taboo
LoRA has a secret word it has learned to withhold. This one does not, so no
interpretability result of any kind can be drawn from it.

## Hyperparameters

Matched to the real taboo LoRA collection, so the fixture has the same shape as
the real thing and only the weights differ:

| | |
|---|---|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.0 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `task_type` | `CAUSAL_LM` |
| `init_lora_weights` | **`false`** |

`init_lora_weights=false` is load-bearing, not a stylistic choice. PEFT's
default (`true`) follows the LoRA paper and zero-initializes `lora_B`, so
`ΔW = B @ A = 0` at initialization regardless of `lora_A`. Such an adapter is an
exact no-op, and a test using one would pass while testing nothing. `false`
gives nonzero random weights on both `A` and `B`.

## Provenance

Generated at seed 0 by `make_smoke_weights.py` in the project that publishes
this fixture. Byte-for-byte reproducible, and verified identical whether
generated on CPU or CUDA. 224 tensors, all fp32.
