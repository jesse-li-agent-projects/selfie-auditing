---
license: mit
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
  - dummy
  - random-weights
  - testing
  - not-for-inference
library_name: safetensors
---

# Dummy SelfIE adapter (random weights, Llama-3.2-1B-Instruct)

> [!WARNING]
> **This is not a trained model.** Every weight in this repository is random.
> It exists only as a test fixture. Any output produced with it is noise by
> construction. Do not use it for inference, evaluation, or any result you
> intend to report.

## What this is

A random-weight SelfIE adapter checkpoint, in the real `selfie_adapters`
on-disk format, sized for `meta-llama/Llama-3.2-1B-Instruct`
(`hidden_dim = 2048`).

Its only purpose is to let an end-to-end test of a SelfIE pipeline run on a
laptop-sized GPU. The published, trained SelfIE adapters are 4096-wide and load
only against the 8B base model, so nothing at 1B scale can load one. Without a
fixture of the right width, a small-model test has to fabricate weights at run
time — which means the test runs a code path the real run does not, and
therefore proves less than it appears to.

## What it proves, and what it does not

**Does prove:** the adapter loader, the safetensors metadata header, the
dimension check, the projection math, and everything downstream of them
(injection, generation, scoring, aggregation, file output) all run without
crashing and produce the expected shapes.

**Does not prove:** anything at all about whether a SelfIE adapter recovers
information from a hidden state. This adapter is untrained. It has learned
nothing and cannot reveal anything. A green test says the plumbing works, not
that the method works.

## Contents

| File | Description |
|---|---|
| `selfie-random-scalar-affine.safetensors` | Random `scalar_affine` projection, `model_dim = 2048` |

Architecture matches the real trained checkpoints (`scalar_affine`,
`normalize_input = true`); only the weights differ. `init_scale` is set to the
median L2 norm of the base model's input embedding rows, so the soft token
lands at a plausible embedding scale for this model. A soft token far outside
embedding scale makes generation degenerate for reasons unrelated to the shapes
this fixture exists to test.

## Provenance

Generated at seed 0 on CUDA, by `make_smoke_weights.py` in the project that
publishes this fixture. `init_scale = 0.9332589507102966`, also recorded in this
checkpoint's safetensors metadata header.

Verify by content, not by hash. Regenerating at seed 0 reproduces the tensors
exactly, but not the file bytes: safetensors serializes the `__metadata__` map
in a per-process order, so two runs on one machine already differ by hash while
holding identical tensors and identical metadata values.

`init_scale` is the one value that depends on the device. It is a median over
the base model's embedding rows, and that reduction differs in its last bits
(`0.9332588315010071` on CPU). Each value is stable on its own device. `bias` is
scaled by `init_scale`, so it shifts with it. The relative difference is ~1e-7
and irrelevant to a fixture whose purpose is shape checking.
