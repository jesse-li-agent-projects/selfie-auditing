# Reference-repro handoff: the 1.3662 investigation, resolved

**2026-08-30 update #6 — ROOT CAUSE FOUND. Everything below update #6 is
retained for provenance but its conclusions are superseded: updates #2-#5
all mis-attributed the gap to prompt construction. The gap was
`position_means.pt` having the wrong shape, so centring was silently a
no-op.**

## The bug

`extract_baseline_vectors.py` wrote `position_means.pt` as a 1-D `[hidden]`
tensor. Every reader goes through `dataset.py::load_vector_store`, which
indexes a *leading position axis*:

```python
vectors[record.start : record.start + n] -= means[:n]
```

With a 1-D means file and the baseline style's `count == 1`, `means[:1]` is
shape `[1]` — a **single scalar**, component 0 of the mean vector — which
then broadcasts across all 4096 dimensions. The baseline vectors reached the
loss effectively **uncentred**, with no error and no warning.

`extract_pangram_vectors.py` already wrote `[n_positions, hidden]`
correctly, and no test round-tripped the baseline extractor's own means
through `load_vector_store`, so nothing caught it.

Verified directly on disk:

| directory | extractor | `position_means.pt` shape |
|---|---|---|
| `outputs/baseline_l19` | ours | `(4096,)` — broken |
| `outputs/baseline_l19_v2` | ours | `(4096,)` — broken |
| `baseline_l19_mechanical` | `reextract_mechanical_prompts.tmp.py` | `(1, 4096)` — correct |
| `outputs/pangram_l19` | ours | `(10, 4096)` — correct |

The one-off `.tmp.py` happened to write `mean.unsqueeze(0)`. That, not its
prompt change, is why it scored better.

## Why every earlier measurement now makes sense

| prompt used | means shape | scored by | measured |
|---|---|---|---|
| `topic.prompt` | 1-D (broken) | reference `compute_loss` | 1.7803 |
| `topic.prompt` | 1-D (broken) | our `evaluate_adapter` | 1.7800 |
| mechanical `Tell me about {title}.` | 1-D (broken) | our `evaluate_adapter` | 1.7973 |
| mechanical | 2-D (correct) | reference `compute_loss` | **1.4065** |

Read down the "means shape" column, not the "prompt" column. The two loss
implementations agree with each other everywhere (PR #50 proved they are
numerically identical to bf16 rounding), and the prompt change moves the
number by ~0.02. Centring moves it by ~0.39.

Update #3's inference — "prompt construction alone explains the 1.7803 →
1.4065 move" — came from a run that changed *three* things at once (prompt,
`position_ids`, and, unnoticed, the means shape). `position_ids` was
correctly ruled out in PR #43. The means shape was never considered.

## Two retractions

**PR #48 (`Extract baseline vectors with the mechanical Tell-me-about
prompt`) fixed a non-bug and should not be merged.** It cites
`create_prompt()` in `data_prep/wikipedia_topics/extract_wikipedia_vectors.py`.
That script reads `vital_articles_level5.json` — a bare titles list — and
emits `labels = titles`, one label per topic, with no train/val split. It
cannot be what produced the published checkpoint.

The script that consumes *our* dataset
(`wikipedia_vital_articles_level5_dataset.jsonl`, which carries `split` and
a `labels` list) is `extract_multilayer_vectors.py`, and it uses the
dataset's own prompt field verbatim:

```python
batch_prompts = batch_data["prompt"] if isinstance(batch_data["prompt"], list) else [batch_data["prompt"]]
```

It also emits `{index, labels, split}` records — exactly the labels-file
schema `training/data.py::create_dataloaders` validates. That JSONL's
`prompt` field is LLM-generated under explicit grammar instructions
(`dataset_generation/prompts/generation_prompt.txt`: *"Be natural with
grammar for the prompt: use articles where needed, pluralize general
concepts"*). So `topic.prompt` — what the extractor did before PR #48 — is
correct, and the "grammar-cleaned rewrite" framing was never a defect.

**PR #50's headline conclusion is wrong.**
`reextract_mechanical_prompts.tmp.py` copies `topics.json` from
`baseline_l19` **unchanged** (`json.dump(records, f)  # unchanged: same
title/labels/split/start/count`). Its `prompt` field is stale metadata from
the *previous* extraction; the vectors really were built from the mechanical
template. PR #50 diffed that stale field against `baseline_l19_v2`'s real
one and concluded the grammar-cleaned phrasing reproduces 1.3662. The field
it compared was never used to extract anything.

PR #50's *other* finding — that the two loss implementations are
numerically identical — stands, and was load-bearing here: it is what makes
the "read down the means-shape column" argument valid.

## The fix (PR #51)

- `extract_baseline_vectors.py` writes `[1, hidden]`, matching the pangram
  extractor's `[n_positions, hidden]` convention.
- `load_vector_store` raises on any means file that is not
  `[n_positions, hidden]`, so a legacy 1-D file is loud instead of silently
  broadcasting. **This means every pre-fix baseline extraction directory now
  errors on load rather than producing a quietly wrong number.**
- Two regression tests: the extractor's own means round-trip through
  `load_vector_store` and actually centre; a 1-D means file raises.

## Gate status: PASSED

Re-scored after the fix — `topic.prompt` vectors, centred: **1.3636** against
the recorded 1.3662, a gap of **0.0026**, inside the plan's top "agreement;
proceed" band. Untrained floor on the same split, centred: 3.8794. See
`plans/notes/step0_findings.md` for the verdict and the tolerance reasoning.

## What is still worth knowing

**No recorded training config.** None of the three YAML configs in
`resources/selfie-adapters/training/configs/` trains on Wikipedia topics —
all three point `labels_file` at Goodfire SAE decoder vectors. There is no
config in this repo snapshot for how `wikipedia-scalar-affine.safetensors`
was actually built. The gate passing means this did not block reproduction,
but the exact recipe stays inferred rather than known.

**`resolve_device` reads `self.embed_layer.weight.device`.** Under
`accelerate` CPU/disk offload that attribute is on the meta device and the
read crashes; the robust form is `_hf_hook.execution_device`. Harmless on a
single fully-resident card, worth revisiting before any offloaded run.

**Train/val split is confirmed.** `create_jsonl_splits.py` shuffles with
`random.seed(42)` and writes exactly the filename
`adapter_training/dataset.py::DEFAULT_DATASET_FILE` expects.

**`position_ids` is inert.** PR #43 tested naive `arange` against
`position_ids_from_mask` on the real model: bit-identical activations. RoPE
attention scores depend only on the relative query/key offset, and the
causal mask excludes padding, so the constant per-row shift cancels.
`position_ids_from_mask` stays as defence-in-depth, not because it fixes
anything measurable.

## Remote environment notes

- Instance: vast-remote-broker alias `vai` (label `vai-0`), single RTX 3090
  24 GB. `HF_HOME=/workspace/hf_cache`.
- **Remote `outputs/` is writable by `agent`** — `root:agent` mode `2775`,
  setgid and group-writable. Re-checked 2026-08-30 with a live `touch` and
  `mkdir` as `agent`; both succeed. Several sessions have recorded the
  opposite and staged runs under `/home/agent/` to dodge it. That is stale,
  not wrong-at-the-time: one worktree's local `outputs/` really was `2755`
  (missing group-write), it synced to the remote that way, and a
  re-extraction lost a ~10-minute forward pass to `PermissionError` on
  `mkdir`. It was fixed by `chmod g+w` the same session, but the note
  recording the failure was never refreshed, and later sessions copied the
  symptom forward as if it were a standing property of the remote. **If you
  hit this again, check the mode before believing it — and if a worktree's
  local `outputs/` is `2755`, fix that rather than staging around it.**
- The synced source tree *is* read-only for `agent` (`root:agent`, `750` on
  directories, `640` on files). This one is real, and is a separate thing
  from `outputs/`; edit source locally.
- `resources/selfie-adapters` is gitignored and does not sync; it was copied
  into the `reference-repro-1p3662` worktree as `resources_selfie_adapters/`
  (renamed to dodge the hyphen-in-package-name import problem).
- HF fetches go through `/run/hf-fetch.sock` (write repo id + `\n`, read one
  line) and only for repos on `/etc/hf-model-allowlist.txt`.
  `keenanpepper/fifty-thousand-things` is **not** on it.
- `remote_exec` calls over ~120s auto-background as an MCP task that does not
  survive the session. Launch long work detached
  (`setsid nohup ... > log 2>&1 < /dev/null & disown`) and poll the log.
- `remote_exec` also **serializes**: a trivial `tail` issued while a long call
  is in flight appears to hang. That is queuing, not the server wedging.
  Distinguish them with a bare `echo` once the queue has drained.
- Every eval log carries
  `Ignoring corrupted tree cache file /workspace/hf_cache/.../trees/*.json: [Errno 13] Permission denied`.
  Non-fatal — it re-resolves — but the shared cache's ownership is wrong.
- `evaluate_adapter` needs `--batch-size 32` on a 24 GB card; the default 256
  OOMs, because this loss path materialises full-sequence logits per example.

## Superseded history (updates #1-#5)

Retained only so the reasoning above can be audited. **Do not act on
anything in this section**; updates #2-#5 attribute the gap to prompt
construction, which update #6 disproves.

- **#1**: our loss path scored the published checkpoint at 1.7800 vs a
  recorded 1.3662, a 0.41-nat gap, over the plan's 0.10 stop-and-report
  threshold.
- **#2**: the reference repo's own `compute_loss` scored our
  `baseline_l19` vectors at 1.7803 — matching our own number — concluding
  the loss code was not the bug and the vectors were. Correct as far as it
  went. A re-extraction with mechanical prompts then scored 1.4065, read as
  confirmation that prompt construction was the cause.
- **#3**: ruled `position_ids` out (correctly, via PR #43) and therefore
  attributed the whole 1.7803 → 1.4065 move to prompt construction. The
  re-extraction also silently changed the means shape; that was the actual
  cause.
- **#4**: applied the mechanical prompt to our own extractor, re-ran the
  gate, got 1.7973 — no improvement — and concluded our loss/eval path had
  an independent bug. The vectors were fine; the means file our extractor
  wrote alongside them was not.
- **#5** (PR #50): proved the two loss implementations numerically
  identical, retracting #4's diagnosis, then mis-read a stale copied
  `prompt` field as evidence that the grammar-cleaned phrasing reproduces
  1.3662.
