# Step 2a: the example pipeline, the loss path, and `evaluate_adapter`

Part 1 of 6 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read its §1, §3 and §5.2 before starting; this file does not restate its reasoning, only
what to build). Read `plans/pangram_adapter_handoff.md` for execution state.

**Deliverable**: `adapter_training/` gains a dataset module, a loss module, and an
`evaluate_adapter` CLI that scores *any* projection checkpoint on a val split. No training
loop -- that is step 2b (`plans/pangram_step2b_training_loop.md`).

**Why this is its own step**: the parent plan's cheapest and most important safety check
(§6 step 2, D10) is "score the published upstream adapter through our loss path and see if
it reproduces its recorded `best_val_loss` of 1.3662". That check needs everything in this
plan and nothing from the training loop. Getting it wrong invalidates every number the
experiment later produces, so it is built and tested on its own.

**No GPU work here.** Everything in this plan is written and unit-tested locally against
Llama-3.2-1B (`config.DUMMY_BASE_MODEL`) or a fake model. The 8B check itself is
`plans/pangram_step0_benchmarks.md`.

## Research question (quoted, parent plan §1 -- never paraphrase)

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

## What already exists

`adapter_training/` currently holds only extraction:

| file | role |
|---|---|
| `extract_common.py` | `Topic`, `load_topics`, `left_pad`, `position_ids_from_mask`, `run_forward`, `formatted_prompt` |
| `extract_pangram_vectors.py` | the pangram style, with the compliance filter and two response variants |
| `extract_baseline_vectors.py` | upstream's own style: one vector at the last prompt token |

Both extractors write the same file names into `outputs/<output-dir>/`:

| file | contents |
|---|---|
| `vectors.pt` | `[n_vectors, hidden]` **bf16, raw (uncentred)**, topic-major, positions contiguous |
| `topics.json` | JSON list of records: `title`, `prompt`, `labels` (list, stored once per topic), `split` (`"train"`/`"val"`), `start`, `count`; the pangram style adds `variant` |
| `positions.json` | run metadata: `prompt_style`, `layer`, `model`, `n_positions`, `position_tokens`, `n_topics`, `n_vectors`, `hidden_size`, `means_file` |
| `position_means.pt` | `[n_positions, hidden]` fp32 -- `[1, hidden]` for the baseline style |
| `filter_report.json` | pangram only: keep rate, `variant_counts`, `first_mismatch_histogram`, failures |

Two consequences you must handle:

- **A topic's `count` is not constant in the pangram style.** ~68% of topics match the
  with-stop variant (10 vectors), ~27% the no-stop variant (9 vectors). Address a topic's
  vectors as `vectors[start : start + count]`; its position index is `i - start`.
- **Vectors on disk are raw.** The trainer subtracts `position_means.pt` itself (parent
  plan D2, and the handoff's "Means are written, not applied"). If it forgets, arm B trains
  on uncentred vectors and nothing on disk says so. Make this hard to forget: no code path
  should load vectors without going through the module built here.

The upstream reference lives read-only at `resources/selfie-adapters/`, and its
`selfie_adapters` package is importable (`from selfie_adapters.projection import
create_projection_module`). Read `resources/selfie-adapters/training/model.py`
(`SelfIEModel.compute_loss`) and `training/data.py` (`VectorLabelDataset.__getitem__`)
before writing the loss -- they are the thing being reproduced.

## Build

### 1. `adapter_training/dataset.py`

Turns an extraction output directory into training examples.

```python
@dataclass(frozen=True)
class Example:
    """One (vector, label) training item -- the parent plan's 'example'."""
    vector_index: int
    label: str

@dataclass
class VectorStore:
    """One extraction run's vectors, centred and ready to index."""
    vectors: Float[Tensor, "n_vectors hidden"]   # fp32, centred
    hidden_size: int

def load_vector_store(directory: Path, *, center: bool = True) -> VectorStore: ...
def load_examples(directory: Path, split: str) -> list[Example]: ...
def pooled_vector_store(directory: Path) -> tuple[VectorStore, list[Example]]: ...  # arm C
```

Requirements, each one a test:

- `load_vector_store` reads `vectors.pt`, casts **bf16 -> fp32** (upstream's trainer casts
  on load; see `compute_loss`'s `vectors.to(dtype=torch.float32)`), and subtracts each
  vector's own position mean: for a vector at index `i` belonging to a topic with `start`,
  subtract `position_means[i - start]`. With `center=False` it returns raw vectors -- that
  is what *downstream interpretation-time* evaluation uses (parent plan §5.3: train centred,
  interpret raw, deliberately inherited from `interpret.py` and upstream's bridge-entity
  eval, `evals/bridge_entity/run_selfie_bridge_extraction.py`).
  **The 1.3662 reproduction check needs `center=True` instead.** Upstream's own `validate()`
  (`training/trainer.py`) draws from `self.val_loader`, and `training/data.py::
  create_dataloaders` builds train and val loaders from one single `vectors_file` (the
  contrastive, mean-subtracted `.pt`) split by the `split` field -- there is no separate raw
  path for validation. 1.3662 was computed on centred vectors; scoring the published
  checkpoint on raw vectors will not reproduce it even if everything else here is correct.
- `load_examples` flattens each topic of the requested split against **every one of its
  labels**, in `topics.json` order, and every vector of that topic. So a 10-vector topic
  with 17 labels yields 170 examples. Deterministic order -- the shuffling belongs to the
  sampler, not here.
- `pooled_vector_store` is **arm C**: mean of a topic's `count` centred vectors, one vector
  per topic, one example per (topic, label). Put it here rather than in the trainer so arms
  differ by a config field, not by a code path.
- A **topic-set intersection** helper, `restrict_to_titles(records, titles)`, because the
  baseline style filters nothing (49,637 topics) while the pangram style keeps only
  compliant ones. The handoff flags this as undecided and unimplemented; implement it here
  and let step 2b expose it as `--restrict-topics-to <dir>`. Recommended default when
  comparing arms: intersect, so an arm difference cannot be a topic-population difference.
  Do not intersect for the 1.3662 check -- that must run on upstream's own population.

### 2. `adapter_training/loss.py`

The soft-prompt forward pass and cross-entropy, reproducing `SelfIEModel.compute_loss`
exactly in what it *measures*, while being free to differ in how it batches.

The template is already in this repo, verbatim, as `interpret.SELFIE_TEMPLATE`, and matches
`training/config.py`'s `SoftPromptConfig.template`. **Import it; do not retype it.** It
tokenizes (with `add_special_tokens=False`) to 26 tokens with the
`<|reserved_special_token_0|>` injection slots at positions **11 and 22** -- assert this at
startup rather than hardcoding the numbers.

The target string for a label is exactly ``label + '"' + '<|eot_id|>'`` (upstream
`VectorLabelDataset.__getitem__`), with `strip_labels: true` meaning `label.strip()` first.

```python
@dataclass(frozen=True)
class LossConfig:
    max_loss: float = 100.0        # upstream TrainingConfig default; per-token clamp
    label_smoothing: float = 0.0
    strip_labels: bool = True
    eos_token: str = "<|eot_id|>"

class SoftPromptLoss:
    def __init__(self, model, tokenizer, projection, config: LossConfig): ...
    def __call__(self, vectors, labels: list[str]) -> tuple[Tensor, dict]: ...
```

The reduction must match upstream or 1.3662 is not comparable: **per-token CE, clamped at
`max_loss`, averaged within a sequence, then averaged over the batch** -- an unweighted
mean of per-sequence means, *not* a token-weighted mean. Upstream's `validate()` then
averages those per-batch losses over batches, which with a fixed batch size is the same
thing except for the last partial batch; match it by averaging over batches too, and note
the discrepancy is <0.1% at 84k examples.

Two departures from upstream, both exact (parent plan §4.2.1, D8):

- **Logit slicing.** Upstream materialises logits for the whole padded sequence and then
  loops in Python over the batch. Instead: run the model with `output_hidden_states=False`,
  take the last hidden state, slice the positions that predict target tokens
  (`template_len - 1 : template_len + target_len - 1` per example), and apply
  `lm_head` to only those. One batched `cross_entropy` with `reduction="none"`, then a
  masked per-sequence mean. This cuts the logits tensor ~75% and collapses the loop.
- **Right padding with an attention mask**, as upstream does (`compute_loss` pads on the
  right and masks). Do not switch to left padding here -- the injection slots are at fixed
  positions from the *left*.

Injection: embed the template once (`model.get_input_embeddings()`), `clone()` per batch,
write `projection(vectors)[i]` into both slots of row `i`, cast to the model dtype, `cat`
the target embeddings. The projection stays **fp32** (upstream keeps it in fp32 "for
training stability") and its output is cast to the model dtype at the boundary.

### 3. `adapter_training/checkpoints.py`

```python
def load_projection(source: str | Path, *, device, dim: int | None = None): ...
def save_checkpoint(path: Path, projection, config: dict, *, global_step: int,
                    best_val_loss: float | None) -> None: ...
def untrained_projection(dim: int, *, device): ...
```

- `load_projection` accepts a local `.pt`, a local `.safetensors`, or a `repo_id:filename`
  pair, and delegates to `selfie_adapters.load_adapter` / `SelfIEAdapter` so the metadata
  parsing (`config_json`, `model_dim`, `global_step`, `best_val_loss`) is not reimplemented.
  It must return the projection module *and* the recorded metadata, because
  `evaluate_adapter` prints the recorded `best_val_loss` next to the measured one.
- `save_checkpoint` writes the same dict upstream's trainer does -- `projection_state`,
  `model_dim`, `checkpoint_format_version: 1`, `config` (with a `projection` section
  holding `type`, `normalize_input`, `init_scale`, `low_rank_rank`), `global_step`,
  `best_val_loss` -- so `selfie_adapters.load_adapter` reads our checkpoints and
  `interpret.py` keeps working unchanged. Step 2b uses it; define it here so the format
  lives with the loader.
- `untrained_projection` is the parent plan's floor comparator: upstream's
  `identity_baseline.yaml` config is `scale_only` with `init_scale: 1.0`, untrained. Read
  that YAML and match it rather than inventing a floor.

### 4. `adapter_training/evaluate_adapter.py` (CLI)

```
python -m adapter_training.evaluate_adapter \
    --vectors vectors/baseline_l19 \
    --checkpoint keenanpepper/selfie-adapters-llama-3.1-8b-instruct:wikipedia-scalar-affine.safetensors \
    --split val --batch-size 256 --report eval/upstream_baseline.json
```

- `--checkpoint untrained` selects `untrained_projection`; anything else goes through
  `load_projection`.
- `--vectors` is under `outputs/` implicitly, matching the extractors' `--output-dir`
  convention (project instruction: scripts prepend `outputs/`).
- `--limit-examples N` with `--seed` draws a fixed random subsample -- this is the same
  mechanism step 2b's in-run validation uses (parent plan §4.5: a fixed 5,000-example
  subsample, drawn once, reused at every validation). Put the sampling function here
  (`subsample(examples, n, seed)`) and have step 2b import it.
- **`--center` / `--no-center` selects the mode; default is `--no-center` (raw), matching
  downstream interpretation-time usage** (`interpret.py`, the bridge-entity eval). Print
  loudly which mode was used -- a silent centring difference is exactly the bug that would
  make 1.3662 unreproducible. **The 1.3662 reproduction check must pass `--center`**: that
  number was computed on the mean-subtracted vectors upstream's own `validate()` uses (see
  `dataset.py` above), not on raw ones.
- Writes a JSON report: measured loss, example count, batch count, checkpoint metadata
  (including any recorded `best_val_loss`), vector directory, centring mode, model, layer.

Follow the project CLI convention: `argparse` and light imports at the top, `args =
parse_args()` before `import torch`, exactly as `extract_pangram_vectors.py` does.

## Tests -- `tests/test_dataset.py`, `tests/test_loss.py`, `tests/test_checkpoints.py`

Fast tests (no marker) with a fake tokenizer/model or the tiny random weights in
`dummy_weights.py`; anything needing real weights gets `@pytest.mark.hf_cache` and is run
under `gpu-exec` (the HF cache is only readable by the `claude` user).

Fast:

1. Centring: a hand-built two-topic directory where topic 0 has `count=10` and topic 1 has
   `count=9`; assert each vector had *its own position's* mean subtracted, and that the
   9-vector topic's last vector used position 8's mean, not position 9's.
2. `load_examples` yields `count × len(labels)` examples per topic, only for the requested
   split, in a deterministic order.
3. `pooled_vector_store` returns one vector per topic equal to the mean of that topic's
   centred vectors, and one example per (topic, label).
4. `restrict_to_titles` keeps the intersection and preserves index integrity (the returned
   examples still address the right vectors).
5. Target construction: ``"a label " -> 'a label"<|eot_id|>'`` under `strip_labels=True`.
6. Loss reduction: with a stub model returning fixed logits, the returned scalar equals a
   hand-computed unweighted mean of per-sequence clamped-mean token losses, and padding
   contributes nothing.
7. `max_loss` clamps per token, not per sequence.
8. `save_checkpoint` -> `selfie_adapters.load_adapter` round-trips: same parameters, same
   `normalize_input`, same type, for both `scalar_affine` and
   `scalar_affine_plus_low_rank` at some rank (rank must be a config value, never a
   literal -- parent plan §5.5).

`hf_cache`-marked, against Llama-3.2-1B:

9. `SELFIE_TEMPLATE` tokenizes to 26 tokens with two injection slots (assert the count of
   slots and that they are recovered by scanning, not hardcoded).
10. Our sliced-logit loss equals a naive reference implementation (materialise all logits,
    loop over the batch, upstream-style) on a fixed 8-example batch, within fp32 tolerance.
    **This test is the one that protects 1.3662** -- write it first and keep it.
11. Loss is invariant to batch composition: the same example scored alone and inside a
    mixed batch of different-length labels gives the same per-sequence loss.

## Done when

- `pytest tests/test_dataset.py tests/test_loss.py tests/test_checkpoints.py` passes, and
  the `hf_cache` ones pass under `gpu-exec`.
- `python -m adapter_training.evaluate_adapter --help` returns without importing torch.
- The naive-vs-sliced equivalence test (10) exists and passes.
- `plans/pangram_adapter_handoff.md` updated: step 2a done, what was decided, what 2b needs.
- Committed on a worktree branch with a PR (undrafted), per the project workflow.

## Do not

- Do not vendor or edit `resources/selfie-adapters/` -- it is read-only reference.
- Do not train anything, and do not add an optimizer here.
- Do not "fix" the train-centred/interpret-raw mismatch (`--no-center` default,
  §5.3, D2). That is about the adapter's own usage at downstream interpretation time
  (`interpret.py`, the bridge-entity eval), not about the 1.3662 gate -- which must use
  `--center` because that is what upstream's `validate()` actually scored.
- Do not change `interpret.SELFIE_TEMPLATE`. The adapter was trained against that exact
  string; the parent plan's D6 has the history of an earlier draft getting this wrong.
