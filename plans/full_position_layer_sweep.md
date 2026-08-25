# Plan: full layer x token-position sweep

Status: ready to execute -- design decisions all resolved, nothing implemented.

Supersedes `archive/selfie_taboo_pipeline.md`. That plan's S4.4 scoped generation
to two named positions per layer to save budget, and its first pass was never
actually run through `run_pipeline.py` (ad-hoc scripts were used instead). So
this sweep is a *replacement*, not a follow-on: the code paths it changes have
no production callers to preserve.

## 1. Goal / research question

The sweep the user specified directly (not inferred):

- 2 pre-specified Taboo words
- every layer (not the every-4th-layer subset `first_pass_config` used)
- every user-prompt token position, from the "What" token in "What is the secret
  word?" through the end of the formatted prompt template
- all 3 arms (control, prompted, fine-tuned)
- 200 SelfIE generations per (arm, word, layer, position) cell

This maps to README.md "Research questions" 1 and 2, verbatim:

> 1. Will a SelfIE adapter correctly uncover something the model is actively hiding?
> 2. How does performance differ if the model is control / prompted / fine-tuned
>    to complete its task?

at finer layer/position resolution than the superseded plan's first pass.

## 2. Preflight: already done, results below

Run against the Llama-3.1-8B-Instruct tokenizer (via the ungated
`NousResearch/Meta-Llama-3.1-8B-Instruct` mirror -- `meta-llama/...` is gated and
the sandbox has no HF token; the tokenizer and chat template are the same files).
Reproduce with `explore_token_map.py`, which already prints exactly this.

Tail of the formatted prompt, identical in all three arms:

| offset | token |
| --- | --- |
| -12 | `'\n\n'` |
| **-11** | `'What'` |
| -10 | `' is'` |
| -9 | `' the'` |
| -8 | `' secret'` |
| -7 | `' word'` |
| -6 | `'?'`  (= `LAST_CONTENT_TOKEN`) |
| -5 | `'<|eot_id|>'` |
| -4 | `'<|start_header_id|>'` |
| -3 | `'assistant'` |
| -2 | `'<|end_header_id|>'` |
| -1 | `'\n\n'`  (= `ASSISTANT_BOUNDARY`) |

Total prompt lengths: control 42, prompted 46, finetuned 16 tokens.

Three conclusions that settle open questions from the previous draft:

1. **Span = offsets -11 .. -1, i.e. 11 positions.** Not the ~8 previously guessed.
2. **`'\n\n'` is its own token; it does not merge with `'What'`.** So the span
   starts cleanly at the first content token.
3. **Negative offsets are identical across arms; absolute indices are not.**
   Control/prompted prepend a system turn, shifting every absolute index. This
   is decisive for the design -- see S3.

Both named positions (`LAST_CONTENT_TOKEN` = -6, `ASSISTANT_BOUNDARY` = -1) lie
*inside* the span, so the span subsumes them. No separate named positions are
needed in the sweep config.

## 3. Design: positions are negative offsets

**Decision: every position in this sweep is a negative offset from the end of
the formatted prompt.** Absolute indices are not comparable across arms (S2
conclusion 3) -- cell `(layer=10, position=31)` means `'What'` in control and
something in the middle of the system prompt in finetuned. Negative offsets
align the three arms at the assistant boundary, which is exactly the alignment
the comparison needs.

`extract.resolve_position()` already documents and supports negative indices,
and torch tensor indexing handles them natively, so this needs no new
resolution machinery -- only the key-construction fixes in S5.

### 3.1 Locating the span: backward decode on the real tokens

Do **not** tokenize `SECRET_PROMPT` standalone and subtract its length. That
compares a standalone tokenization against an in-context one and assumes they
agree. S2 shows they happen to agree here, but the assumption is exactly the
kind this project's norms say to verify rather than bake in, and it would fail
silently under a template change.

Instead, count backward over the formatted prompt's *own* tokens:

```
end = pos_index[Position.LAST_CONTENT_TOKEN]
for start in range(end, -1, -1):
    if tokenizer.decode(ids[start : end + 1]).endswith(user_prompt):
        return list(range(start - len(ids), 0))   # negative offsets, start .. -1
raise ValueError(...)
```

The match is **lenient at the front, exact at the back**: the decoded slice must
*end with* `user_prompt`, but may begin with extra characters. That tolerates a
first token that merges template whitespace into the first content word (e.g. a
hypothetical `'\n\nWhat'`) -- the standalone-tokenization approach cannot.

Because the loop returns the **first** (largest) `start` that matches, the slice
is minimal, which is exactly the user's "first token being necessary" condition:
dropping the first token would leave a slice that no longer ends with
`user_prompt`. Minimality and necessity are the same property here, so no
separate check is needed.

Failure is loud: if no slice matches, raise, rather than returning a plausible
wrong span.

New helper in `extract.py`:

```python
def user_prompt_span(
    tokenizer: TokenizerLike,
    input_ids: Int[Tensor, "seq"],
    user_prompt: str,
    pos_index: dict[Position, int],
) -> list[int]:
```

Cost is one decode per candidate start, terminating in ~6 iterations for this
prompt, called once per (arm, word). Irrelevant next to the forward pass.

### 3.2 Declaring the span in config

Add a third `Position` member acting as a sentinel:

```python
FULL_USER_SPAN = "full_user_span"  # expands to every offset from S3.1's span
```

`config.positions` stays a static, declarative list (`[Position.FULL_USER_SPAN]`)
while the actual offsets get resolved per prompt, where the token ids exist.
Expansion happens in `extract.py`:

```python
def expand_positions(
    tokenizer, input_ids, user_prompt, positions: list[Position | int]
) -> list[Position | int]:
```

replacing the sentinel with its offsets in place, de-duplicating while
preserving order (so a config listing both the sentinel and `ASSISTANT_BOUNDARY`
does not produce a duplicate cell for offset -1).

`extract_hidden_states()` calls `expand_positions()` before its existing
`[extract] ...` print loop, so every run's own log shows all 11 selected tokens
decoded -- the verification stays in the logs rather than in a one-off script.

`run_pipeline.run()` then iterates `hidden_states.keys()` instead of
re-deriving cells from `config.positions`, since only the extraction step knows
the expanded list. This also removes the existing duplicated `product(...)`.

## 4. Design: parallel execution across GPUs

Target: K GPUs, each holding one model copy, each running one process. K is not
fixed by this design and nothing below hardcodes it -- pick it at launch time
from whatever hardware is actually available, including K = 1.

**Shard axis: samples, not cells.** Each shard runs *every* cell but only
`n_samples` of that cell's generations, starting at `sample_start`. K shards of
`200 / K` give the full 200 for every cell (2 x 100, 4 x 50, 8 x 25, ... --
200's divisors make an even split easy, but `merge_results.py`'s coverage check
in S4.1 does not require the shards to be equal-sized, so an uneven split across
heterogeneous GPUs is fine too). Rationale: it divides the expensive
work (generation) perfectly evenly with no scheduling logic, and it duplicates
only the forward pass, which is negligible. Sharding by layer would unbalance
the work and complicate merging. Layer/cell sharding is deliberately out of scope.

Three additions:

- `--device` already exists and already flows to `load_base_model(device_map=...)`.
  Pass `cuda:0` ... `cuda:{K-1}`, one per shard process. **Bug to fix on the
  way:** `first_pass_config`
  never set `PipelineConfig.device` from `args.device`, so `config.device`
  stayed at its `"cuda"` default while the model loaded on `args.device` --
  extraction and generation would target the wrong GPU under any non-default
  `--device`. `full_sweep_config` must take and set `device`.
- `--sample-start` (default 0) and `--n-samples` (default 200): shard N runs
  generations `[sample_start, sample_start + n_samples)`.
- Per-cell seeding so shards are reproducible *and* independent:

  ```python
  def cell_seed(arm, word, layer, position, sample_start) -> int:
      # blake2b, not hash(): Python's hash() is randomized per process, so it
      # would give a different stream on every run and silently break replay.
      digest = hashlib.blake2b(
          f"{arm}|{word}|{layer}|{position}|{sample_start}".encode(), digest_size=8
      ).digest()
      return int.from_bytes(digest, "big") % (2**31)
  ```

  `torch.manual_seed(cell_seed(...))` immediately before each cell's
  `generate_interpretations()` call. Distinct `sample_start` values give
  distinct streams, so shards do not silently regenerate the same samples.

### 4.1 Output files under sharding

- Results: `results_{start:06d}_{end:06d}.json`, so shards never collide.
  New top-level shape, replacing the bare nested dict:

  ```json
  {"sample_range": [0, 100], "cells": { ...arm -> word -> layer -> position... }}
  ```

  No existing consumer to break (`run_pipeline.py` was never used for a real run).
- Hidden-state cache: path unchanged (`hidden_states/{arm}/{word}.safetensors`).
  Every shard computes byte-identical content, so concurrent writes are benign
  *provided* they are atomic -- `save_hidden_states()` must write to a temp file
  in the same directory and `os.replace()` it into place, instead of writing the
  final path directly. Without that, a reader can observe a half-written file.
- New `merge_results.py`: reads `results_*.json` from a directory, concatenates
  each cell's `generations` and `hits`, recomputes `hit_rate`, and writes
  `results.json`. It validates that the shards' `sample_range`s are disjoint and
  together cover `[0, total)` with no gaps, and errors out otherwise -- a
  silently missing shard would otherwise look like a completed run with a
  quietly smaller `n`. Pure dict logic, no heavy imports, unit-testable.

## 5. Plumbing changes

- `extract.py`: extract a shared `position_key(position) -> str` --
  `position.value` for a `Position`, `f"pos{position}"` for an int (`pos-11`,
  `pos-1`). Use it in `_tensor_key()` (`extract.py:135`) and in
  `run_pipeline.run()`'s results key (`run_pipeline.py:93`). Both currently
  call `position.value` unconditionally and crash on a raw int.
  `load_hidden_states`'s `positions` parameter widens to `list[Position | int]`.
- `config.py`: **delete `first_pass_config`**, add

  ```python
  def full_sweep_config(
      words: list[str],
      num_hidden_layers: int,
      output_dir: Path,
      n_samples: int = 200,
      sample_start: int = 0,
      device: str = "cuda",
  ) -> PipelineConfig
  ```

  using `layers_full()` and `positions=[Position.FULL_USER_SPAN]`. Add
  `sample_start: int = 0` to `PipelineConfig`. Deleting rather than keeping both
  constructors is the point of "this replaces the first pass" -- there is no
  `--sweep` selector flag, because there is only one real sweep.
- `run_pipeline.py` CLI: `--word` -> `--words` (comma-separated), plus
  `--n-samples`, `--sample-start`. `--device`/`--dtype` already exist.
- Grep for remaining `--word` / `args.word` users before implementing.
  `explore_selfie.py:34` and `explore_token_map.py` each have their own
  `--word`; those are single-word exploration scripts and **keep** `--word`.
  Only `run_pipeline.py` changes.
- `explore_token_map.py`'s docstring claims `run_pipeline.py` has a
  `--token-index` flag. It does not (only `explore_selfie.py` does). Fix the
  docstring while touching this area.
- `smoke/small_llama_config.py` keeps its two named positions -- the smoke path
  exercises plumbing, not the sweep's shape. Optionally add
  `Position.FULL_USER_SPAN` to it so the smoke run also covers the expansion
  path end-to-end; recommended, since it is one list entry.

## 6. Tests

All run locally, CPU-only, no model weights, no network. The point is that the
span algorithm is fully testable without a GPU.

`tests/test_extract.py` -- extend the existing `FakeTokenizer` with a `decode()`
backed by a token-id -> string table (`find_positions`'s tests already establish
the pattern):

1. `test_user_prompt_span_basic` -- span starts at the first content token, ends
   at -1, has the expected length.
2. `test_user_prompt_span_merged_leading_token` -- the first content token is
   `'\n\nWhat'`. The span must still include it. This is the case the
   standalone-tokenization approach gets wrong, and the reason for the lenient
   `endswith` match; without this test the leniency is untested.
3. `test_user_prompt_span_is_minimal` -- decoding the span minus its first token
   no longer ends with the prompt. Encodes "the first token is necessary"
   directly.
4. `test_user_prompt_span_identical_across_arms` -- run against a
   with-system-turn and a without-system-turn id list whose tails match; assert
   the returned negative offsets are *equal*. This is the cross-arm
   comparability invariant from S2/S3, and the single most important test here.
5. `test_user_prompt_span_raises_when_absent` -- prompt not present -> `ValueError`.
6. `test_expand_positions` -- sentinel expands in place; order preserved; an
   explicit `ASSISTANT_BOUNDARY` alongside the sentinel does not duplicate -1.
7. `test_position_key` -- `Position` -> its `.value`; `-11` -> `"pos-11"`.
8. `test_hidden_states_roundtrip_negative_positions` -- `save_hidden_states` /
   `load_hidden_states` through `tmp_path` with negative int positions (catches
   the `.value` crash and any safetensors key rejection).

`tests/test_config.py`:

9. `test_full_sweep_config` -- layers == `layers_full(N)`; `positions ==
   [Position.FULL_USER_SPAN]`; `device` and `sample_start` actually land on the
   returned config (this is the S4 bug, pinned).

New `tests/test_merge_results.py`:

10. `test_merge_concatenates_and_rescores` -- two shards merge to `n == 200`
    with a correctly recomputed `hit_rate`.
11. `test_merge_rejects_overlapping_shards` and
    `test_merge_rejects_gapped_shards` -- both raise.

New `tests/test_seeding.py`:

12. `test_cell_seed_is_stable_across_processes` -- assert against hardcoded
    expected values (a `hash()`-based implementation would fail this).
13. `test_cell_seed_differs_by_shard` -- same cell, different `sample_start` ->
    different seed.

Optional, opt-in (`@pytest.mark.skipif` on tokenizer availability): tokenize the
real chat template and assert the span decodes to exactly
`"What is the secret word?"` and is 11 long in all three arms. Tokenizer-only,
no model weights. This is the regression test for S2's findings; it is optional
only because it needs network/HF access, which the sandbox lacks.

## 7. Compute cost

With the position count now measured rather than guessed:

```
32 layers x 11 positions x 3 arms x 2 words x 200 samples = 422,400 generations
```

At 50 `max_new_tokens` each, batched 25 at a time (`interpret.py`'s default),
that is 16,896 batched generate() calls total, or `16,896 / K` per GPU across K
shards.

Time per batched call has never been measured on the real 8B model, so wall
clock is unknown. **Measure it before committing:** run a single cell
(`--n-samples 25`) on one GPU, time it, multiply by 16,896, divide by however
many GPUs are actually available. Report the number back before launching the
full sweep -- K is a launch-time input to that estimate, not a design constant.

## 8. Build order

1. `position_key()` + the `Position`-vs-int key fixes, with tests 7 and 8.
   Small, isolated, unblocks the rest.
2. `user_prompt_span()` + `expand_positions()`, with tests 1-6.
3. `full_sweep_config()`, delete `first_pass_config`, with test 9.
4. Sharding: `cell_seed()`, `PipelineConfig.sample_start`, atomic
   `save_hidden_states()`, sharded results filename, with tests 12 and 13.
5. `merge_results.py`, with tests 10 and 11.
6. `run_pipeline.py` CLI: `--words`, `--n-samples`, `--sample-start`; docstring
   fix in `explore_token_map.py`.
7. Local smoke pass (`--smoke`) end-to-end, then a 2-shard smoke run into one
   output dir followed by `merge_results.py`, to exercise the parallel path
   locally before running it on real GPUs. Two shards is enough to cover the
   merge path regardless of the K eventually used.
8. Time one real cell on the 8B model (S7) and report before the full launch.
