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

Also do **not** anchor the search on `Position.LAST_CONTENT_TOKEN`. A named
position is itself a bet that the current formatted prompt never changes:
`LAST_CONTENT_TOKEN` is defined as `last <|eot_id|> - 1`, so it only equals "the
last token of the user's question" while the user turn ends immediately after
that question. Add anything to the user turn -- a suffix, a tool block, a
template revision -- and the anchor moves silently, taking the span with it.
Deriving the span from a named position would inherit that assumption wholesale.

Instead, locate the prompt by its own text, scanning the formatted prompt's own
tokens in both directions and anchoring on nothing:

```
n = len(ids)
# End: last token index at which the decoded prefix ends with the prompt.
# Scanning downward takes the *last* occurrence, which is the user turn's copy
# even when an arm's system prompt happens to quote the same text.
for end in range(n - 1, -1, -1):
    if tokenizer.decode(ids[: end + 1]).endswith(user_prompt):
        break
else:
    raise ValueError(...)

# Start: smallest slice ending at `end` that still ends with the prompt.
for start in range(end, -1, -1):
    if tokenizer.decode(ids[start : end + 1]).endswith(user_prompt):
        return list(range(start - n, 0))   # negative offsets, start .. -1
raise ValueError(...)
```

The only structural fact this now relies on is that the span runs to the end of
the formatted prompt, which is the definition of the requested sweep, not a
template assumption. Both named positions remain available for other callers
(`explore_selfie.py`, the smoke config); this sweep just does not depend on them.

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
) -> list[int]:
```

No `pos_index` parameter, and no `find_positions()` call -- that absence is the
point of the paragraph above.

Cost is at most one decode per token in each of two downward scans (~50 decodes
for this prompt), called once per (arm, word). Irrelevant next to the forward pass.

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
`[extract] ...` print loop, so every run's own log shows each selected token
decoded -- the verification stays in the logs rather than in a one-off script.
(11 is what today's template yields, per S2; nothing in the code should hardcode
it.)

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
  {"sample_range": [0, 100],
   "secret_prompt": "What is the secret word?",
   "spans": {"control": {"-11": "What", "-10": " is", "...": "..."}},
   "cells": { ...arm -> word -> layer -> position... }}
  ```

  `secret_prompt` and `spans` make the results **self-describing**: a stored
  offset like `-11` only means `'What'` relative to a particular formatted
  prompt, so recording the decoded token each offset actually resolved to keeps
  old result files interpretable after any prompt or template change, and makes
  two runs' comparability checkable instead of assumed. Same reasoning as
  dropping the named-position anchor in S3.1 -- do not let the current template
  become an unstated premise of the data.

  `merge_results.py` must refuse to merge shards whose `secret_prompt` or
  `spans` disagree; that mismatch means the shards are not measuring the same
  thing, and concatenating them would produce a silently meaningless cell.

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

Two tiers. Tier 1 (tests 1-13) is CPU-only, no model weights, no network -- the
span algorithm is fully testable without a GPU, and that is deliberate. Tier 2
(tests 14-16, S6.1) uses the local 8GB GPU and the 1B smoke model to check the
things a fake tokenizer cannot.

### Tier 1: pure logic, CPU only

`tests/test_extract.py` -- extend the existing `FakeTokenizer` with a `decode()`
backed by a token-id -> string table (`find_positions`'s tests already establish
the pattern):

1. `test_user_prompt_span_basic` -- span starts at the first content token and
   ends at -1. Assert against offsets derived from the fixture's own id list,
   not a hardcoded 11: 11 is a property of today's template (S2), and a test
   that hardcodes it is making the same never-changes assumption S3.1 removes.
   The one place hardcoding 11 is correct is the S6.2 preflight regression test,
   where pinning the measured value against template drift *is* the purpose.
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
5a. `test_user_prompt_span_survives_trailing_user_turn_tokens` -- id list where
   the user turn continues *after* the question (extra tokens before its
   `<|eot_id|>`). The span must still start at the question's first token. This
   is the case that a `LAST_CONTENT_TOKEN`-anchored implementation gets wrong,
   so it is the direct regression test for S3.1's anchor removal -- without it,
   reintroducing the anchor would pass every other test here.
5b. `test_user_prompt_span_takes_last_occurrence` -- a system prompt that quotes
   the question verbatim, followed by the real user turn. The span must land on
   the user turn's copy, mirroring how `find_positions()` already takes the last
   `<|eot_id|>` for the same reason.
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
11a. `test_merge_rejects_mismatched_spans` -- two shards recording different
    `secret_prompt`/`spans` raise rather than concatenate.

New `tests/test_seeding.py`:

12. `test_cell_seed_is_stable_across_processes` -- assert against hardcoded
    expected values (a `hash()`-based implementation would fail this).
13. `test_cell_seed_differs_by_shard` -- same cell, different `sample_start` ->
    different seed.

### 6.1 Tier 2: GPU-backed, local

Hardware confirmed present: one NVIDIA RTX PRO 1000 Blackwell Laptop GPU,
8151 MiB. Llama-3.2-1B-Instruct in bf16 is ~2.5 GB of weights, so the smoke
model fits with room to spare -- including **two concurrent shard processes**
(~5 GB total), which is what makes test 16 possible on one card.

Mark these `@pytest.mark.gpu` and default to skipping them, so a plain `pytest`
stays fast and CPU-only.

**Dummy adapters already exist -- use the strongest one.** `smoke/small_llama_config.py`
provides three stand-ins, in increasing fidelity:

- `IdentityAdapter` -- passes the vector through unchanged.
- `RandomAffineAdapter` -- fixed random affine, in-memory only.
- `create_random_selfie_adapter()` -- writes a real random-weight checkpoint in
  the actual `selfie_adapters` safetensors format, metadata header included, so
  it loads through the ordinary `load_adapter()` -> `SelfIEAdapter.transform()`
  path. `make_smoke_weights.py` is the CLI that generates it alongside a random
  taboo LoRA.

Worth noting: `run_pipeline.py --smoke` currently hardcodes `IdentityAdapter()`,
the weakest of the three, even though the real-format checkpoint exists and
`explore_selfie.py` already consumes it via `--adapter-path`. So the smoke
*pipeline* path never exercises the real adapter loader, dimension check, or
projection math -- the one part of tier 2 that could catch a real adapter-side
break. Switching `--smoke` to `create_random_selfie_adapter()` is a small change
and makes tier 2 meaningfully stronger; recommended, and cheap enough to fold
into build-order step 8. None of these adapters can say whether the sweep
*finds* anything -- they are untrained by construction.

14. `test_span_reads_the_intended_tokens` -- real tokenizer, real 1B model.
    Extract with `Position.FULL_USER_SPAN` and assert (a) the decoded token at
    each returned offset matches the expected string, and (b) the hidden state
    at offset `-k` is bitwise identical to the one at absolute index
    `len(ids) - k`. Confirms negative indexing addresses what S3 claims it does,
    end to end through the real forward pass rather than through a fake.
15. `test_span_identical_across_arms_real_template` -- the real-tokenizer
    counterpart to test 4, which only proves the algorithm is consistent on a
    hand-built id list. This one proves the *actual* chat template produces
    equal offsets for all three arms. Between them, test 4 catches an algorithm
    regression and test 15 catches a template change.
16. `test_two_shards_produce_different_generations` -- run the smoke config
    twice, `--sample-start 0` and `--sample-start n`, merge, assert the merged
    `n` is the sum **and that the two shards' generation lists are not equal**.
    This is the one that matters: if `cell_seed` were ignored, or both shards
    seeded identically, every shard would regenerate the same samples and a
    "200-sample" cell would really be 100 samples counted twice. Nothing in
    tier 1 can catch that, and the merged output would look perfectly healthy.

**Run tier 2 as the normal user, not via `gpu-exec`.** The `claude` account's HF
cache holds only `keenanpepper/selfie-adapters-...`, has no network egress, and
cannot be topped up from the agent sandbox: `meta-llama/Llama-3.2-1B-Instruct`
is a gated repo and the sandbox has no HF token (confirmed -- 401 GatedRepoError,
no `HF_*` environment variable set). The normal user's cache already has the 1B
model from the earlier smoke work, and that user has the GPU directly, so this
is a non-issue as long as tier 2 is not routed through `gpu-exec`. Do not
silently skip tier 2 because the marker made it easy to.

If a future agent-run tier 2 is ever wanted, the two options are copying the
model into claude's cache from outside the sandbox, or pointing `SMOKE_MODEL` at
an ungated mirror (the same trick S2's preflight used for the 8B tokenizer).
Neither is needed now.

**Marker registration:** the repo has no `pytest.ini`/`pyproject.toml`, so a
bare `@pytest.mark.gpu` raises `PytestUnknownMarkWarning`. Add a minimal
`pytest.ini` registering the marker rather than letting a warning ride.

### 6.2 Optional: real tokenizer, no GPU

`@pytest.mark.skipif` on tokenizer availability: tokenize the real chat template
and assert the span decodes to exactly `"What is the secret word?"` and is 11
long in all three arms. Tokenizer-only, no model weights, no GPU. The regression
test for S2's findings; optional only because it needs HF access to a gated repo.

## 7. Compute cost

With the position count measured (S2) rather than guessed -- but recompute it
from the run's own preflight print if the prompt or template ever changes,
rather than reusing this number:

```
32 layers x 11 positions x 3 arms x 2 words x 200 samples = 422,400 generations
```

At 50 `max_new_tokens` each, batched 25 at a time (`interpret.py`'s default),
that is 16,896 batched generate() calls total, or `16,896 / K` per GPU across K
shards.

Per-generation timing was already confirmed acceptable in earlier ad-hoc runs,
so **no timing preflight is needed** -- this section is a scale reference, not a
gate. The count matters only for choosing K and for noticing if a template
change moves the position count.

The sweep itself still needs the Vast.ai box: Llama-3.1-8B-Instruct in bf16 is
~16 GB of weights, so it does not load on the 8151 MiB local card at all. Do not
substitute a quantized 8B to make it fit locally -- quantization perturbs
exactly the hidden states this experiment reads. The local GPU's role is S6.1's
tier-2 tests on the 1B smoke model, not 8B work.

## 8. Build order

1. `pytest.ini` registering the `gpu` marker (S6.1) -- first, so tier-2 tests
   can be written as they are earned instead of retrofitted.
2. `position_key()` + the `Position`-vs-int key fixes, with tests 7 and 8.
   Small, isolated, unblocks the rest.
3. `user_prompt_span()` + `expand_positions()`, with tests 1-6 (including 5a/5b,
   which pin the anchor-free behaviour).
4. `full_sweep_config()`, delete `first_pass_config`, with test 9.
5. Sharding: `cell_seed()`, `PipelineConfig.sample_start`, atomic
   `save_hidden_states()`, sharded results filename + `secret_prompt`/`spans`
   metadata, with tests 12 and 13.
6. `merge_results.py`, with tests 10, 11 and 11a.
7. `run_pipeline.py` CLI: `--words`, `--n-samples`, `--sample-start`; docstring
   fix in `explore_token_map.py`.
8. Local GPU, 1B smoke model: tier-2 tests 14-16 (S6.1), run as the normal user
   rather than via `gpu-exec`. Fold in the `--smoke` adapter upgrade
   (`IdentityAdapter` -> `create_random_selfie_adapter()`) noted in S6.1.
9. Local smoke pass (`--smoke`) end-to-end, then a 2-shard smoke run into one
   output dir followed by `merge_results.py`, to exercise the parallel path
   locally before running it on real GPUs. Both shards fit concurrently on the
   8GB card at 1B scale. Two shards is enough to cover the merge path
   regardless of the K eventually used.
10. Launch on the Vast.ai box (not locally -- S7). No timing preflight: already
    confirmed acceptable in earlier runs.
