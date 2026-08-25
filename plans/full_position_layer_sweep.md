# Plan: CLI/config support for a full layer x token-position sweep

Status: not started -- design only, nothing implemented.

## 1. Goal / research question

This plan implements a specific sweep the user specified directly (not inferred):

- 2 pre-specified Taboo words
- every layer (not the every-4th-layer subset `first_pass_config` currently uses)
- every user-prompt token position, from the "What" token in "What is the secret
  word?" through the end of the formatted prompt template (not just the two named
  positions `selfie_taboo_pipeline.md` S4.4 currently scopes)
- all 3 arms (control, prompted, fine-tuned)
- 200 SelfIE generations per (arm, word, layer, position) cell

This maps to README.md "Research questions" 1 and 2 ("Will a SelfIE adapter
correctly uncover something the model is actively hiding?" / "How does
performance differ if the model is control / prompted / fine-tuned...") at finer
layer/position resolution than the existing pipeline plan's first pass.

**Open question, flagged not assumed:** `selfie_taboo_pipeline.md` S4.4
deliberately scoped generation to two named positions per layer ("only spend the
generation budget on two position candidates per layer") and treats the full
every-4th-layer sweep as the smoke-scale default, full-layer as a named but
unused alternative. This plan is a scope expansion beyond that section, not a
continuation of it. Confirm with the user whether this full sweep supersedes
S4.4's two-position budget decision or is a separate, deliberately more
expensive follow-on run -- the cost estimate in S5 below matters for that call.

## 2. Gaps in the current pipeline

`run_pipeline.py` and `config.py` today only support the single-word first pass
(`config.py:102` `first_pass_config`). Four gaps block the requested sweep:

1. **Single word only.** `run_pipeline.py:26` CLI arg is `--word` (singular);
   `first_pass_config(word: str, ...)` builds `words=[word]`. Per the user's
   comment, **`--words` replaces `--word` outright** -- no need to keep both.
2. **Layer subset only.** `first_pass_config` calls `layers_smoke()`
   (`config.py:68`, every 4th layer). `layers_full()` (`config.py:73`, every
   layer) already exists but nothing wires it up.
3. **Fixed sample count.** `n_samples=100` is hardcoded in
   `first_pass_config` (`config.py:115`); not exposed via CLI.
4. **No full-token-span position support.** `config.Position` (`config.py:27`)
   has exactly two named members, `ASSISTANT_BOUNDARY` and `LAST_CONTENT_TOKEN`.
   `extract.resolve_position()` (`extract.py:75`) already passes a raw `int`
   token index straight through, bypassing the enum, so the extraction forward
   pass itself can already address any token position. But two things assume
   `Position` enum, not `int`, and will break on a raw index:
   - `extract._tensor_key()` (`extract.py:135`) calls `position.value` when
     building the safetensors cache key.
   - `run_pipeline.run()` (`run_pipeline.py:93`) also calls `position.value`
     when building the flat results dict key.

   Nothing today computes the token span itself ("What" through end of
   template) -- see S3 below.

## 3. Design: locating the full user-prompt token span

Need a token-index range `[start, len(ids) - 1]` where `start` is the first
token of the user's literal content ("What...") and the end is the existing
`ASSISTANT_BOUNDARY` index (`extract.find_positions()`, `extract.py:70`,
`len(ids) - 1`).

`find_positions()` already locates the *end* of user content
(`LAST_CONTENT_TOKEN = eot_positions[-1] - 1`) robustly, by anchoring on the
last `<|eot_id|>` rather than a fixed offset, because CONTROL/PROMPTED prepend
a system turn with its own `<|eot_id|>` (`extract.py:56-59`). The *start* of
user content needs the same care -- a fixed offset from the front of the
sequence is wrong whenever a system turn is present.

Two candidate approaches:

- **(a) Token-count-back from the known end.** Tokenize `config.secret_prompt`
  standalone (`tokenizer(secret_prompt, add_special_tokens=False)`), count its
  tokens `k`, and take `start = LAST_CONTENT_TOKEN - k + 1`. Simple, but
  assumes the standalone tokenization of the content string is token-identical
  to its tokenization in the full templated context (preceded by whatever
  whitespace/newlines the template inserts after the user-role header). This
  is usually true for BPE across a whitespace boundary but per this project's
  "confirm, don't assume" norm (see `selfie_taboo_pipeline.md` S2, S4.4), it
  needs a one-time preflight check against the actual formatted prompt, not an
  assumption baked into the code.
- **(b) Anchor on the user-role header tokens.** Locate the token sequence for
  `<|start_header_id|>user<|end_header_id|>` (tokenize that substring once,
  match it in `ids` -- it's the *last* such header before the final
  `<|eot_id|>`, mirroring how `LAST_CONTENT_TOKEN` already anchors on the last
  `<|eot_id|>`), then skip forward past whatever whitespace tokens the
  template inserts before content starts.

Recommendation: (a), with the preflight check made explicit -- print the
decoded token span the same way `extract_hidden_states()` already prints
`[extract] {label} -> token {idx}: {token_str!r}` for named positions
(`extract.py:108-112`), so a run's own logs make the chosen span
inspectable/verifiable rather than trusted blindly. This reuses
`find_positions()`'s existing anchoring instead of adding a second,
differently-anchored mechanism.

This becomes a new helper, e.g. `extract.user_prompt_span(tokenizer,
input_ids, pos_index) -> list[int]`, returning raw token indices, called once
per (arm, word) alongside `find_positions()`.

## 4. Plumbing changes

- `extract._tensor_key()`, `save_hidden_states`/`load_hidden_states`, and
  `run_pipeline.run()`'s key construction (`run_pipeline.py:93`): branch on
  `isinstance(position, Position)` the same way the existing print loop in
  `extract_hidden_states()` already does (`extract.py:110`), so a raw `int`
  position gets a stable cache/result key (e.g. `f"layer_{layer}__pos{position}"`)
  instead of crashing on a missing `.value`.
- `config.py`: new constructor alongside `first_pass_config`, e.g.
  `full_sweep_config(words: list[str], num_hidden_layers: int, output_dir: Path,
  n_samples: int = 200) -> PipelineConfig`, using `layers_full()` for layers and
  a positions list built from the new span helper (S3) plus the two existing
  named positions, if both are wanted. Confirm with the user whether the named
  positions should be included alongside the full span or replaced by it --
  the two named positions are a subset of the full span in one case
  (`LAST_CONTENT_TOKEN`, the last token of the question) but not the other
  (`ASSISTANT_BOUNDARY` sits after the template's own trailing tokens, past the
  literal user content, so it's already the span's own end token -- no overlap
  issue there).
- `run_pipeline.py` CLI: replace `--word` with `--words` (comma-separated,
  parsed into `list[str]`), matching the user's explicit instruction that
  `--words` replaces `--word` rather than adding alongside it. Add flags to
  select the sweep config, e.g. `--sweep {first-pass,full}` dispatching to
  `first_pass_config` vs. `full_sweep_config`, plus `--n-samples` (default
  stays whatever the chosen config sets, override optional).
- Update every other caller of `--word` / `args.word` (tests, docs, any smoke
  path) for the rename -- grep before implementing to enumerate them.

## 5. Compute cost -- recompute before running

`selfie_taboo_pipeline.md` S7 estimated the single-word, every-4th-layer,
2-position, 100-sample first pass at ~4,800 generations, and the same
single-word full-layer version at ~19,200 (`N=32`). This sweep multiplies that
by 2 words, a full token-position span instead of 2 fixed positions, and 2x the
samples per cell:

- Position count depends on the tokenized length of `config.SECRET_PROMPT`
  ("What is the secret word?") -- likely on the order of 7-9 tokens with the
  Llama-3.1 tokenizer, to be confirmed by the S3 preflight print, not assumed.
- Rough order of magnitude at N=32 layers, ~8 positions, 3 arms, 2 words, 200
  samples: `32 x 8 x 3 x 2 x 200` ~= 307,200 generations, roughly 15-60x
  `selfie_taboo_pipeline.md`'s already-estimated single-word full-layer pass,
  depending on final position count.
- This is a real budget/time question, not just a plumbing one -- flag it back
  to the user once the position count is confirmed, before running for real.

## 6. Build order

1. Preflight: tokenize `config.SECRET_PROMPT` under the real chat template for
   one CONTROL-arm formatted prompt, confirm the token count and decoded span
   look right (S3), and get a real position count to refine the S5 cost
   estimate.
2. Implement the `Position`-vs-`int` key-construction fix (S4, first item) --
   small, testable in isolation, unblocks everything else.
3. Implement `extract.user_prompt_span()` and its unit tests (mirroring the
   existing `find_positions()` tests, using the `TokenizerLike` protocol
   already in place for testability without loading a real tokenizer).
4. Implement `config.full_sweep_config()`.
5. Update `run_pipeline.py` CLI: `--word` -> `--words`, add sweep-selection and
   `--n-samples` flags.
6. Re-run the local smoke pass (`--smoke`) to confirm the rename and new code
   paths don't break the existing end-to-end smoke coverage.
7. Confirm the S5 cost estimate with the user before running against the real
   8B model.
