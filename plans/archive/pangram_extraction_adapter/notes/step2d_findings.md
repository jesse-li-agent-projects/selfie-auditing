# Step 2d findings

Execution of `plans/pangram_step2d_retrieval_eval.md`. Built
`adapter_training/retrieval_eval.py` (the generate-then-score core) and
`adapter_training/evaluate_retrieval.py` (the CLI), plus
`tests/test_retrieval_eval.py`. Stacked on PR36 (step 2b, `worktree-
pangram-step2b-parent`), since this step needs step 2a's loaders and nothing
from the trainer.

## `resources/selfie-adapters` is gitignored, but the reference module is
## still imported, not copied

The plan is explicit that `evals/embedding_retrieval/topic_retrieval_eval.py`
must be imported, not vendored. That module is not part of the
`selfie_adapters` pip package (only `selfie_adapters/` itself is installed);
it only exists as a file under `resources/selfie-adapters/`, which
`.gitignore` excludes from the repo. `retrieval_eval.py` therefore does what
no earlier step needed to: `sys.path.insert(0, "resources/selfie-adapters")`
at import time, then `import evals.embedding_retrieval.topic_retrieval_eval`.
This assumes cwd is the repo root at runtime, the same assumption every CLI's
`outputs/`-prepending already makes.

One consequence worth flagging for whoever runs this on a fresh checkout or a
new git worktree: `git worktree add` does not copy gitignored files, so a new
worktree needs `resources/` populated by hand (a symlink to the main
checkout's copy is enough — it is read-only reference material). Neither
`pangram_step2a_loss_and_eval.md` nor `pangram_step2b_training_loop.md`
needed this, because their reads of `resources/selfie-adapters/` are all in
docstrings/comments or a hardcoded value matching a YAML
(`checkpoints.untrained_projection`), never an actual import at runtime.

## The projection-to-`Adapter` shim

`checkpoints.load_projection` and `.untrained_projection` intentionally
return a bare `nn.Module` (loss.py calls it directly:
`projection(vectors)`), but `interpret.generate_interpretations_batch` wants
an object satisfying `Adapter.transform(vector) -> vector`. Rather than
wrapping every checkpoint load in `selfie_adapters.SelfIEAdapter` (which
duplicates the metadata parsing step 2a already delegates to `load_adapter`),
`retrieval_eval._ProjectionAdapter` is a two-line shim doing exactly what
`SelfIEAdapter.transform` does when `normalize_input` isn't overridden:
`projection(vector.float()).to(vector.dtype)`.

## `--positions`, and how the per-position mean falls out of one code path

`resolve_position_offsets` turns `"all"` into `range(max_count)` and a
comma-separated list into itself; both then go through the same per-offset
generate-and-score loop, averaged at the end. `"last"` is handled separately
because it is a genuinely different operation — each topic's *own* last
vector (`start + count - 1`), not a fixed offset, since `count` is 9 or 10
depending on which forced variant a topic matched (step 1, S9.2). This means
a one-vector-per-topic directory (baseline, arm C pooled) needs no special
case at all: every record has `count == 1`, so `"all"` and any offset list
degenerate to `offset == 0` and `"last"` also lands on the same single
vector — `--positions` really is ignored there, exactly as the plan says,
with nothing coded to make that true.

## Generation is not reproducible across different `batch_size`s (a real
## finding, not a bug)

The plan's test 7 asked for `generate_interpretations_batch` output to be
reproducible "across two calls with different batch sizes" given a fixed
seed. Running it under `gpu-exec` against Llama-3.2-1B disproved that: rows
before the first chunk boundary matched, rows after it diverged. The cause
is structural, not a bug to fix here: `do_sample=True` draws from the
process-global `torch` RNG stream, and `generate_interpretations_batch`
chunks rows into separate `model.generate()` calls at `batch_size`
boundaries (`interpret.py`) — a row's position in that shared stream depends
on how many other rows' decode steps were drawn before it, which depends on
the chunking, not merely on `config.seed`. The property that actually holds,
and the one this eval needs, is reproducibility for a *fixed* `batch_size`
(test rewritten accordingly, `test_generation_reproducible_for_a_fixed_batch_size`)
— which is what every call in this codebase gets, since `batch_size` is
`interpret.py`'s own default and is never varied between runs here.

## Deviations from the plan's literal sketch

All additive:

- `evaluate_positions` is one function covering both the `"last"` case and
  the `"all"`/list case, returning `{"mode": "last", ...}` or `{"mode":
  "per_position", "per_position": {...}, "recalls": <mean>, "best_position":
  ...}`. The plan's pseudocode showed `score()` as a single-pass function;
  the per-position averaging for arm B needed a caller above it, which is
  this function rather than inline in the CLI, so it stays unit-testable
  without a model.
- `evaluate_retrieval.py`'s `--index-cache` is additive (not in the plan's
  CLI sketch): it calls the reference's own `build_or_load_index`, which is
  what makes the "build once, reuse across arms" cost story in the plan's
  §"Cost" section actually happen without every run rebuilding the 49,637-
  document index.
- `IndexStrategy.TITLE_PLUS_ALL_LABELS` and `thenlper/gte-large` are not CLI
  flags for the strategy (embedding model is, for testing against a smaller
  model); the plan calls out setting the strategy *explicitly* to avoid a
  silent comparison across index strategies, so it is a constant
  (`retrieval_eval.DEFAULT_INDEX_STRATEGY`), recorded in every report,
  rather than something a flag could accidentally change.

## Testing notes

- All 6 fast tests pass locally, no model, no GPU, no `sentence_transformers`
  call — `score` and `build_index` are exercised against a fake
  `SentenceTransformer` / a hand-built fake index, since the reference's
  `TopicRetrievalIndex.__init__` loads a real embedding model.
- Full repo suite (`pytest`, fast-only): 125 passed, 1 pre-existing failure
  unrelated to this step (`test_extract_pangram_vectors.py::
  test_pangram_prompt_is_the_requested_wording`, flagged in step 2a's and
  step 2b's findings notes, not touched here).
- `python -m adapter_training.evaluate_retrieval --help`: ~33ms, no torch
  import.
- The `hf_cache`-marked determinism test passes under `gpu-exec`
  (`HF_HUB_OFFLINE=1`, ~14s against the cached Llama-3.2-1B) — see the
  finding above for why it tests same-`batch_size` reproducibility rather
  than the plan's literal "different batch sizes" framing.

## Done-when checklist

- [x] `pytest tests/test_retrieval_eval.py` passes (fast tests).
- [x] The `hf_cache`-marked test passes under `gpu-exec`.
- [x] `python -m adapter_training.evaluate_retrieval --help` returns without
      importing torch.
- [x] The preflight names `sentence_transformers` before any generation.
- [x] This note.
- [ ] Committed on a worktree branch with an undrafted PR — next.
