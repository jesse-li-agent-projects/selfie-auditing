# Step 5 findings: set-level retrieval recall

Done, no GPU run needed for this step. `score_sets()` lands in
`adapter_training/retrieval_eval.py`, `--grouped` in
`adapter_training/evaluate_retrieval.py` (dispatches to a new
`evaluate_grouped_positions`, the `score_sets` counterpart to
`evaluate_positions`), and `tests/test_set_retrieval.py` (9 tests) plus 3
new `--grouped`-wiring tests in `tests/test_retrieval_eval.py`. All 246
non-`hf_cache` tests pass.

## The k=1 reduction

Confirmed exactly, against a fake index (`test_k1_no_semicolon_reduces_exactly_to_score`
and `test_rank_convention_matches_score_on_a_tie`): with k=1 and no `;` in
the generation, `score_sets`'s `recalls`/`mrr` and per-query rank are bit-
identical to `score`/`evaluate_labels` on the same inputs, including on a
tie between two non-ground-truth topics. No surprises here -- the rank
convention (count of strictly-higher-similarity *other* topics, plus one)
excludes the topic being ranked from its own count for free, since `x > x`
is never true; no explicit mask was needed to match `evaluate_labels`'
belt-and-suspenders one.

## One thing not in the step file: `load_vector_store`'s `records` argument

`--grouped` centres `store.vectors` by passing `load_group_records(args.vectors)`
(the *full*, unfiltered group list) to `load_vector_store`, not the
split-filtered query records `load_grouped_query_records` returns -- mirrors
exactly how the existing single-topic path centres against all of
`topics.json` regardless of which split is being queried. Missing this on
the first pass produced a `FileNotFoundError` on `topics.json` (a grouped
directory has no such file); caught by
`test_grouped_flag_dispatches_to_evaluate_grouped_positions`.

## Design calls made that the step file left implicit

- **`score_sets` refuses mixed `k` across queries** (`ValueError`), rather
  than silently reporting a `"k"` field that only describes some of the
  queries. Reinforces D12 (never pool across k) at the metric level, not
  just at the report-writer level.
- **A query with zero segments** (e.g. `";;;"`) leaves every one of its true
  topics' best rank as `None`: excluded from every `recall@N`'s numerator
  but still counted in `n_true_topics`, and contributes 0 to MRR. No
  exception, per the step file's requirement.
- **`--restrict-topics-to` is rejected with `--grouped`** (`parser.error`,
  same pattern `train_adapter.py`'s `--vectors-k` uses) -- a group's true
  topic set is plural, so "intersect with another directory's topics" has no
  single obvious meaning here, and the step file doesn't ask for it.

## Not yet run

The real scoring pass (against `outputs/bg_think_many_l19_k2`/`_k3` and
`baseline`/`bg_think`, all at `--max-new-tokens 110`) is step 6's job, not
this one -- this step is code plus fake-index tests only.
