# Step 5: set-level retrieval recall

Part 5 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read D10, D11 and D12).

**No GPU for the code**; the unit tests run against a fake index. The real
scoring pass happens in step 6. Independent of steps 1-4, so it can be built in
parallel with any of them.

**Deliverable:** `score_sets()` in `adapter_training/retrieval_eval.py`, a
`--grouped` path in `adapter_training/evaluate_retrieval.py`, and tests.

## The metric

Agreed with the user (2026-09-05), who described it as:

> splitting the generation on ";" then tracking, for each true label topic, the
> topk value that the closest item in the set gets, and e.g. recall@N could
> report the fraction of topics in the true label set, that did actually show up
> in the generation. Some validation is needed to check whether the generation
> gives more `;` than expected; I think for now just score all of them while
> noting this as a design caveat.

and, on how much weight it carries:

> Note that I'm not too worried about the evaluation metric; it's more here to
> act as a sanity check and the OOD generalization is what I'm really after.

Read that second quote as a scope limit: this is a sanity check. Do not gold-plate
it, and do not let it grow into its own research question.

Precisely:

1. One **query** is one held-out vector. Its **true topic set** is the k topics
   its extraction prompt named.
2. Generate one description per query, then split on `;`. Strip whitespace and
   drop empty segments. The segment count is **not** forced to k.
3. Embed every segment and compute its similarity to all 49,637 topic documents.
4. For each true topic *t*, its **rank** is the *best* (lowest) rank *t* achieves
   over that query's segments.
5. *t* is **recovered at N** if that best rank <= N.
6. **recall@N** is the fraction of (query, true topic) pairs recovered at N. A
   k=3 query can score 0, 1/3, 2/3 or 1.

## The rank convention -- match it exactly

`topic_retrieval_eval.evaluate_labels` computes rank as *the number of other
topics with strictly higher similarity, plus one*, with the true topic itself
masked out. Use that identical definition. If you use a different tie-breaking
rule, the k=1 reduction check below will drift for no visible reason and cost
somebody an afternoon.

## Why `evaluate_labels` cannot be reused, and what to reuse instead

`evaluate_labels` assumes one ground-truth topic per description, and it only
keeps the top `max_k` columns of the similarity matrix. This metric needs the
rank of a *specific* topic, which may be far outside the top k. So compute
similarities directly:

- `index._embed_texts(segments)` for the embeddings (the same method
  `evaluate_labels` uses -- the single source of truth for embedding).
- `torch.mm(segment_embeddings, index.topic_embeddings.T)` for the similarities.
- Batch over segments and free each similarity block, exactly as `evaluate_labels`
  does. 49,637 topics x a few thousand segments is the only thing here that can
  exhaust memory.

Do not reimplement the embedding or the index build. `build_index()` in
`retrieval_eval.py` already exists and already avoids the reference's Hub-only
`load_dataset`.

## The k=1 reduction -- the most valuable test in this step

With k=1 and a generation that contains no `;`, `score_sets` must return
**exactly** the recall that `evaluate_labels` returns on the same inputs. That
property is what makes the k=1 slice of this experiment comparable to
`bg_think`'s existing 0.404 recall@1. Write it as a unit test against a small
fake index with fixed embeddings, and assert equality, not approximate equality.

## What to report

Per k (D12 -- never pool across k), and never a bare recall number without its
companions:

    {
      "k": 3,
      "n_queries": ...,
      "n_true_topics": ...,          # n_queries * k
      "recalls": {1: ..., 5: ..., 10: ...},
      "mrr": ...,                    # over best ranks
      "segments": {
        "mean": ...,
        "histogram": {"0": n, "1": n, "2": n, "3": n, "4+": n},
        "fraction_not_equal_k": ...
      },
      "per_query": [ ... ]
    }

**The `segments` block is not optional.** It is the whole visibility we have
into D11's known precision hole: nothing penalises extra segments, so an adapter
that emits many segments gets more attempts at each true topic and scores
higher. If `fraction_not_equal_k` is large, say so loudly in the findings note
and beside any recall number you quote. A reader who sees recall without the
segment distribution cannot tell a good adapter from a verbose one.

Also record a **per-position breakdown** as the existing `evaluate_positions`
does, and keep its `--positions` semantics (`all`, `last`, or an explicit list).
The grouped directories have the same 9-or-10 positions per group as the
single-topic ones.

## The CLI

`adapter_training/evaluate_retrieval.py` gains:

- `--grouped` (or auto-detection of `groups.json` in the extraction directory --
  pick one and be explicit about it; auto-detection is friendlier and the file
  name is unambiguous)
- the true topic set per query comes from `GroupRecord.titles`
- `--max-new-tokens` **default stays 30**, but every grouped invocation in step 6
  passes **110** (parent plan D10). Do not change the default: an old
  single-topic comparison rerun must keep producing the old numbers.

Decoding settings must be identical across every adapter compared, or the
comparison is void -- `GenerationConfig`'s docstring already says this. That
includes `max_new_tokens`: when you score the `baseline` and `bg_think`
adapters on grouped vectors for comparison, they get 110 too.

## Tests

`tests/test_set_retrieval.py`, all against a fake index with hand-chosen
embeddings so ranks are known exactly:

- the k=1 reduction to `evaluate_labels` (§ above), exact equality
- best-rank-over-segments: a query where segment 1 ranks topic A 1st and topic B
  900th, and segment 2 the reverse, recovers both at N=1
- a generation with no `;` and k=3: all three true topics scored against the one
  segment
- a generation with 7 segments and k=3: all 7 are scored, and
  `segments.histogram` records it
- empty and whitespace-only segments dropped; a generation that is only `";;;"`
  yields no segments and scores 0 without raising
- the rank convention matches `evaluate_labels` on a tie
- per-k reporting keeps the k slices separate

## Findings note

`plans/bg_think_many/notes/step5_metric.md`: confirmation that the k=1 reduction
test passes, and any surprises in how the reference index behaves. Short.
