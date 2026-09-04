# Step 2d: generation and the embedding-retrieval eval

Part 4 of 7 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §5.3, §5.6 and D11 before starting).

**Depends on** `plans/pangram_step2a_loss_and_eval.md` -- it needs `dataset.load_vector_store`
and `checkpoints.load_projection`. It does **not** need the trainer, so it can be built and
tested alongside `plans/pangram_step2b_training_loop.md`.

**Deliverable**: `adapter_training/retrieval_eval.py` and a CLI that takes any projection
checkpoint plus an extraction directory, generates one description per held-out vector, and
reports recall@k against a GTE-large topic index.

**This produces the experiment's headline number.** Validation loss says how well an adapter
fits its own labels; it does not say whether a human reading the generated text would
recover the topic, and losses from different extraction prompts are not on a common scale
(§5.4). Recall@k is on a common scale: every arm queries the same index, and the random
floor is 1/49,637 for all of them.

**No GPU work here.** Everything is written and unit-tested locally against fakes or
Llama-3.2-1B (`config.DUMMY_BASE_MODEL`). The real runs happen in
`plans/pangram_phase0_run.md`.

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

## The metric, quoted from the paper

> For contrastive vectors, we evaluate via embedding-based retrieval. Each topic is
> represented as a document embedding (title plus all labels) using GTE-large (Li et al.,
> 2023). We embed each generated description and report recall@k, the fraction of topics
> for which the correct topic appears among the top-k nearest neighbors.

The paper's abstract gives the scale to read our numbers against: adapters "identify topics
with **94% recall@1** versus **1% for untrained baselines**". Those are arm-A-style numbers
-- a topic the extraction prompt named out loud -- so they are **reference points, not
targets** for arm B, which is asked to recover a topic the response never says (§5.4).

**That 94% was measured on contrastive vectors**, as the sentence above says and as the
paper's Figure 1 shows ("extract h from layer l, subtract mean h over all topics"). Raw
injection at eval time belongs to the bridge-entity/TwoHopFact experiment and to
`interpret.py`, not here (§5.3). This eval therefore defaults to **centred** vectors.

## What the reference already provides

`resources/selfie-adapters/evals/embedding_retrieval/topic_retrieval_eval.py` (read-only
reference; import it, do not copy it) has the whole scoring half:

| name | role |
|---|---|
| `TopicRetrievalConfig` | embedding model (`thenlper/gte-large`), batch size, normalisation, index strategy |
| `IndexStrategy` | `TITLE_ONLY`, `TITLE_PLUS_FIRST_LABEL`, `TITLE_PLUS_ALL_LABELS`, `MEAN_OF_ALL` |
| `TopicRetrievalIndex` | loads topics, embeds documents, holds `topic_embeddings` |
| `evaluate_labels` | one matmul against the whole index; recall@k, ranks, MRR, margins |
| `print_eval_summary` | the human-readable dump |

Two places it does not fit as-is, both small:

- **`TopicRetrievalIndex.load_dataset` calls `load_dataset` against the Hub**, and the vast
  remote's `agent` account has no egress. Do not work around this with a download: build the
  index from `extract_common.load_topics`, which already reads either the Hub or a local
  JSONL copy and returns the same `original_title` / `labels` fields the index wants. Pass
  the same `--dataset-file` the extractors were given.
- **The config's default `index_strategy` is `TITLE_ONLY`, but the paper used title plus all
  labels.** Set `IndexStrategy.TITLE_PLUS_ALL_LABELS` explicitly and record it in the report,
  so a number is never compared against one built with a different index.

`sentence_transformers` is assumed present in the run environment. It cannot be installed on
the egress-free remote, so **check the import in a preflight before any GPU time is spent**
and fail loudly with the package name if it is missing, rather than after generation.

## Build

### 1. `adapter_training/retrieval_eval.py`

```python
@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings; identical across arms or the comparison is void."""
    max_new_tokens: int = 30
    temperature: float = 0.7
    n_samples: int = 1
    seed: int = 42

def build_index(topics, *, strategy, embedding_model, device) -> TopicRetrievalIndex: ...
def generate_descriptions(model, tokenizer, projection, vectors, config) -> list[str]: ...
def score(index, descriptions, ground_truth_titles, k_values) -> dict: ...
```

- **Generation reuses `interpret.generate_interpretations_batch`**, which already injects
  into `interpret.SELFIE_TEMPLATE` at both slots, batches across cells, and strips the
  trailing quote. Do not write a second injection path -- a template or slot difference
  between the loss path and the generation path is exactly the bug that would make the two
  halves of the write-up disagree.
- It always samples (`do_sample=True`, upstream's `generate_descriptions` defaults are the
  same: 30 new tokens, temperature 0.7). So **fix the seed and use one sample per vector**,
  and hold every setting identical across arms and conditions. Record them in the report.
- **The index covers all 49,637 topics; the queries are val topics only.** A bigger index is
  a harder, fixed task, and it is the same task for every arm. Restrict the query set to the
  val topics that survived the pangram filter, for **every** arm including arm A, so a recall
  difference cannot be a topic-population difference (`dataset.restrict_to_titles`).
- **Centring is a flag with the same two meanings as everywhere else** (§5.3): `--center`
  (default) is the paper-comparable, matches-training condition; `--no-center` is the
  downstream deployment condition the taboo pipeline will actually be in. Print which one
  ran.

### 2. `adapter_training/evaluate_retrieval.py` (CLI)

```
python -m adapter_training.evaluate_retrieval \
    --vectors vectors/pangram_l19 --split val \
    --checkpoint runs/phase0_armB/best.pt \
    --dataset-file <jsonl> --center \
    --positions all --report eval/armB_retrieval.json
```

- `--checkpoint untrained` selects `checkpoints.untrained_projection` -- the floor, and the
  thing the paper's 1% corresponds to.
- `--positions` takes `all`, `last`, or a comma-separated list. For a one-vector-per-topic
  directory (baseline, or arm C pooled) it is ignored.
- `--limit-topics N` with `--seed` subsamples the query set, for a cheaper first pass.
- `--vectors` and `--report` are under `outputs/` implicitly, matching the extractors.
- Light imports first, `args = parse_args()` before `import torch`, per the project CLI
  convention.

**Arm B is scored at every position, and the primary number is the mean over positions.**
Arm B trains on all 10 positions as equal examples, so the mean is what its training
objective corresponds to; report the per-position vector and the best position beside it.
This retires the separate per-position exploration §5.6 used to schedule -- one eval yields
both -- but keep reading the per-position numbers as exploratory, not as a result.

Report JSON: recall@{1,5,10}, MRR, per-position breakdown, n queries, index size, index
strategy, embedding model, centring mode, the full `GenerationConfig`, checkpoint metadata,
vector directory, model, layer.

## Cost

Well under the training run, which is the point of scheduling it as a headline. For ~4,964
val topics on a 24 GB Ampere card:

| component | scale | note |
|---|---|---|
| index build | 49,637 documents through GTE-large (335M) | once, cached to disk and reused by every arm |
| generation, one position | 4,964 × 30 new tokens, batched | decode-bound, not FLOP-bound |
| generation, arm B all 10 positions | 10× the above | the per-position breakdown falls out of it |
| scoring | one matmul, queries × 49,637 | negligible |

Measure it in `plans/pangram_step0_benchmarks.md` and re-derive; if all 10 positions prove
slower than expected, drop to `--limit-topics` on the query set rather than thinning
positions, and say which you did.

## Tests -- `tests/test_retrieval_eval.py`

Fast (no marker), against fakes:

1. `score` computes recall@k correctly on hand-built embeddings where the ranking is known
   by construction, including the case where the correct topic is exactly at rank k.
2. The index built from `load_topics` on a small local JSONL has one document per topic, in
   topic order, and `TITLE_PLUS_ALL_LABELS` formatting matches
   `topic_retrieval_eval.format_topic_document` for the same input.
3. The query set is restricted to val topics of the requested split, and
   `restrict_to_titles` gives every arm the identical query set from two different extraction
   directories.
4. `--positions last` on a directory with mixed per-topic `count` (10 and 9) selects each
   topic's own last vector -- index `start + count - 1`, not `start + 9`.
5. Centring mode reaches `load_vector_store` and is recorded in the report.
6. The preflight raises a named error when `sentence_transformers` is absent.

`hf_cache`-marked, against Llama-3.2-1B (run under `gpu-exec`; the HF cache is only readable
by the `claude` user):

7. Generation through `generate_interpretations_batch` with a fixed seed is reproducible
   across two calls with different batch sizes.

## Done when

- `pytest tests/test_retrieval_eval.py` passes, and the `hf_cache` test passes under
  `gpu-exec`.
- `python -m adapter_training.evaluate_retrieval --help` returns without importing torch.
- The preflight names `sentence_transformers` and `thenlper/gte-large` before any generation.
- Committed on a worktree branch with an undrafted PR.

## Do not

- Do not reimplement recall@k, the index, or the document formatting -- the reference's
  `topic_retrieval_eval.py` is correct and is what the paper's numbers came from.
- Do not vendor or edit `resources/selfie-adapters/`.
- Do not write a second soft-token injection path; reuse `interpret.py`'s.
- Do not compare a recall number against one built with a different index strategy,
  embedding model, query set, or decoding settings.
- Do not treat the paper's 94% as a target for arm B. It is a reference point measured on a
  different task (§5.4).
