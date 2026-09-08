# Step 1 findings: label-complexity bucketing

**Decision: rule A (within-topic terciles), via the "below 0.15" branch.**
Cross-topic overlap probability = **0.0139**, bare-title check clean (0.29%
failure), no empty or singleton buckets. `adapter_training/label_complexity.py`
implements it; `tests/test_label_complexity.py` covers the required
properties.

Data: `outputs/baseline_l19/topics.json`, 49,637 topics, all labels.

**Environment note.** `config.BASE_MODEL_8B`
(`meta-llama/Llama-3.1-8B-Instruct`) is not in the local HF cache — only a
stale `.locks/` entry, no snapshot, which looks like an interrupted download
rather than an intentional gap. Its tokenizer is identical across LoRA
fine-tunes of the same base, so this analysis loaded the tokenizer from the
locally-cached `bcywinski/llama-3.1-8b-instruct-taboo-gold` snapshot instead
(tokenizer-only; no model weights touched). Flagging so the base-model cache
gap gets fixed rather than silently worked around again in step 2/6.

## The five measurements (rule A)

1. **Bucket sizes.** 0 / 49,637 topics have an empty or singleton bucket.
   Sizes are near-even by construction (6-label topics: 2/2/2; most common
   patterns are (6,6,5), (6,6,6), (6,5,5), consistent with the 6-20 labels/topic
   range).

2. **Bucket boundaries across topics** (mean tokens per bucket, per topic):

   | bucket | mean | stdev | p10 | p50 | p90 |
   |---|---|---|---|---|---|
   | 0 | 7.98 | 1.61 | 6.0 | 7.8 | 10.2 |
   | 1 | 11.13 | 1.92 | 8.7 | 11.0 | 13.7 |
   | 2 | 15.22 | 2.41 | 12.3 | 15.0 | 18.4 |

   The three distributions are well separated (each p90 sits below the next
   bucket's p10 only loosely, which is exactly what measurement 3 quantifies).

3. **Cross-topic overlap probability (headline number): 0.0139.**
   Computed exactly (sorted merge over 296,659 bucket-0 lengths and 262,891
   bucket-2 lengths, not sampled). Well under the 0.15 "ship it" threshold.

4. **Bare-title check.** 14,294 / 49,637 topics carry their bare title as a
   label. Of those, only 41 (0.29%) do *not* land the title in bucket 0 —
   clean.

5. **Read-it-yourself samples.** Within-topic triples read as intended
   (bucket 0 clearly the least detailed, bucket 2 clearly the most):

   - *Mamluk*: `Mamluks, slave soldiers in Islamic societies` (0) → `the
     enslaved military caste prominent in Egypt and the Levant` (1) →
     `Mamluks, military slaves who became powerful warriors and rulers in
     medieval Islam` (2)
   - *Discrete uniform distribution*: `discrete uniform distribution` (0) →
     `the probability distribution with equally likely finite outcomes` (1) →
     `discrete uniform distribution, a probability distribution giving equal
     weight to finite discrete values` (2)
   - *Lauren Jackson*: `three-time WNBA MVP from Australia` (0) → `the
     Australian forward who dominated the WNBA` (1) → `Lauren Jackson,
     Australian basketball player and WNBA star for the Seattle Storm` (2)

   Cross-topic tuples (bucket *i* from 3 random topics, the shape step 3
   builds) read as comparable, not mismatched:

   - `[Microsoft Excel] bucket 0: "Excel as part of the Microsoft Office
     suite"` / `[Revelation] bucket 1: "supernatural communication of truth
     in religious traditions"` / `[Valine] bucket 2: "valine, one of the nine
     essential amino acids for human nutrition"`
   - `[Rummikub] bucket 0: "Rummikub"` / `[Jim Clark] bucket 1: "Jim Clark,
     winner of the 1965 Indianapolis 500"` / `[Riyadh] bucket 2: "Riyadh, the
     largest city and capital of the Kingdom of Saudi Arabia"`
   - `[Saddle] bucket 0: "saddles for horseback riding"` / `[Expressionist
     dance] bucket 1: "early modern dance movement with Mary Wigman and
     Rudolf Laban"` / `[Automotive industry] bucket 2: "the sector of
     companies involved in designing and manufacturing motor vehicles"`

   One rough edge, noted for completeness: occasionally a bucket-0 label from
   a verbose topic (e.g. `[Cimarron River (Arkansas River tributary)] bucket
   0: "the 698-mile tributary of the Arkansas River system"`, 12 tokens) is
   about as long as a bucket-1 label from a terse topic (`[San people] bucket
   1: "San people in southern African anthropology and history"`, 8 tokens).
   This is the residual the 0.0139 overlap number already accounts for; it
   never inverted bucket 0 vs. bucket 2 in the 30 sampled tuples inspected by
   hand, matching the exact measurement.

## Rules B and C, for comparison (not chosen)

Not needed by the decision procedure since A cleared the 0.15 bar, but
computed anyway since they were cheap:

| rule | overlap prob | bare-title failures | topics with an unusable bucket |
|---|---|---|---|
| A (chosen) | 0.0139 | 0.29% | 0% |
| B (global thresholds, 33rd/67th pct = 9/13 tokens) | 0.0000 | 1.50% | 8.78% (empty bucket) |
| C (hybrid: B + A fallback) | 0.0016 | 0.93% | 18.00% (bucket ≤1 label) |

B's overlap is even lower by construction (global thresholds partition by an
absolute cutoff), but at the cost of an 8.78% empty-bucket rate and a worse
bare-title score — both because a genuinely terse or verbose topic gets all
its labels shoved into one or two buckets under a fixed global cutoff. C's
fallback fixes the *empty*-bucket case but the resulting mix of two different
partition rules produces more size-1 buckets than A alone, and is also
unnecessary complexity for step 3 to carry. A is simpler and better on every
axis except the already-tiny overlap number, so it stands as chosen with no
need to invoke the 0.15-0.30 "ask the user" branch.

## Unanticipated findings

- Character/token length correlation across all 839,602 distinct labels:
  **0.774**. This is positive but looser than "closely agree" — apostrophes,
  numerals, and multi-word proper nouns (e.g. `"Hon'ami Kōetsu"`) tokenize
  unevenly relative to raw character count. Token length is used throughout
  per the plan; characters should not be substituted as a stand-in without
  re-checking this correlation if it matters for a future step.
- The base-model tokenizer cache gap (see Environment note above).
