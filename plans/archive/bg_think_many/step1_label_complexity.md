# Step 1: choose the label-complexity bucketing rule, and prove it transfers across topics

Part 1 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read its §1, D4 and D5 before starting).

**No GPU. No network. No model.** This is a pure data-analysis step over one
JSON file. It is the cheapest step in the plan and it constrains step 3, so do
it first or in parallel.

**Deliverable:** a findings note at `plans/bg_think_many/notes/step1_complexity.md`,
and one function `label_buckets()` in a new module
`adapter_training/label_complexity.py` that implements the rule the note
chooses, with unit tests.

## Why this step exists

Quoted from the user's request (2026-09-05):

> Note that for any given topic, there are multiple labels with varying levels
> of description; labels in one tuple should have comparable complexities for
> consistency. (I'm not sure if the dataset actually notes the complexity of
> explanation? If it doesn't, we'll need some reasonable workaround like
> checking the length of the label/explanation.)

and, on this step specifically:

> I'd in particular have a step to investigate the tertile complexity breakdown
> and make sure that the resulting divisions have comparable complexity across
> different topics.

**The dataset does not note complexity.** This is established, not to be
re-checked: upstream generated "five varied descriptions at different levels of
detail" per run, then merged several runs and deduplicated with
`dict.fromkeys` (`resources/selfie-adapters/data_prep/wikipedia_topics/dataset_generation/choose_best_prompts.py`
and `merge_topics.py`). The block structure is gone, and topics end with 6 to 20
labels in no recoverable order. So a proxy is needed, and length is the obvious
one.

**The failure this step is looking for.** Splitting each topic's own labels into
terciles gives equal *counts* per bucket, not equal *complexity*. A uniformly
verbose topic's "short" bucket can be longer than a terse topic's "long" bucket.
If that happens often, a label tuple built from "bucket 1 of each topic" mixes a
one-liner with a paragraph, which is exactly what D5 is trying to prevent.

## The data

`outputs/baseline_l19/topics.json` -- 49,637 records, each with `title`,
`labels`, `split`. (The same labels appear in `outputs/bg_think_l19/topics.json`
for the 47,001 topics that survived the compliance filter; use the baseline file,
which has all of them.) Known statistics, for orientation, so you can tell
quickly whether you have loaded the right thing:

- labels per topic: min 6, max 20, mean 16.9
- label length in characters: mean 55.2, p10 35, p50 54, p90 78, max 149

## What to measure

Use **target token length**, not characters, as the primary length. It is what
the trainer's length-bucketed batcher sorts by
(`train_adapter.compute_target_lengths`) and what the sequence-cost estimate in
the parent plan §5 assumes. Tokenize with the Llama-3.1-8B tokenizer via
`model_loading.load_tokenizer` -- the tokenizer alone, never the model, so this
stays a CPU step. Report the character correlation too: if the two agree closely,
say so, and later code may use characters as a cheap stand-in.

Measure all of the following, for the **within-topic tercile** rule (bucket a
topic's own labels by its own length quantiles, 3 buckets):

1. **Bucket sizes.** How many topics have an empty bucket, or a bucket with one
   label? A topic with 6 labels gives 2 per bucket; check nothing is starved.
2. **Bucket boundaries across topics.** The distribution, over topics, of each
   bucket's mean length. Plot or tabulate the spread.
3. **The cross-topic overlap probability** -- the headline number of this step.
   Draw two topics at random and one label from topic A's bucket 0 and one from
   topic B's bucket 2. What fraction of the time is A's "short" label *longer*
   than B's "long" label? A rule that transfers well makes this small. Compute it
   exactly rather than by sampling if it is cheap enough.
4. **The bare-title check.** Many topics carry their bare title as a label (e.g.
   `"William Wallace"`). It is the least detailed description there is, so it
   should land in bucket 0 essentially always. Report the fraction where it does
   not; a rule that puts bare titles in bucket 2 is broken regardless of what the
   other numbers say.
5. **A read-it-yourself sample.** Print ~30 random (topic, bucket) triples --
   one label from each of the three buckets of the same topic -- and read them.
   Then print ~30 *cross-topic* tuples of the shape step 3 will actually build
   (one bucket-i label from each of 3 random topics) and read those. Numbers can
   look fine while the tuples read as mismatched. Put a handful of both, verbatim,
   in the findings note; they are the most useful thing in it for a human reader.

## The candidate rules

Compare at least these three on measurements 1-4:

| rule | how |
|---|---|
| **A: within-topic terciles** | each topic's own 33rd/67th length percentiles |
| **B: global thresholds** | one pair of length thresholds for the whole corpus, from the global label-length distribution |
| **C: hybrid** | global thresholds, but a topic whose labels all land in one bucket falls back to its own terciles |

Rule B makes buckets mean the same thing everywhere by construction, but can
leave a topic with an empty bucket -- measure how often. Rule C exists because
that is the obvious repair; only build it if B's empty-bucket rate is a real
problem.

## How to decide

Prefer the simplest rule that transfers. Concretely:

- If rule A's cross-topic overlap probability (measurement 3) is **below 0.15**
  and its bare-title check is clean, take rule A and stop. It has no empty-bucket
  failure mode, which makes step 3 simpler.
- If A's overlap is **above 0.30**, A does not transfer; evaluate B and C and
  take whichever has the lower overlap while leaving under ~2% of topics with an
  unusable bucket.
- If A lands **between 0.15 and 0.30**, that is genuinely marginal. Write up all
  three, state your recommendation and your reason, and **ask the user** rather
  than picking. This is the one decision in the plan worth a round trip.

These thresholds are a default to keep an agent unblocked, not a measured
standard. Say in the note which branch you took and what the number was.

## What to build

`adapter_training/label_complexity.py`, small and dependency-light:

    def label_buckets(labels: Sequence[str], n_buckets: int = 3) -> list[list[str]]:
        """Partition one topic's labels into complexity buckets, least
        detailed first."""

Rules to hold to:

- Bucket 0 is least detailed. Step 3 and the findings note both depend on this
  order, so assert it in a test.
- Deterministic: same labels in, same partition out, no RNG. Ties in length must
  break deterministically (by the label text, not by input order, so that
  reordering the label list cannot change the buckets).
- Every label lands in exactly one bucket, and no label is dropped.
- Pure Python plus the tokenizer's length function passed in by the caller --
  do not import torch or transformers at module level, so step 3 and the tests
  stay fast. Take a `length_of: Callable[[str], int]` argument, defaulting to
  `len`, and let the caller inject the tokenizer-based one.

Unit tests in `tests/test_label_complexity.py`: the order property, the
partition property, determinism under label reordering, a 6-label topic (the
minimum) and a 20-label topic (the maximum), and the tie-breaking rule.

## The findings note

`plans/bg_think_many/notes/step1_complexity.md`. Keep it short. It must state:
the rule chosen and which decision branch that came from; the five measurements
with their numbers; the verbatim sample tuples; and anything you found that the
plan did not anticipate. If the answer is "rule A, overlap 0.09, ship it", the
note can be one page.
