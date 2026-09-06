# Step 2: the multi-topic prompt and the grouped extractor

Part 2 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read D1, D3, D8 and §6 Gate 1 before starting).

**Deliverable:** `prompts.background_topics_prompt`, a new
`adapter_training/extract_grouped_vectors.py`, a `GroupRecord` in
`dataset.py`, and two extraction directories under `outputs/`.

**GPU, but small** (~0.25 A100-hours total). The code is written and tested
locally against `config.DUMMY_BASE_MODEL` (Llama-3.2-1B) or a fake model; only
the two real extraction runs need the 8B. Do not hold the GPU while writing
code.

## Research question

Quoted in the parent plan §1. Do not restate it here in other words; read it
there.

## 1. The prompt (D1)

Add to `prompts.py`, beside `PANGRAM_PROMPT_TEMPLATE`:

    def background_topics_prompt(pangram: str, titles: Sequence[str]) -> str:
        """The extraction prompt naming one, two or three background topics."""

Output, exactly:

    k=1: Write "{pangram}." Think about the topic "A" while writing the sentence. Do not write anything else or change the words.
    k=2: Write "{pangram}." Think about the topics "A" and "B" while writing the sentence. Do not write anything else or change the words.
    k=3: Write "{pangram}." Think about the topics "A", "B", and "C" while writing the sentence. Do not write anything else or change the words.

Singular "topic" for k=1, plural "topics" otherwise; "and" before the last;
Oxford comma from k=3 upward. Raise on k=0 or k>3 -- nothing in this plan needs
a longer list, and silently accepting one would produce vectors nothing else can
read.

**The k=1 test is the important one.** Assert that
`background_topics_prompt(PANGRAM, ["A"])` is byte-identical to
`PANGRAM_PROMPT_TEMPLATE.format(pangram=PANGRAM, topic="A")`. This is what makes
the existing `outputs/bg_think_l19` extraction reusable (D1); if a future edit
breaks it, that test must fail loudly rather than the k=1 slice quietly becoming
a different experiment.

## 2. The group records

Groups need a record type: several titles, and each title's own labels. Add to
`adapter_training/dataset.py`:

    @dataclass(frozen=True)
    class GroupRecord:
        titles: tuple[str, ...]
        labels_per_topic: tuple[tuple[str, ...], ...]   # aligned with `titles`
        split: str
        start: int
        count: int

`start`/`count` mean exactly what they mean in `TopicRecord`: the group's
vectors are `vectors[start : start + count]`, and a vector's position index is
`i - start`. `count` is not constant (the compliance filter accepts two response
shapes).

Write these to **`groups.json`**, not `topics.json`, so no existing reader can
half-understand a grouped directory. Add `load_group_records(directory)` beside
`load_topic_records`.

**One change to an existing function.** `load_vector_store` currently calls
`load_topic_records(directory)` itself, purely to get each record's
`start`/`count` for centring. Give it an optional `records` parameter:

    def load_vector_store(directory, *, center=True, records=None):

defaulting to `load_topic_records(directory)` as now, so every existing caller
is untouched, and grouped callers pass their own. Do not duplicate the centring
loop into a second function -- centring must stay in one place, which is the
whole reason `load_vector_store` is documented as the only supported way to read
`vectors.pt`.

## 3. Forming the groups (D3, D8)

New module or a section of the extractor, your choice, but keep it separately
testable -- it is pure combinatorics with no model in it:

    def build_groups(topics: list[Topic], k: int, rounds: int, seed: int) -> list[tuple[Topic, ...]]

Rules:

- **Split-pure (D3).** Partition the train topics and the val topics
  separately, and never mix. The simplest safe shape is to call this function
  once per split and concatenate.
- **One round is a disjoint partition.** Shuffle the split's topics with a
  seeded RNG, then cut into consecutive groups of k. Leftovers (fewer than k at
  the end) are dropped for that round; with 42,320 train topics and k=3 that
  drops 2 topics from the round, which is nothing.
- **`rounds` independent partitions**, each with its own derived seed (follow
  the existing convention in `train_adapter.example_stream`: seed strings like
  `f"{seed}-round-{r}"`, not a mutated global RNG).
- Deterministic given `(topics, k, rounds, seed)`.
- The topic order *inside* a group is the shuffled order; do not sort it. The
  label order is permuted independently later (D2), so a fixed prompt order here
  is not a bias the adapter can exploit.
- `--rounds` decides how many distinct **activations** exist, not how many
  training examples: one group is one forward pass and 10 vectors, and the
  example count is step 3's sampling budget. See the parent plan's D8 for the
  `rounds ≈ examples_k × k / (N × positions)` rule and what `rounds=2` implies
  for k=3.

Expected counts, for k=2 and k=3 with `rounds=2`, over the 47,001 topics that
survived the single-topic filter (42,320 train / 4,681 val):

| k | train groups | val groups |
|---|---|---|
| 2 | 42,320 | 4,680 |
| 3 | 28,212 | 3,120 |

**Drop the topics with a `;` in a label** (parent plan §8) before grouping.
This step owns that filter: it is the first step that decides which topics
exist as groupable units. 9 of the 47,001 source topics are affected.

**Which topic list to group.** Use the topics that appear in
`outputs/bg_think_l19/topics.json`, not the full 49,637. Those already passed
the single-topic compliance filter, so grouping them keeps the k=1, k=2 and k=3
populations on the same topic set, and a difference between the slices is then
not secretly a difference in which topics they cover.

## 4. The extractor

`adapter_training/extract_grouped_vectors.py`, CLI-shaped like its siblings
(light imports, `parse_args()` before the torch import -- see the project style
rules and `extract_pangram_vectors.py`'s own header).

    python -m adapter_training.extract_grouped_vectors \
        --k 3 --rounds 2 --layer 19 --output-dir bg_think_many_l19_k3 \
        --source-topics bg_think_l19

**Reuse, do not reimplement.** The compliance filter, the two response variants,
the per-position mean accumulation and the output writing all already exist in
`extract_pangram_vectors.py` and work unchanged -- none of them cares how many
topics the prompt named. The honest way to build this is to *lift the shared
body out of* `extract_pangram_vectors.py` into a function both extractors call,
parameterised by a "prompt for this item" callback and a record-builder. Copying
the 120-line extraction loop into a second file would be the wrong answer, and
the two copies would drift the first time the filter changes.

If the lift turns out awkward -- if the two record types force branching all
through the loop -- stop and say so rather than forcing it; a shared helper that
is harder to read than two loops is not a win. Report which way you went.

Everything else about the output format is unchanged: `vectors.pt` (bf16, raw,
group-major, positions contiguous), `position_means.pt`
(`[n_positions, hidden]`, fp32, written but not applied), `positions.json`,
`filter_report.json`. `positions.json` gains `"k"` and `"rounds"`, and its
`prompt_style` becomes `"grouped"`.

## 5. Gate 1: probe before the full run

**Run this before spending the full extraction, and report the number.**

Extract with `--k 3 --limit 500`. Read `filter_report.json`:

- `keep_rate` -- the single-topic run kept **94.7%** (47,001/49,637).
- `variant_counts` -- in the single-topic run **every** kept topic matched the
  with-stop variant `"The quick brown fox jumps over the lazy dog."`, and the
  no-stop variant matched nothing at all. If that holds again, note it; the
  second forward pass per batch is then pure cost, and a later step may drop it.
  Do not drop it now -- it is already written, correct, and cheap, and this is
  not the step to optimise.
- `first_mismatch_histogram` -- where the rejections diverge. Divergence at
  position 0 means the model never started the sentence; divergence at the final
  `<|eot_id|>` means it started and would not stop. With three topics named, "it
  starts talking about the topics instead" is the plausible new failure, and it
  shows up as an early mismatch.

**If the k=3 keep rate is below ~80%, stop and report.** Do not extract the full
corpus and do not proceed to step 6. A filter rejecting one group in five is
selecting on something, and *which* groups it drops matters more than the
throughput. Include the histogram and a dozen rejected groups in the report.

## 6. The full runs

Only after Gate 1 passes. Two runs, one GPU, sequentially:

    python -m adapter_training.extract_grouped_vectors --k 2 --rounds 2 \
        --layer 19 --output-dir bg_think_many_l19_k2 --source-topics bg_think_l19
    python -m adapter_training.extract_grouped_vectors --k 3 --rounds 2 \
        --layer 19 --output-dir bg_think_many_l19_k3 --source-topics bg_think_l19

(`--output-dir` is written under `outputs/`, which the argument type prepends.)

Expect ~0.15 and ~0.10 A100-hours, and ~3.9 GB and ~2.6 GB of vectors. The k=1
population is **not** extracted: it is `outputs/bg_think_l19`, reused per D1.

## 7. Tests

`tests/test_extract_grouped_vectors.py`:

- the three prompt shapes, and the k=1 byte-identity assertion (§1)
- `build_groups`: split purity (no group mixes splits), disjointness within a
  round, determinism, the leftover-drop count, and that `rounds=2` gives each
  topic exactly two groups
- `GroupRecord` round-trip through `groups.json`
- `load_vector_store(records=...)` centres a grouped directory correctly, and
  the default path still centres a single-topic one -- the existing
  `tests/` coverage for that must keep passing untouched
- a tiny end-to-end extraction against the dummy 1B model or a fake, checking
  `start`/`count` address the right rows

## 8. Findings note

`plans/bg_think_many/notes/step2_extraction.md`: the Gate 1 numbers, the final
keep rates and variant counts for both real runs, whether the shared-body lift
worked or was abandoned, and any way the real output differed from what this
file predicted.
