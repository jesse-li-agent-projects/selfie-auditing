# Step 3: composed labels and the 1:2:3 example sampler

Part 3 of 6 in the execution of `plans/bg_think_many/bg_think_many.md` (the
parent plan -- read D2, D4, D5, D6 and D7 before starting).

**Depends on** step 1's chosen bucketing rule (`label_buckets`) and step 2's
`GroupRecord` / `groups.json` format. It does **not** need step 2's vectors, so
it can be written against a hand-made `groups.json` fixture while step 2's
extraction is still running.

**No GPU. No network.** Everything here is combinatorics over label strings.

**Deliverable:** `adapter_training/grouped_examples.py`, its tests, and one CLI
flag on the trainer.

## The idea in one paragraph

`Example(vector_index, label)` stays exactly as it is -- a vector index and one
label string. This step's whole job is to produce the *right list of them*, with
composed labels, in the right 1:2:3 mixture. Nothing in `loss.py`,
`train_adapter.py`'s loop, or `checkpoints.py` needs to know that a label is now
three labels joined with `"; "`. Keeping that boundary is the point: if you find
yourself adding a `k` field to `Example` or branching inside the loss path, stop
and reconsider, because the design intent is that all the new complexity lives
in this one module.

## 1. Composing one label tuple (D2, D5)

    def compose_label(labels: Sequence[str]) -> str:
        """Join one label per topic into the adapter's target string."""

Join with `"; "`. Exactly that: semicolon, one space. No trailing separator, no
trailing full stop -- `loss.target_text` already strips and appends the eos
token, and adding punctuation here would change the target relative to `bg_think`
for the k=1 case, which must stay comparable (parent plan D1).

The order of `labels` is the caller's business, and the caller permutes it (§2).

## 2. Sampling the examples (D4, D6)

    def sample_examples(
        records: Sequence[GroupRecord],
        n_examples: int,
        *,
        buckets_of: Callable[[Sequence[str]], list[list[str]]],
        seed: int,
    ) -> list[Example]

One draw is:

1. Pick a `(record, position)` uniformly from all valid pairs -- a group of
   `count` vectors offers `count` positions, and every one is an equally good
   query. The vector index is `record.start + position`.
2. Pick a **complexity bucket index** uniformly from `range(3)`.
3. For each topic in the group, pick one label uniformly from that topic's
   labels in that bucket. If a topic's bucket is empty (possible under step 1's
   rule B -- see step 1 §"candidate rules"), fall back to that topic's nearest
   non-empty bucket, and count how often this happens.
4. Draw a **uniformly random permutation** of the k chosen labels, and
   `compose_label` them.

Rules:

- **Deterministic given `(records, n_examples, seed)`.** Use seeded
  `random.Random` instances derived by string, matching the convention already
  in `train_adapter.example_stream` (`f"{seed}-..."`), never a mutated global.
- **Exactly `n_examples` returned.** This is why sampling beats enumeration:
  no rounding over groups, no fractional epochs.
- **Deduplicate.** Reject a draw that repeats an already-drawn
  `(vector_index, composed label)` pair and draw again. The k=3 space is ~29,000
  composed labels per group per position, so collisions are rare and rejection
  is cheap; the k=1 space is much smaller, so bound the retries and raise a
  clear error if the requested `n_examples` exceeds what the pool can supply
  without repeats. That error is a real guard: the k=1 target is 251,797
  examples against a pool of about 42,320 groups x ~10 positions x ~17 labels,
  which is comfortable, but a future change to the mixture could cross the line
  silently.
- Return them in draw order. Shuffling is the trainer's job
  (`example_stream`), not this module's.

## 3. Assembling the mixture (D6, D7)

    def build_mixture(
        directories: Mapping[int, Path],   # k -> extraction directory
        split: str,
        total_examples: int,
        *,
        ratio: Mapping[int, int] = {1: 1, 2: 2, 3: 3},
        seed: int,
    ) -> tuple[VectorStore, list[Example]]

The awkward part, and the reason this function exists: the three k values live
in **three separate extraction directories**, each with its own `vectors.pt` and
its own index space. `Example.vector_index` has to address one tensor. So this
function concatenates the three (centred) vector tables and offsets each
directory's indices by the running row count.

- Load each directory with
  `load_vector_store(dir, center=True, records=..., means=pooled)` (step 2
  added all three parameters). **Centring stays on**: the whole training
  distribution is contrastive, per the project's research background, and only
  *evaluation* uses raw vectors. `pooled` is
  `dataset.pooled_position_means([(dir, records), ...])` over all three
  directories -- **one reference for every k, not each directory's own**
  (D13). Passing each directory's own means instead would delete the "how many
  topics are named" component, which is constant within a k and therefore
  exactly what that directory's mean subtracts.
- k=1's directory is `outputs/bg_think_l19`, which has `topics.json` and
  `TopicRecord`s, not `groups.json`. Adapt it: a `TopicRecord` is a
  `GroupRecord` with one title. Write that adapter as an explicit named function
  (`group_record_from_topic`), not an inline hack, because it is exactly the
  bridge that makes D1's reuse legitimate.
- **Apply the `;` filter to the k=1 records too.** Step 2 drops those topics
  when it forms groups, but `outputs/bg_think_l19` predates that filter and
  still contains 9 of them, so the k=1 path must drop them itself --
  `extract_grouped_vectors.drop_semicolon_topics` takes `TopicRecord`s and is
  the same function. The parent plan's §8 asks for an assert here; the assert
  belongs *after* the filter, not instead of it, or it fails on the first run.
  Filtering also keeps the three k slices on the same 46,992-topic population,
  which is the point of D1's reuse.
- Split the total by `ratio`. For 1,510,782 and 1:2:3 that is 251,797 / 503,594
  / 755,391 -- confirm your arithmetic reproduces those three numbers, and
  handle the remainder explicitly rather than letting integer division lose a
  few examples.
- Concatenating the three centred tables costs about **26 GiB** in fp32 at full
  size: 1,712,330 rows (470,010 at k=1, 467,990 at k=2, 774,330 at k=3 -- five
  rounds, per D8) x 4096 dims x 4 bytes. `torch.cat` holds the chunks and the
  result at once, so one call peaks near 52 GiB, and train and val each build
  their own table (§4), so a run needs ~52 GiB resident and peaks near 78 GiB.
  **Check the target machine's host RAM against that before booking it.** If it
  does not fit, the fallbacks are to keep the tables separate behind a small
  index-mapping object, and to load once and slice by split, rather than to
  reduce the data. Do not assume the GPU box has the RAM because it holds an 8B
  model; that is VRAM, and this is host memory.

## 4. Val examples

Build the val pool with the same function, the same 1:2:3 ratio and a **different
seed**, from the val groups. Two sizes matter:

- the **full** val pool, for `final_eval.json`
- the `--val-subsample` (5,000 by default) that the loop validates on

Additionally, tag the val examples by k so step 6 can report **per-k validation
loss** (parent plan D12, Gate 2). The cleanest way that does not touch
`Example`: have `build_mixture` also return the index ranges belonging to each
k, and let the caller slice. Do not add a field to `Example`.

## 5. Wiring it into the trainer

`train_adapter.load_train_and_val` currently takes one `--vectors` directory.
Add the multi-directory path beside it, do not replace it -- the single-directory
path is what every earlier run used and what the `bg_think` comparison reruns
need.

Suggested flag shape:

    --vectors-k 1=outputs/bg_think_l19 \
    --vectors-k 2=outputs/bg_think_many_l19_k2 \
    --vectors-k 3=outputs/bg_think_many_l19_k3 \
    --mixture-ratio 1:2:3

`--vectors` and `--vectors-k` are mutually exclusive; error clearly if both or
neither is given. Record the resolved mixture (the directories, the ratio, the
per-k example counts and the seed) in `run_config.json`, so a checkpoint can be
traced back to the exact pool it saw.

## 6. A performance note that will bite

`train_adapter.compute_target_lengths` tokenizes **every distinct label** once
and caches it in a dict keyed by the label text. With single-topic labels there
were ~840k distinct strings; with composed labels nearly all 1.5M are distinct.
Expect a few hundred MB of extra resident memory and a slower startup, both
one-off. This is acceptable -- but measure it and put the number in the findings
note, because it is the kind of thing that looks like a hang on a first run.

Do not "fix" it by removing the length bucketing. That bucketing is what keeps
the padded batches tight, and with a 1:2:3 mixture the target lengths now vary
much more than they used to (roughly 16 to 50 tokens), so it is doing *more*
good here than it did for `bg_think`, not less.

## 7. Tests

`tests/test_grouped_examples.py`:

- `compose_label`: the exact `"; "` join; k=1 returns the bare label unchanged
  (this is what keeps the k=1 slice comparable to `bg_think`)
- `sample_examples`: exact count; determinism under a fixed seed; no duplicate
  `(vector_index, label)`; the raise when the pool cannot supply the request;
  the empty-bucket fallback path
- permutation coverage: over many draws from one 3-topic group, all 6 orders
  appear, and roughly uniformly (a loose statistical check, seeded so it cannot
  flake)
- complexity matching: every composed label's parts come from the same bucket
  index, except where the documented fallback fired
- `build_mixture`: the 251,797 / 503,594 / 755,391 split; index offsetting across
  three directories addresses the right rows (build three tiny fixtures with
  known distinguishable vectors and assert the right ones come back); split
  purity -- no train example ever addresses a val group
- `group_record_from_topic`: a single-topic record produces a k=1 group whose
  composed labels equal the original labels

## 8. Findings note

`plans/bg_think_many/notes/step3_examples.md`: the realised per-k counts, the
duplicate-rejection rate, the empty-bucket fallback rate, the measured startup
cost and memory from §6, and the resident size of the concatenated vector table.
