# Step 1: key the mixture by source, not by k

Part 1 of 2 in the execution of `plans/tell_and_think/tell_and_think.md` (the
parent plan -- read §5 D2, D3, D6, D7, D8, D9 and §6 Gate 1).

**No GPU, no network.** This is the whole code change the plan needs; step 2
spends the GPU hours and must not start until this has merged.

## Research question

Quoted in the parent plan §1. Read it there; do not restate it in your own
words, here or in the report.

## 1. Why the current code cannot express the mixture

`build_mixture` and `--vectors-k` key everything on *k*, an integer: it is the
directory-map key, the ratio order, and the reporting-slice label all at once.
The parent plan's D2 needs **two sources at k=1** -- `tell`
(`outputs/baseline_l19`) and `bg1` (`outputs/bg_think_l19`) -- so the key
collides and `parse_vectors_k` rejects the second as a repeat.

Splitting *identity* from *k* is the change. Note that k is already derivable
from the data: `sample_examples` reads `len(record.titles)`, so nothing outside
the CLI actually needs k as a key.

## 2. What to build

Four changes. Keep them in separate commits -- the flag rename touches many
call sites but is trivial; the centring-group change is small but is the only
one with real logic in it (project rule on commit size vs complexity).

### 2.1 A source-keyed mixture

Replace the `Mapping[int, Path]` keys in `grouped_examples.build_mixture` and
`_load_mixture_store` with source names (`str`), and derive each source's k from
its records rather than from the key. `k_ranges` becomes `source_ranges`.

`_load_mixture_store`'s `k == 1` branch currently decides *by the key* whether a
directory holds `topics.json` or `groups.json`. That was always a proxy; make it
explicit -- dispatch on which file the directory actually has. Both `tell` and
`bg1` are single-topic directories and both need the `TopicRecord` ->
`GroupRecord` adaptation via `group_record_from_topic`.

### 2.2 A CLI that can name two sources at the same k

Add `--vectors-source NAME=DIR`, repeatable, with `DIR` written under `outputs/`
(implicitly prepended, as every other path flag in this script does -- and note
the parent project's warning that the `outputs/` prefixing is *not* consistent
across flags, so check each one individually).

`--mixture-ratio` takes its weights **in the order `--vectors-source` was
given**, not sorted. Sorting is what the k-keyed version did and it has no
meaning for names.

Keep `--vectors-k K=DIR` working, as an alias that names its sources `k1`, `k2`,
`k3` and keeps the existing sorted-by-k ratio order, so `bg_think_many`'s
archived command still reproduces its run. `--vectors`, `--vectors-k` and
`--vectors-source` are mutually exclusive; `--val-total-examples` is required
with either mixture flag.

### 2.3 Centring groups (D3)

This is the part with actual logic. Today `_load_mixture_store` calls
`pooled_position_means` over *every* source and stores one
`[n_positions, hidden]` table in `_MixtureVectors.means`, indexed by position.

It needs to become one mean table **per centring group**, with each row of the
store knowing which group it belongs to -- so `_MixtureVectors` grows a
`group_of` alongside its existing `position_of`, and `_gather` subtracts
`means[group, position]` instead of `means[position]`.

Add `--centring-group NAME=GROUP`, repeatable. **Default: every source in one
group**, which reproduces `bg_think_many`'s behaviour exactly and keeps the
alias in §2.2 honest.

Watch the position count: `tell` has `n_positions = 1` and the `bg*` sources
have 10, so the per-group tables are ragged. Do not pad one group's table to
another's length -- a `tell` row must never index a position that source has no
mean for. Assert it instead.

### 2.4 Per-source validation slices, and a slice check that works

`val_k_ranges` -> `val_source_ranges` through `train()`, `final_eval.json` and
`run_config.json`'s `mixture_k_ranges`. Keep the JSON keys human-readable --
they are what step 2's Gate 2 table is read off.

The separator-count check `bg_think_many` used to prove a slice really was k=1
(count `"; "` in the composed label) **cannot distinguish `tell` from `bg1`**:
both are single labels and both compose to zero separators. Replace it with a
row-offset check -- every example in a source's range must have a
`vector_index` inside that source's own global offset range. That is a stronger
check than the old one, and it is what Gate 1 asserts.

## 3. Tests

Unit tests only; there is no GPU here. **Note the project rule: passing unit
tests do not mean the codebase works.**

1. Two sources at the same k both load, and their examples land in disjoint,
   correctly-offset index ranges. This is the case the old code could not
   express at all.
2. `--mixture-ratio` follows `--vectors-source` order, including an order that
   is not sorted by k.
3. `--vectors-k 1=A --vectors-k 2=B --mixture-ratio 1:2` produces exactly the
   same sources, ranges and means as the equivalent `--vectors-source` form.
4. With the default single centring group, `_load_mixture_store` returns means
   bit-identical to the current implementation's, on a fixture with more than
   one source. This is the regression guard for D3's "the `bg*` pooled mean is
   unchanged" claim.
5. Two centring groups give each group its own mean, and a ragged group (1
   position against 10) neither pads nor reads out of range.
6. `_split_by_ratio` over 3:1:2:3 and 2,266,173 returns the parent plan D4's
   table exactly, with no remainder to distribute.

## 4. Findings note

Write `plans/tell_and_think/notes/step1_mixture.md`. Record anything that
contradicted this file -- that section is the most valuable part of the note,
as `bg_think_many`'s own step 6 note shows.

## 5. Out of scope

- Do not touch `prompts.SELFIE_TEMPLATE` (parent plan §2).
- Do not change `example_stream` or `bucketed_batches`. The parent plan's D6
  settles the interleaving question: the shuffle is already global, and length
  bucketing is a memory requirement, not a tunable.
- Do not re-extract anything. `outputs/baseline_l19` already exists (§2 of the
  parent plan).
- Do not change any hyperparameter or the projection type (D5).
