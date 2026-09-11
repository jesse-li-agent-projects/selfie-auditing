# Step 1 findings: source-keyed mixture

Execution note for `plans/tell_and_think/step1_source_keyed_mixture.md`. No GPU
was used. Three commits, in the order the step file asked for.

## What contradicted the plan

The most valuable section, per the predecessor's own note. Four things.

**1. `evaluate_retrieval.py` still keys on `k`, and step 2's Gate 3 needs it not
to.** The step file scopes the change to `train_adapter` and `grouped_examples`,
and both are done. But `evaluate_retrieval.py` carries its *own* copy of
`parse_vectors_k` (deliberately duplicated, so the module's `--help` stays free
of `torch`), keyed by `int`, plus `--pool-vectors-k` with the same keying and no
centring-group support at all. So `--vectors-k 1=... 1=...` fails there for
exactly the reason it failed in `train_adapter`: `tell` and `bg1` collide.

**Gate 3 as written in step 2 §5 -- "reporting per source", "centred per D3" --
cannot be run against this module today.** Step 2 must either extend it the same
way (source names plus centring groups, and a decision about what
`--pool-vectors-k` means once sources are named) or narrow Gate 3. This is a
prerequisite, not a nice-to-have, and it is the user's call which way it goes.

**2. `bg1`'s `;`-filter drops 9 topics, which no plan file mentions.** D9 predicted
10 for `tell` and that is exactly right, but the filter has always applied to
`bg_think_l19` as well -- the old `k == 1` branch did it, and the new
file-presence dispatch does the same. Realised drops, from the real
directories:

| source | `;`-filter drops |
|---|---|
| `tell` | 10 |
| `bg1` | 9 |
| `bg2` | 0 |
| `bg3` | 0 |

`bg2`/`bg3` drop nothing because the grouped extractor already applied the filter
before writing `groups.json`. The 9 is not new behaviour and does not change
`bg_think_many`'s footing; it is only undocumented. Note the arithmetic it
implies: 47,001 - 9 = 46,992 addressable `bg1` topics, which is the N
`bg_think_many`'s D8 rounds rule was derived from. That is a useful independent
confirmation that the `bg*` population is unchanged.

**3. `build_mixture`'s return grew past what a tuple should carry.** The step file
asks for `k_ranges -> source_ranges` and, separately, for a row-offset check to
replace the separator count. The check needs each source's *row* range, which the
old return did not expose, and step 2's Gate 1 also wants the `;`-drop counts and
the per-group means. That is a six-tuple. `build_mixture` now returns a `Mixture`
dataclass instead. Not what the step file specified, but a tuple of six was worse.

**4. The `--vectors-k` alias is thinner than the step file implies.** It suggests
keeping a parallel k-keyed path. There is no path: `parse_vectors_k` now returns
`{"k1": ..., "k2": ...}` in sorted-k order and everything downstream reads one
`args.mixture_sources`. The alias is a spelling, not a mode. `--mixture-ratio`
has always been read smallest-k-first, so sorted-k insertion order preserves an
archived command's meaning exactly; a test asserts the two spellings agree.

## What the step file got right, confirmed against real data

`smoke_mixture.tmp.py` (throwaway, not committed) built the real four-source
mixture on CPU at a 2,400-example budget. Results:

- **The separator check really is useless here.** Realised separator counts per
  source: `tell` `[0]`, `bg1` `[0]`, `bg2` `[1]`, `bg3` `[2]`. `tell` and `bg1`
  are indistinguishable, as §2.4 says. The row-offset check separates them
  cleanly.
- **The group mean tables are ragged and are not padded**: `tell` `(1, 4096)`,
  `bg` `(10, 4096)`.
- **The `bg` group's pooled mean is bit-identical** to pooling `bg1`/`bg2`/`bg3`
  alone, with no `tell` contribution. Caveat worth carrying into step 2: this was
  checked by recomputation, against the same three directories, not against a
  stored artefact of `bg_think_many`'s own run -- no such artefact exists, since
  `run_config.json` records the `position_means.pt` *paths* and the pooled mean is
  recomputed from records rather than read from them. Recomputation is the only
  available form of Gate 1's item 3.
- Realised row layout: `tell` 0-49,637, `bg1` 49,637-519,647, `bg2`
  519,647-987,637, `bg3` 987,637-1,761,967.

## Design decisions taken here, not in the plan

- **`--centring-group` must name every source when given at all.** A partial
  mapping would silently drop the unnamed sources into some default reference,
  which is the exact failure D3 exists to prevent. It is an error instead.
- **A row no record addresses is an error at gather time, not a silent -1
  index.** `position_of` is -1 for rows a filter dropped; `means[-1]` would have
  quietly centred them against the last position. Those rows are unreachable
  through any example, so this only ever fires on a bug.
- **The ragged-table assertion runs at construction**, over whole tensors, rather
  than per gather, so it costs nothing in the training loop.

## Not done, and deliberately

- `evaluate_retrieval.py` (see contradiction 1) -- outside this step's stated
  scope, but it blocks step 2's Gate 3.
- `prompts.SELFIE_TEMPLATE`, `example_stream`, `bucketed_batches`, any
  re-extraction, any hyperparameter or projection change. All out of scope per
  §5, all untouched.

289 unit tests pass. **Passing unit tests do not mean the codebase works**
(project rule); they cover the mixture builder and the CLI parsing only, which is
why the real-data check above was run as well.
