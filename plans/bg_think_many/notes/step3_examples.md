# Step 3 findings: composed labels and the 1:2:3 example sampler

**Core module and tests done; trainer wiring done; not yet run against the
real extraction directories (needs GPU-adjacent memory/timing to confirm at
scale).** `adapter_training/grouped_examples.py` plus
`tests/test_grouped_examples.py` (20 tests) implement `compose_label`,
`sample_examples`, `group_record_from_topic`, `_split_by_ratio` and
`build_mixture` exactly as specified (D2, D4, D5, D6, D7); `train_adapter.py`
gained `--vectors-k`/`--mixture-ratio`/`--val-total-examples` beside the
existing `--vectors` path, plus `load_grouped_train_and_val` and matching
tests in `tests/test_train_adapter.py`. All 227 non-GPU tests pass.

## Design choices beyond the step file's sketch

- **`_load_mixture_store` is a private helper `build_mixture` calls
  internally**, not something the step file names. It exists so
  `load_grouped_train_and_val` *could* share one load across train and val --
  but it currently doesn't (see below); it's there for whoever wants that
  optimisation later without re-deriving the loading/centring logic.
- **`sample_examples`'s bucket count is read from `len(buckets_of(labels))`**,
  not hardcoded to 3, so a differently-configured `buckets_of` (a different
  `n_buckets`) still works; the default `label_buckets` gives 3, matching D5.
- **`--val-total-examples` has no default and is required with
  `--vectors-k`.** The step file's §4 says the "full" val pool needs "two
  sizes" (this total, and `--val-subsample`) but does not fix what the full
  pool's size should be, and nothing in the parent plan does either -- D7
  only fixes the *train* budget. Rather than invent a number, the flag is
  required, forcing whoever runs step 6 to pick one deliberately. **This is
  open and should be confirmed with the user before step 6.**
- **`--pool-positions` and `--restrict-topics-to` are rejected with
  `--vectors-k`.** Neither has an obvious meaning across three directories
  with three different index spaces (which directory would `--restrict-topics-to`
  intersect against?), and nothing in the parent plan asks for either in the
  grouped case.
- **Train and val each call `build_mixture` independently**, so each loads
  and centres all three directories' `vectors.pt` from scratch -- the step
  file's §3 already flags this cost for one call; doing it twice doubles it
  (see the memory note below). `_load_mixture_store` exists precisely so this
  can be fixed later (call once, slice by split) without touching
  `sample_examples` or the public `build_mixture` contract; not done now
  because it adds indexing complexity for a memory cost the step file itself
  treats as acceptable ("the training machine holds an 8B model already").

## Empty-bucket fallback

Rule A (`label_buckets`, chosen in step 1) never produces an empty bucket on
its own -- every topic has >= 6 labels split three ways, so `divmod` always
gives >= 2 per bucket. The fallback path (`_nearest_nonempty_bucket`) is
therefore dead code under the current corpus and bucketing rule; it is
exercised in tests only via a deliberately bucket-hostile `buckets_of` fixture
that forces an empty middle bucket. **Realised fallback rate on the actual
corpus: 0%**, not measured empirically since it cannot occur under rule A --
this is a structural guarantee, not a measurement.

## Duplicate-rejection rate

Not measured against the real directories (needs `outputs/bg_think_many_l19_k2`
and `_k3`, and the CLI wiring exercises them directly rather than through a
standalone script). The design guard (`max_attempts = max(n_examples * 50,
1000)`) never fired in any test, including one deliberately built to exhaust
a 2-composed-label pool at a 3-example request. On the real k=1 pool the step
file already computes the collision probability as low ("comfortable"); k=2
and k=3 have far larger composed-label spaces (~17^2 x 2 and ~17^3 x 6
per group per position) so collisions there should be rarer still. This
should be measured directly once step 6 runs `load_grouped_train_and_val`
against the real directories, by passing `stats=` through (currently plumbed
in `sample_examples` but not surfaced by `build_mixture` -- a caller wanting
the aggregate would need to add that, since `build_mixture`'s return
signature is fixed by the step file's §3).

## Startup cost (§6): `compute_target_lengths` over ~1.5M composed labels

Measured with a synthetic proxy (not a real tokenizer or real corpus, since
those need the extraction directories and a GPU-adjacent machine): 1,510,782
labels built with the mixture's 1:2:3 k-proportions, each composed from
random short tokens, tokenized with a trivial whitespace-split "tokenizer".

- **1,482,679 of 1,510,782 were distinct** (98.2%) -- close to the step
  file's "nearly all" prediction.
- **Peak Python-level memory for the `dict[str, int]`: ~94 MB.** This is a
  lower bound: a real HF tokenizer's `input_ids` lists and Python's per-object
  overhead for the real label strings (not the short synthetic ones used
  here) will push this up, but it stays in the "few hundred MB" range the
  step file predicted, not GB.
- **Time: ~9s** for the dict-building loop alone with the trivial tokenizer;
  a real BPE tokenizer will dominate this instead of the dict logic, so this
  number is not informative about real wall-clock startup and is not worth
  re-measuring without a GPU machine and the real tokenizer.

Conclusion: the memory number matches the step file's expectation and is not
a blocker. The real startup-time cost (tokenizer calls, not dict bookkeeping)
should be measured once step 6 runs on the real directories, since a trivial
tokenizer cannot predict it.

## Concatenated vector table size

Computed from the realised directories rather than measured by loading them.
`outputs/bg_think_l19` (3.8 GB on disk), `bg_think_many_l19_k2` (3.6 GB) and
`bg_think_many_l19_k3` (5.9 GB, five rounds per D8) hold 470,010 + 467,990 +
774,330 = 1,712,330 rows of 4096 dims. In fp32 that is **~26 GiB for one
concatenated table**, and `torch.cat` holds the chunks and the result together,
so a single `build_mixture` call peaks near 52 GiB. Train and val each build
their own table (see "train and val call `build_mixture` independently",
above), so a run sits at **~52 GiB resident and peaks near 78 GiB**.

This is far above the ~6.5 GB the step file originally estimated. That estimate
predated D8's move from two rounds to five at k=3, which alone grew the k=3
table from 2.4 GB to 5.9 GB, and it did not count the `torch.cat` peak or the
second (val) table. **Check the training machine's free host RAM against 78 GiB
before step 6's run**; the mitigations are in the step file's §3.

## What step 6 should confirm before relying on this

1. **Pick `--val-total-examples`.** Nothing here or in the parent plan fixes
   it; a reasonable starting point is a multiple of `--val-subsample`'s
   default (5000) large enough that `final_eval.json`'s full-pool pass is
   more informative than the periodic subsample, but this is a judgement
   call, not a derived number.
2. **Measure the real duplicate-rejection rate and empty-bucket fallback rate**
   (structurally 0% for the latter, per above) against the real directories,
   by threading `sample_examples`'s `stats=` through if that visibility is
   wanted.
3. **Measure real startup time and peak memory** with the actual tokenizer
   and the actual ~1.5M composed labels, not the synthetic proxy above.
