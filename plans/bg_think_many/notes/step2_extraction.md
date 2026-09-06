# Step 2 findings: the multi-topic prompt and the grouped extractor

**Code complete; Gate 1 passed for both k=2 and k=3, and both full
extractions are done and synced to local `outputs/`.** Run on the remote 8B
(RTX 4090); see "Gate 1 and the full runs" below for the numbers.

## The shared-body lift worked

`extract_pangram_vectors` kept the compliance filter, the forced-response
variants, the extraction loop and the report writer; all four moved to
`adapter_training/pangram_extraction.py`, parameterised by three callbacks
(`instruction_of`, `record_of`, `identify`) plus a unit noun and a records
filename. Both extractors are now ~150-line CLIs over it, and the loop did not
have to branch on the record type anywhere.

## Deviations from the step file

- **No `--dataset-file`/`--dataset` on the grouped extractor.** The step file's
  §4 example passed one, but `--source-topics`' own `topics.json` already
  carries every field a group needs (title, labels, split), so the upstream
  dataset is not consulted at all. This also makes the extractor work with no
  network egress, which the Hub path would not.
- **`--limit` counts groups, not topics** (the single-topic extractor's counts
  topics). Gate 1 asks for 500 *groups*.
- **`GroupRecord` has a `variant` field** the step file's sketch did not list.
  Without it `filter_report.json` cannot report `variant_counts`, which Gate 1
  reads. It is optional and defaults to None, exactly like `TopicRecord`'s.
- **`filter_report.json` count keys are named for groups**
  (`groups_seen`/`groups_kept`/`train_groups`/...), and `positions.json` has
  `n_groups`, not `n_topics`. Calling a group a topic in a file whose whole
  point is the group population seemed worse than the rename; the keys Gate 1
  actually reads (`keep_rate`, `variant_counts`,
  `first_mismatch_histogram`) are unchanged, and nothing in the codebase reads
  the renamed ones.
- `positions.json` also records the pool provenance: `source_topics`, `seed`
  and the dropped semicolon titles.

## The semicolon filter (parent plan §8)

9 topics dropped, of the 47,001 in `outputs/bg_think_l19`: `Mary Shelley`,
`Walden`, `Mamoru Miyano`, `Paris Street; Rainy Day`, `The Second Coming
(poem)`, `Clarissa; or, The History of a Young Lady`, `Semicolon`, `Gustave
Caillebotte`, `Frankenstein`. The tenth, `BoA`, had already failed the
single-topic compliance filter, so it never reached this pool.

## Group counts, against the step file's predictions

Pool after the filter: 42,313 train / 4,679 val. (These were the file's
pre-run predictions; realised counts, from the actual full runs, are below.)

| k | train groups | val groups | predicted (train/val) | vectors | bf16 size |
|---|---|---|---|---|---|
| 2 | 42,312 | 4,678 | 42,320 / 4,680 | 469,900 | 3.85 GB |
| 3 | 28,208 | 3,118 | 28,212 / 3,120 | 313,260 | 2.57 GB |

The small shortfalls are the 9 dropped topics plus the per-round leftovers.

## Local smoke test

`--k 3 --rounds 1 --limit 12` against `Llama-3.2-1B-Instruct` at layer 8 runs
end to end and writes all five files. It keeps **0/12** groups -- but the same
command at `--k 1` keeps only 1/12, so this is the 1B being a weak instruction
follower, not evidence about the 8B's k=3 keep rate. The dominant 1B failure is
at the final sentence position, predicting `".\n"` where `"."` was forced.

**Environment note.** `HF_HOME=/work/hf-cache` emits `Ignoring corrupted tree
cache file ... Permission denied` on every model load: 23 `*/trees/*.json`
files are mode 600 and owned by `ubuntu`, so the `agent` user cannot read them.
Harmless (the loader falls back), but it is noise on every run. The 8B's
absence from that cache is *not* a fault -- only models that fit the local GPU
are cached, by design.

## Gate 1 and the full runs (2026-09-06, remote RTX 4090)

**Gate 1 passed for both k=2 and k=3.** The user extended Gate 1 (originally
k=3 only) to also require k=2 above ~80%; both probes were run before either
full extraction, 500 groups each, `--rounds 2 --layer 19`:

| k | keep_rate | groups_kept | variant_counts | first_mismatch_histogram |
|---|---|---|---|---|
| 2 | 0.998 | 499/500 | 499 with-stop, 0 no-stop | `{3: 1}` |
| 3 | 0.992 | 496/500 | 496 with-stop, 0 no-stop | `{3: 3, 9: 1}` |

Both hold the single-topic run's pattern: every kept group matches the
with-stop variant, the no-stop variant matches nothing. The handful of
rejections are early-position mismatches (position 3, once position 9) -- the
model drifting off-script near the start of the sentence, not the "talks
about the topics instead" failure this step's plan anticipated as the
plausible new one. Both probes report `val_groups: 0`, as expected (groups are
split-major and `--limit` takes a prefix); this does not bias the numbers
above (see the step file's own note on this).

**The two full runs**, sequentially, one GPU job at a time, `--rounds 2
--layer 19 --source-topics bg_think_l19`:

| k | groups seen | groups kept | keep_rate | train groups | val groups | vectors.pt |
|---|---|---|---|---|---|---|
| 2 | 46,990 | 46,799 | 0.996 | 42,140 | 4,659 | 3.6 GB |
| 3 | 31,326 | 30,982 | 0.989 | 27,899 | 3,083 | 2.4 GB |

Both `variant_counts` show 100% with-stop, 0% no-stop, matching every prior
run. Group counts and sizes are close to this file's pre-run predictions
(42,312/4,678 and 3.85 GB predicted for k=2; 28,208/3,118 and 2.57 GB for
k=3) -- the small differences are the keep-rate rejections, which the
predictions could not know in advance.

k=1 was **not** extracted: it is `outputs/bg_think_l19`, reused per the
parent plan's D1. `outputs/README.md` now lists the two new directories.
Checksums of `vectors.pt` were verified to match between the remote and the
local synced copy for both k=2 and k=3.

Everything the extractor needs is in `--source-topics`; there was no dataset
download and no network egress on the extraction path. The only manual step
was materialising `outputs/bg_think_l19/topics.json` (not the 3.9 GB
`vectors.pt`, which the grouped extractor never reads) onto the remote via a
direct `scp`, since the remote's `outputs/` only syncs *from* the remote
automatically, not to it.
