# Step 2 findings: the multi-topic prompt and the grouped extractor

**Code complete; the two real extractions and Gate 1 are not run.** They need
the 8B on a remote GPU (the local card is 8 GB and `BASE_MODEL_8B` is not in
the local HF cache). Everything below the "Still to run" heading is pending.

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

Pool after the filter: 42,313 train / 4,679 val.

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

## Still to run (needs the remote 8B)

The vectors this step produces are inputs to steps 3 and 6; nothing else in
the plan is blocked on them. Runbook, on the remote (see the `vastai` skill;
one GPU job at a time):

1. **Gate 1, first, and stop on it.**

       python -m adapter_training.extract_grouped_vectors --k 3 --rounds 2 \
           --layer 19 --limit 500 --output-dir bg_think_many_l19_k3_probe \
           --source-topics outputs/bg_think_l19

   Read `outputs/bg_think_many_l19_k3_probe/filter_report.json` and report
   three things: `keep_rate` (the single-topic run kept 94.7%),
   `variant_counts` (there, every kept topic matched the with-stop variant and
   the no-stop variant matched nothing) and `first_mismatch_histogram`. With
   three topics named, the plausible new failure is the model talking about the
   topics instead of writing the sentence, which shows up as an early mismatch.
   **If `keep_rate` is below ~0.80, stop.** Do not run step 2's full
   extractions and do not start step 6. Report the histogram and a dozen
   rejected groups (`failures[]` carries each group's `titles`) -- *which*
   groups are dropped matters more than the throughput, because the surviving
   population is then selected on something.
2. **Confirm `--rounds` first** -- the parent plan's D8 now records that at
   `rounds=2` each k=3 vector is re-used ~2.4 times by the example budget while
   half the k=1 vectors go unused, and that `rounds=5` at k=3 would even that
   out for ~0.15 extra A100-hours and ~3.8 extra GB. The commands below use the
   plan's current `rounds=2`; do not change it without the user saying so.
3. **The two full runs**, sequentially, ~0.15 and ~0.10 A100-hours:

       python -m adapter_training.extract_grouped_vectors --k 2 --rounds 2 \
           --layer 19 --output-dir bg_think_many_l19_k2 \
           --source-topics outputs/bg_think_l19
       python -m adapter_training.extract_grouped_vectors --k 3 --rounds 2 \
           --layer 19 --output-dir bg_think_many_l19_k3 \
           --source-topics outputs/bg_think_l19

   Expect 46,990 and 31,326 groups before filtering, and 3.85 GB / 2.57 GB of
   vectors if nothing is rejected. k=1 is **not** extracted: it is
   `outputs/bg_think_l19`, reused per the parent plan's D1.
4. **Fill in this note**: the Gate 1 numbers, and both runs' realised
   `keep_rate`, `variant_counts` and group counts. Update `outputs/README.md`
   with the two new directories, as the earlier extractions did.

Everything the extractor needs is in `--source-topics`; there is no dataset
download and no network egress on the extraction path.
