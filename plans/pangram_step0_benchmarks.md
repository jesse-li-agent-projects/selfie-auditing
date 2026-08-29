# Step 0, items 3-5: the trainer-correctness gate, the benchmarks, and the debug run

Part 4 of 6 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §4.2, §4.5, §6 step 0 and D10). Read `plans/pangram_adapter_handoff.md` for execution
state; its "Next: step 0, items 3-5" section is what this plan expands.

**Depends on** `plans/pangram_step2a_loss_and_eval.md` and
`plans/pangram_step2b_training_loop.md` being merged.

**This is the first GPU step, and it is a gate.** Its most important output is a single
number: does the *published* upstream adapter, scored through our loss path, reproduce its
recorded `best_val_loss` of **1.3662**? If it does not, nothing downstream means anything,
and finding that out costs ~0.1 GPU-hours instead of a full training run (parent plan §6
step 2, D10).

Items 1-2 of step 0 (compliance generation and the failure taxonomy) are **already done** --
see the handoff: 500 topics on the real 8B, 68.0% exact with the full stop, 27.4% without,
4.6% genuine non-compliance, zero quoting, zero preamble. The extractor already accepts
both variants. Do not redo them.

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

## Machine and preconditions

The local GPU is an 8 GB laptop card and **cannot hold an 8B model** (parent plan D5). Use
the vastai remote via the `vast-remote-broker` MCP server, and only if the user has
directed you to (project rule). A 24 GB 3090 was used for the step-0 probe.

Two logistics facts from the handoff, both of which will bite:

- The remote `agent` account has **no network egress**. Models come from the
  `hf-fetch.sock` daemon (write `<repo id>\n`, read one line back); it serves **models
  only**, not datasets, and only from `/etc/hf-model-allowlist.txt`. Start the
  `meta-llama/Llama-3.1-8B-Instruct` fetch as one of the first things you do -- it is slow
  and can run while you set up.
- The topic dataset JSONL (`wikipedia_vital_articles_level5_dataset.jsonl`, ~55 MB) must be
  placed by hand. **Only `*.py` files sync automatically.** The base64-in-a-`.tmp.py` trick
  used for the 500-topic probe sample does not scale to 55 MB -- ask the user how they want
  the file placed rather than rebuilding a bigger version of that hack.
- The synced tree lands at `<remote-root>/.claude/worktrees/<name>/...` (the sync mirrors
  the whole repo including worktrees), and is read-only for the `agent` account. Point
  `remote_exec`'s `cwd` there; write outputs to `/home/agent/`.

The published adapter (`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`,
`wikipedia-scalar-affine.safetensors`) is a model repo, so the fetch daemon may serve it --
try; if it is not on the allowlist, ask the user to add it.

## Item 3+: the 1.3662 gate (do this first, it is cheap and it gates everything)

1. **Extract baseline vectors.** `python -m adapter_training.extract_baseline_vectors
   --layer 19 --output-dir vectors/baseline_l19 --dataset-file <jsonl>` over the whole
   corpus (~0.08 A100-hours; the parent plan D9 says extract everything once rather than
   subsample). This is also phase 0's baseline extraction, so
   `plans/pangram_phase0_run.md` will not need to repeat it -- keep the output.
2. **Score the published adapter** on the baseline **val** split, **centred** vectors --
   this is the reproduction check. 1.3662 was computed by upstream's own `validate()`,
   which draws train and val from the same mean-subtracted `vectors_file`
   (`training/data.py::create_dataloaders`); there is no raw path at val time upstream, so
   this run must pass `--center` or it measures something else and cannot reproduce 1.3662:

       python -m adapter_training.evaluate_adapter \
           --vectors vectors/baseline_l19 --split val --center \
           --checkpoint <repo>:wikipedia-scalar-affine.safetensors \
           --report eval/upstream_published_centred.json

   Optionally also score with `--no-center` (raw) as additional context -- e.g. as a sanity
   check on how much the centring choice matters -- but do not present that number as the
   1.3662 reproduction; only the centred score is.

3. **Score the untrained floor** (`--checkpoint untrained`) on the same split, **centred**
   vectors, for the parent plan §5.4.1 comparator table (must match the centring the
   published-adapter score above used, or the table is not comparable).

**Interpreting the result.** Exact agreement is not expected; the comparison crosses
trainers *and* extractors. Known sources of drift, from the handoff: our extractor derives
padding-aware `position_ids` where upstream uses `arange`; batching differs; our per-batch
mean-of-sequence-means aggregation differs from upstream's mean-over-batches on the final
partial batch. Suggested reading:

| measured | reading |
|---|---|
| within ~0.02 of 1.3662 | agreement; proceed |
| 0.02-0.10 off | investigate extraction before the trainer (handoff's own advice), but likely proceedable if you can name the cause |
| > 0.10 off, or the untrained floor is not clearly worse | **stop and report**; do not start a training run |

Record the number, the tolerance you applied, and your reasoning in the handoff either way.
This is the plan's headline output.

## Item 3: throughput and memory benchmarks

A throwaway `.tmp.py` (project convention for throwaway scripts) that measures
**examples/second and peak memory** for the real arm-B config across the configurations the
parent plan §4.2 puts in play, on the card actually being used:

| configuration | what changes |
|---|---|
| reference batching, checkpointed | pad to batch max, gradient checkpointing on |
| + length bucketing | the trainer's default sampler |
| + gradient checkpointing off, micro-batch 32 / 64 / 128 | `--micro-batch-size`, `--gradient-checkpointing` absent |
| + prefix cache | only if `plans/pangram_step2c_prefix_cache.md` has landed; skip otherwise |

Run each for enough steps to get a stable rate (~50 steps after a warmup of ~10) and record
`torch.cuda.max_memory_allocated`. Expectations to check against, from the parent plan
§4.2: 53.0 -> 39.7 tokens/example from bucketing, ~1.5× from dropping checkpointing, ~22.5
GB at micro-batch 64 uncheckpointed on a 24 GB card. **The honest expectation is that a 24
GB card needs micro-batch 32-64 uncheckpointed.**

Then re-derive the run's cost from the *measured* rate rather than the parent plan's table
(§4.6 says to do exactly this; its figures carry ±40%).

## Item 4: prefix-cache equivalence

Only if `plans/pangram_step2c_prefix_cache.md` has been executed. Its own equivalence test
is the deliverable; here you just confirm it passes on the real 8B and record the measured
speedup. If 2c has not been done, write "not attempted" in the handoff and move on -- it is
explicitly optional.

## Item 5: the ~50-step debug run

A throwaway run of the **real arm-B config** at `--max-steps 50`, on real pangram vectors.
Its only job is to catch a mis-wired pipeline: extraction misaligned, centring applied
twice, the loss path mis-indexed. It is **not an experiment** and its loss is not a result
(parent plan §5.4.1 is explicit that an earlier draft's 1/8-budget "preliminary experiment"
was paying for a measurement it could not fairly interpret).

It needs pangram vectors. Either run the full pangram extraction now (~0.16 A100-hours;
phase 0 needs it anyway, so this is not wasted) or a `--limit 2000` subset purely for the
debug run. Prefer the full extraction if the JSONL is in place -- it removes work from
`plans/pangram_phase0_run.md`.

What to check afterwards, concretely:

- Loss at step 0 is in the region of the untrained floor measured above, not wildly off.
- Loss falls over 50 steps.
- The LR curve in the metrics JSONL matches the 2,951-step schedule at steps 0-50, not a
  50-step schedule.
- Sampled soft-token norms and the projection scale are finite and moving.
- The vectors the run consumed were **centred** (the run config records the means file) and
  the val pass was **raw**.

## Output

A findings note at `plans/notes/step0_findings.md` (create the directory if needed):

- the measured published-adapter loss vs 1.3662, and the untrained floor;
- the benchmark table, with the chosen micro-batch size and checkpointing setting;
- the re-derived cost of a full-budget run on this card;
- the debug run's observations;
- **the exact command line the phase-0 run should use.**

Then update `plans/pangram_adapter_handoff.md`: step 0 complete (or which items were
skipped and why), and the settings phase 0 inherits.

## Done when

- The 1.3662 gate has a number and a verdict, both written down.
- The benchmark table exists and names a micro-batch size.
- The 50-step debug run completed and its checks were made.
- Any extraction artefacts produced are left in place for phase 0, and the handoff says
  where they are (remote path *and* whether they synced back).
- Committed on a worktree branch with an undrafted PR (the note and any `.tmp.py`).

## Do not

- Do not treat the debug run's loss as a result, and do not scale it up into a
  "preliminary experiment".
- Do not start phase 0 if the gate failed. Report instead.
- Do not run compute-intensive work concurrently with another agent's (project rule: at
  most one at a time locally; ask before using the remote).
