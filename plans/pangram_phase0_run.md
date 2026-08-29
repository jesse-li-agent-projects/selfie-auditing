# Step 3 / phase 0: the full-budget arm-B run and its comparators

Part 5 of 6 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §5.4.1, §4.6 and D7, which define phase 0). Read
`plans/pangram_adapter_handoff.md` for execution state.

**Depends on** `plans/pangram_step2a_loss_and_eval.md`,
`plans/pangram_step2b_training_loop.md`, and `plans/pangram_step0_benchmarks.md` --
in particular **the 1.3662 gate must have passed**. If it did not, stop and report; this
plan's numbers would be meaningless.

**This is the headline result of the whole experiment**, not a probe: arm B alone, at the
full budget, scored against two free comparators. ~2.8 A100-hours all in (parent plan
§4.6), of which ~1.8 is the single training run and cannot be parallelised.

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

Motivation given: "I'm hoping this approach will make the adapter better at surfacing
hidden thoughts." Standing project question (README Q1): "Will a SelfIE adapter correctly
uncover something the model is actively hiding?"

## The claim being tested

Arm B's val loss **below 1.3662** is the headline. That comparison is fair because the
published upstream adapter was trained at *exactly* this budget (2,951 steps, batch 256),
with these hyperparameters, on this layer, on the same topic-level val split -- so arm A
never needs to be trained in phase 0 (parent plan §5.4.1, D7).

Two caveats to carry into the write-up, neither a reason to skip:

- The comparison crosses trainers. The step-0 gate bounds this; state the gate's measured
  number next to the result.
- Phase 0 does not test arm C, so a win is attributable to "the pangram prompt", **not** to
  per-position training specifically. That attribution is what phase 1 buys.

## Run it

### 1. Extraction, both styles, whole corpus

Skip whichever of these `plans/pangram_step0_benchmarks.md` already produced -- extract
once is the operating principle (parent plan §3.2), and both are needed by phases 1-2 too.

    python -m adapter_training.extract_pangram_vectors  --layer 19 \
        --output-dir vectors/pangram_l19  --dataset-file <jsonl>
    python -m adapter_training.extract_baseline_vectors --layer 19 \
        --output-dir vectors/baseline_l19 --dataset-file <jsonl>

**Do not subsample the corpus and do not thin the 10 positions** (parent plan D9): at a
fixed examples-seen budget, pool size does not enter the cost, so cutting either buys
nothing and introduces a confound plus a re-extraction.

Sanity-check `filter_report.json` before training:

- keep rate should land near the probe's **95.4%** (68.0% with-stop + 27.4% without) over
  the full 49,637 topics. A materially lower rate means something changed; investigate
  before spending 1.8 GPU-hours on it.
- `variant_counts` should show roughly the same ~68/27 split. A very different split makes
  the last position's mean rest on a different population.
- `first_mismatch_histogram` and the failure list: the probe found the dominant real failure
  mode is the model **substituting topic words into the pangram** ("The quick brown
  **monarch** jumps..."), at ~4%. Confirm at full scale, because that mode is evidence
  against the per-position centring assumption (parent plan §5.3) and the handoff flags it
  as a risk to re-check at scale rather than extrapolate from 500 topics.
- Record `n_topics`, `n_vectors`, train/val topic counts. Expect ~47k topics and ~470k
  vectors, against the parent plan's idealised 49,637 / 496,370.

### 2. The arm-B run

    python -m adapter_training.train_adapter \
        --vectors vectors/pangram_l19 --run-dir runs/phase0_armB \
        --budget-examples 755391 --batch-size 256 \
        --micro-batch-size <from step 0's benchmark> \
        --projection-type scalar_affine \
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 --seed 42 \
        --val-subsample 5000 --validate-every 100

2,951 steps, cosine over its own horizon, 10 warmup steps -- upstream's configuration
exactly. Run it as a **complete run**, not a truncated one.

Long job: launch it with `run_in_background` and monitor its output file. Never use `ps` to
check whether it is alive (project rule -- each Bash call has its own PID namespace, so
empty output means "not visible", not "dead"). If a process that should be running cannot
be found, ask the user rather than assuming it died.

### 3. The three final evaluations

All on the **full** val split, **centred** vectors, through the same loss path -- this table
is compared against 1.3662, which upstream's own `validate()` computed on centred vectors
(see `plans/pangram_step0_benchmarks.md`'s 1.3662 gate); raw vectors would not be comparable
to that number even though raw injection is the correct, deliberate choice for the adapter's
own downstream interpretation-time use (§5.3, D2 -- unrelated to this table):

| what | vectors | checkpoint |
|---|---|---|
| arm B | `vectors/pangram_l19` | `runs/phase0_armB/best.pt` |
| arm A, converged, free | `vectors/baseline_l19` | published `wikipedia-scalar-affine.safetensors` |
| the floor | `vectors/baseline_l19` | `untrained` |

The latter two may already exist from the step-0 gate; reuse those reports rather than
re-running them, and say so.

**Arm B's val examples are pangram vectors** -- one per position per val topic, ~842k
examples. That is a bigger val pass than the baseline's 84k. Either accept the cost (~10×
0.10 A100-hours) or evaluate arm B on a fixed, seeded subsample of its val examples and say
which you did. Do not silently change what "full val" means between arms.

## Report

Write `plans/notes/phase0_results.md`:

- the three losses in one table, with the published `best_val_loss` of 1.3662 and the
  step-0 gate's measured value beside them;
- the **loss-vs-examples-seen curve** for arm B from the metrics JSONL, not just the
  endpoint -- the parent plan §5.6 wants the curve because it is what says whether the
  budget was adequate. If arm B is still improving at the cap, that is a finding: say so,
  and say that the cap can be extended deliberately;
- the extraction filter numbers at full scale, against the 500-topic probe;
- measured wall clock and cost, against the parent plan's ~2.8 A100-hour estimate;
- the verdict, stated plainly, including whether the gate holds.

Then update the handoff: step 3 done, where the artefacts live (remote path and whether
they synced back), and whether the phase-0 gate opens step 4.

## The gate

Phase 0 is a gate as well as a result (parent plan §5.5). If arm B cannot beat the
published checkpoint, **the interesting question becomes why**, and arm C is not
automatically the next thing to buy. Stop, report, and let the user decide -- do not roll
straight into `plans/pangram_phases12_and_report.md`.

## Done when

- The arm-B checkpoint, its metrics JSONL, and `final_eval.json` exist and are durable
  (synced back or explicitly saved -- a vast instance is disposable).
- The three-way table and the curve are written up.
- The handoff records the verdict and the gate decision.
- Committed on a worktree branch with an undrafted PR. Vectors and checkpoints go to
  `outputs/` (gitignored) -- commit the notes, not the artefacts.

## Do not

- Do not train arm A. It is free (parent plan D7).
- Do not shrink the corpus, thin the positions, or change the budget to save time.
- Do not report an arm-B number without the step-0 gate's number beside it.
