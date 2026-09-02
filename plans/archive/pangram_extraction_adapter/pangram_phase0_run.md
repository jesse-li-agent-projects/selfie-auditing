# Step 3 / phase 0: the full-budget arm-B run and its comparators

Part 6 of 7 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §5.4, §5.4.1, §4.6, D7 and D11, which define phase 0).

**Depends on** `plans/pangram_step2a_loss_and_eval.md`,
`plans/pangram_step2b_training_loop.md`, `plans/pangram_step2d_retrieval_eval.md`, and
`plans/pangram_step0_benchmarks.md` -- in particular **the 1.3662 trainer-correctness gate
must have passed**. If it did not, stop and report; this plan's numbers would be meaningless.

**This is the headline result of the whole experiment**, not a probe: arm B alone, at the
full budget, scored against forward-only comparators. ~3.0 A100-hours all in (parent plan
§4.6), of which ~1.76 is the single training run and cannot be parallelised.

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

**Can the adapter recover a topic the model never says out loud?** Arm B's activations come
from a response whose surface text is identical for every topic, so the topic is present
only as an unverbalised influence. The claim is that an adapter trained on those activations
can still be made to name the topic -- and the test is whether it does so far better than an
untrained projection reading the same vectors.

**The headline is recall@k, measured against an untrained floor on pangram vectors.** Loss
is reported but is not the claim: it says how well an adapter fits its own labels, not
whether a reader would recover the topic.

**Do not compare arm B against 1.3662 or against the published upstream adapter as a
target.** Arm A's activations sit one token after a prompt that names the topic out loud;
arm B's do not. Different tasks, different example populations (~842k pangram val examples
against 84k baseline), different topic sets. The published checkpoint appears in this plan
twice, both times legitimately: as the D10 trainer-correctness check, and as a **labelled
reference point** in the retrieval table. Neither is a gate on arm B. See parent plan §5.4
for the table of which comparisons are valid.

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
  the full 49,637 topics (parent plan §9.3). A materially lower rate means something
  changed; investigate before spending 1.76 GPU-hours on it.
- `variant_counts` should show roughly the same ~68/27 split. A very different split makes
  the last position's mean rest on a different population.
- `first_mismatch_histogram` and the failure list: the probe found the dominant real failure
  mode is the model **substituting topic words into the pangram** ("The quick brown
  **monarch** jumps..."), at ~4%. Confirm at full scale, because that mode is evidence
  against the per-position centring assumption (parent plan §5.3, §9.4) and is a risk to
  re-check at scale rather than extrapolate from 500 topics.
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

### 3. The retrieval eval -- the headline

Build the index once and reuse it for every row. All rows share the index, the query topic
set, the decoding settings and the seed (`plans/pangram_step2d_retrieval_eval.md`).

| row | vectors | checkpoint | role |
|---|---|---|---|
| **arm B** | `vectors/pangram_l19` | `runs/phase0_armB/best.pt` | **the headline** |
| **untrained floor** | `vectors/pangram_l19` | `untrained` | **what the headline is measured against** |
| upstream reference | `vectors/baseline_l19` | published `wikipedia-scalar-affine.safetensors` | labelled reference point only |

- **Centred vectors** are the primary condition, matching the paper's own contrastive-vector
  eval (parent plan §5.3). Also run arm B **raw** as a labelled secondary condition, because
  raw is what the taboo pipeline will feed it (D6). Report the two separately.
- **Query the val topics that survive the pangram filter, for every row**, including the
  upstream one, so a recall difference cannot be a topic-population difference.
- **Arm B is scored at all 10 positions**; the primary number is the mean over positions,
  with the per-position vector reported beside it. That per-position breakdown is
  exploratory (parent plan §5.6) and does not feed any conclusion.
- The paper's published **94% recall@1 for trained adapters against 1% untrained** is the
  scale to read these on. The 1% is the meaningful anchor here -- it is a floor on the same
  kind of measurement.

### 4. The validation losses

Through the same loss path, **centred** vectors (parent plan §5.3):

| what | vectors | checkpoint |
|---|---|---|
| arm B | `vectors/pangram_l19`, seeded 84,211-example val subsample | `runs/phase0_armB/best.pt` |
| the floor | same subsample | `untrained` |
| the D10 check | `vectors/baseline_l19`, full val | published `wikipedia-scalar-affine.safetensors` |

**Arm B's val split is scored on a fixed seeded subsample, not in full.** Arm B has 10 val
vectors per topic, so its full val split is ~842k examples against the baseline's 84,211; a
full pass costs ~1.0 A100-hours and its floor another ~1.0 (parent plan §4.6). Draw 84,211
once with a recorded seed and reuse it for both rows. Say that it is a subsample, and never
change what "val" means between two numbers being compared.

The D10 row may already exist from the step-0 gate; reuse that report rather than re-running
it, and say so.

## Report

Write `plans/notes/phase0_results.md`:

- **the retrieval table first** -- arm B against the untrained pangram floor, centred and
  raw, with the upstream reference row labelled as a reference point and the paper's 94%/1%
  beside them;
- the per-position recall breakdown, marked exploratory;
- the loss table, with arm B against its own floor, and the D10 check's measured value
  beside 1.3662. State in one sentence that arm B's loss is not comparable to 1.3662 and
  why, so the next reader does not redo that mistake;
- the **loss-vs-examples-seen curve** for arm B from the metrics JSONL, not just the
  endpoint -- the parent plan §5.6 wants the curve because it is what says whether the
  budget was adequate. If arm B is still improving at the cap, that is a finding: say so,
  and say that the cap can be extended deliberately;
- the extraction filter numbers at full scale, against the 500-topic probe (parent plan
  §9.3), including the word-substitution rate;
- measured wall clock and cost, against the parent plan's ~3.0 A100-hour estimate;
- the verdict, stated plainly, including whether the gate holds.

## The gate

Phase 0 is a gate as well as a result (parent plan §5.5). The gate is **whether arm B clears
its own untrained floor by a margin worth attributing** -- on recall@k first, on loss as
support. If it does not, the pangram activations did not carry recoverable topic signal this
adapter could learn, the interesting question becomes *why*, and arm C is not automatically
the next thing to buy. Stop, report, and let the user decide -- do not roll straight into
`plans/pangram_phases12_and_report.md`.

## Done when

- The arm-B checkpoint, its metrics JSONL, `final_eval.json` and the retrieval reports exist
  and are durable (synced back or explicitly saved -- a vast instance is disposable).
- The retrieval table, the loss table and the curve are written up.
- `plans/notes/phase0_results.md` records the verdict and the gate decision.
- Committed on a worktree branch with an undrafted PR. Vectors and checkpoints go to
  `outputs/` (gitignored) -- commit the notes, not the artefacts.

## Do not

- Do not compare arm B's loss to 1.3662, to the published checkpoint, or to arm A, and do
  not gate on any of those (parent plan §5.4).
- Do not train arm A. Phase 0 does not need it; phase 1 runs it as a replication check.
- Do not shrink the corpus, thin the positions, or change the budget to save time.
- Do not report an arm-B number without saying which centring condition and which query set
  produced it.
