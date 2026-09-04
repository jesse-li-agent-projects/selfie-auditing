# Steps 4-5 / phases 1-2: attribution, capacity, generation eval, final report

Part 7 of 7 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §5.4, §5.5, §5.6 and §6 steps 4-5).

**Depends on** `plans/pangram_phase0_run.md` having completed **and its gate having
opened**. If arm B did not clear its own untrained floor, the parent plan says the
interesting question becomes *why*, and arm C is not automatically the next thing to buy --
check with the user before spending ~8.6 A100-hours here.

**No extraction is needed.** Phase 0 extracted both prompt styles over the whole corpus
(parent plan §3.2, D9). Four training runs at the same budget, plus the generation eval and
the write-up.

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

## The arms, and what each buys

All arms share layer 19, the same topics, the same upstream topic-level splits, the same
per-position centring, and the **same examples-seen budget of 755,391** (2,951 steps at
batch 256). Because the budget is equal, every arm costs the same: one more run.

| arm | vectors per topic | what it tests |
|---|---|---|
| **A** baseline | 1, last prompt token, upstream's own prompt | replication of upstream, under *our* trainer |
| **B** pangram-per-position | 10 (one per response token) | the proposed method -- **already trained in phase 0, reuse it** |
| **C** pangram-mean | 1 (mean of the 10 positions) | pooling before the adapter instead of after |

B and C read the same activations and differ only in *where the pooling happens*. In B the
10 activations never meet: each is its own example, and the adapter must map any single
position to the topic on its own. In C they are averaged before the adapter sees anything.

- If C ≈ B, position-specific detail was not load-bearing, and C is the better deal.
- If B > C, averaging destroys something and per-position training earns its pool.

**B vs C is the phase-1 comparison, and arm A is not part of it.** B and C share vectors,
task and topic population, so their losses and their recall numbers are directly comparable.
Arm A reads an activation from a prompt that names the topic out loud, which is a different
task; its loss is not comparable to either, and its recall number is a labelled reference
point only (parent plan §5.4). What arm A buys is a **replication of upstream under our own
trainer** -- a stronger version of the D10 check -- not an attribution baseline.

## Phase 1: two runs

    # arm A -- upstream replication under our trainer
    python -m adapter_training.train_adapter \
        --vectors vectors/baseline_l19 --run-dir runs/phase1_armA \
        --budget-examples 755391 --batch-size 256 --projection-type scalar_affine \
        --lr 0.01 --init-scale 5.0 --warmup-steps 10 --grad-clip 0.5 --seed 42 \
        --val-subsample 5000 --validate-every 100

    # arm C -- pooled pangram positions
    python -m adapter_training.train_adapter \
        --vectors vectors/pangram_l19 --run-dir runs/phase1_armC --pool-positions \
        ...same flags...

**Arm A is run once, with `scalar_affine` only** (parent plan §5.4): its purpose is a
replication check, so it does not need repeating across architectures. Its result against
the published 1.3662 is the stronger version of the step-0 gate.

**The topic-population question.** The baseline style filters nothing (49,637 topics) while
the pangram style keeps only compliant topics (~95%), so any A-vs-B/C table rests on
slightly different topic populations even where the comparison is otherwise legitimate. Use
`--restrict-topics-to vectors/pangram_l19` on the arm-A run, and report both the restricted
number and (if cheap) the unrestricted one. Parent plan §9.2 flagged this as undecided;
**decide it here and record the decision.** Note that restricting removes only one of the
confounds between A and B/C -- the difference in task remains, which is why A is never B's
target (parent plan §5.4).

## Phase 2: capacity, two runs

The **winning arm** from phase 1 (B or C), trained at `scalar_affine_plus_low_rank` with
`--projection-rank 16` and `--projection-rank 64`. Parameters: `4096 + 1 + 2·4096·r` --
135,169 at r=16, 528,385 at r=64, against 4,097 for `scalar_affine`. Rank is a config value
and other ranks must need no code change.

The parent plan §4.1 notes larger adapters plausibly need a larger budget, and that the
loss-vs-budget curve is what says whether they did. **Keep the budget equal anyway** -- the
comparison is at equal compute -- but if a capacity run is still improving at the cap, say
so explicitly rather than concluding capacity did not help.

## Scheduling

The four runs are independent, so on an N-GPU rig run **one config per GPU** with a
`--device` flag (parent plan §4.3): that reaches full utilisation with no distributed code
and is the better deal here. DDP only shortens a *single* run and phase 0 was the case for
it. Each run is ~2.1 A100-hours (training + in-run validation + one full-val pass + one
generation eval).

Launch with `run_in_background` and monitor output files; never use `ps` to check liveness
(project rule).

## Retrieval accuracy (parent plan §5.6, D11)

The headline metric, already built (`plans/pangram_step2d_retrieval_eval.md`) and already
run once in phase 0. Here it is run for arms A and C and for the two capacity checkpoints.

    python -m adapter_training.evaluate_retrieval \
        --vectors <arm's vectors> --split val --checkpoint <run-dir>/best.pt \
        --dataset-file <jsonl> --center --positions all \
        --report eval/<arm>_retrieval.json

Non-negotiable, or the table is void: **the same index, the same query topic set, the same
decoding settings and the same seed for every arm.** Rebuild nothing between arms -- cache
the index from phase 0 and reuse it. Arm C is fed the average of the same 10 positions it
was trained on; arm A has one vector per topic, so `--positions` does not apply.

Report recall@{1,5,10} for every arm, in the centred condition primarily and the raw one
beside it for arms B and C. Arm A's row is a **labelled reference point**, not a target
(parent plan §5.4).

## The per-position breakdown (arm B only, exploratory)

This falls out of arm B's phase-0 retrieval run at no extra cost -- it is scored at every
position already, with the mean over positions as its primary number. **It does not feed the
B/C conclusion** (parent plan §5.6). Its value is shaping the next iteration: if late
positions (`lazy`, `dog`, `.`) carry most of the topic signal, a follow-up could read only
those and shrink the pool. Report it as exploratory, and remember position 9 exists only for
the ~68% of topics that wrote the full stop (parent plan §9.3).

## Step 5: the report

`plans/notes/pangram_adapter_report.md`, pulling together:

- **the retrieval table first** -- recall@{1,5,10} per arm against the untrained floor on
  each arm's own vectors, with arm A's row and the paper's 94%/1% labelled as reference
  points;
- loss-vs-examples-seen curves for every arm on one axis, plus the endpoint table with each
  arm's own untrained floor. Put the published 1.3662 beside **arm A only**, and state in
  one sentence why arms B and C are not measured against it (parent plan §5.4);
- the B/C verdict and the attribution it does or does not support, and separately whether
  arm A replicated upstream;
- the capacity sweep;
- the per-position exploration, marked exploratory;
- the step-0 failure taxonomy at full-corpus scale (from phase 0's `filter_report.json`),
  including the word-substitution mode;
- measured timings and cost against the parent plan's estimates;
- what the next experiment should be. The parent plan §8 names one candidate already:
  whether the taboo elicitation should *also* use the pangram prompt -- a genuine
  experiment, which belongs in its own plan rather than in this report.

Note for the write-up: **no change to the taboo pipeline is required** to use a winning
adapter (parent plan D6). The adapter maps an activation from whatever prompt produced it
into the fixed interpretation prompt, so a trained checkpoint drops into `run_pipeline.py`
unchanged; `interpret.SELFIE_TEMPLATE` is the *interpretation* prompt and must stay exactly
as trained, while the taboo pipeline's *extraction* prompt is `config.SECRET_PROMPT` read
across `Position.USER_PROMPT_SPAN`.

## Done when

- Four checkpoints exist with their metrics and final evaluations, durable.
- Retrieval accuracy is measured for every arm on the same index, the same query topics and
  the same decoding settings and seed.
- The report is written.
- `plans/pangram_extraction_adapter.md`, this file, and the other six step plans are moved
  to `plans/archive/` with their index entries, per `plans/CLAUDE.md`.
- Committed on a worktree branch with an undrafted PR.

## Do not

- Do not vary the budget between arms. Equal examples seen *is* the comparison.
- Do not compare arms B or C against arm A's loss or against 1.3662 (parent plan §5.4).
- Do not let the per-position breakdown drive the B/C conclusion.
- Do not re-extract anything.
- Do not report a capacity result as "capacity did not help" if the run was still improving
  at the cap.
