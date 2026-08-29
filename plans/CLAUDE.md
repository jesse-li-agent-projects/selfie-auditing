This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- pangram_extraction_adapter.md
    - Trains a layer-19 SelfIE adapter on an extraction prompt that asks the
      model to write a fixed pangram while thinking about a topic, reading
      activations from every response token instead of one. Covers the
      pangram-fidelity filter, the arms needed to attribute any win, and why
      budgeting by examples seen rather than by epochs stops the 10x larger
      example pool from costing 10x the training time. Phase 0 trains just the
      one proposed arm at full budget and scores it against the published
      upstream adapter, which is a fair comparator at zero training cost, so
      the headline result costs about a quarter of the whole plan.
- pangram_step2a_loss_and_eval.md
    - Execution part 1/6 of pangram_extraction_adapter.md. The example
      pipeline, the soft-prompt loss path, and an `evaluate_adapter` CLI that
      scores any checkpoint -- everything the "does the published adapter
      reproduce its own 1.3662?" check needs, and nothing else.
- pangram_step2b_training_loop.md
    - Execution part 2/6. The trainer: budget in examples seen, cosine over
      its own horizon, length-bucketed batching, gradient accumulation,
      subsampled validation, checkpoints the reference loader reads.
- pangram_step2c_prefix_cache.md
    - Execution part 3/6. **Opt-in: do not execute unless the user asks for it
      by name**; the default path through the six steps skips it. The
      shared-prefix KV cache worth ~1.39x, behind a flag, with the equivalence
      test that justifies it -- and explicit permission to abandon it if that
      test is awkward.
- pangram_step0_benchmarks.md
    - Execution part 4/6, first GPU step. The trainer-correctness gate (score
      the published adapter through our loss path against its recorded
      1.3662), throughput and memory benchmarks, and a 50-step debug run.
- pangram_phase0_run.md
    - Execution part 5/6, the headline result. Full extraction of both prompt
      styles, then arm B at full budget scored against the published upstream
      adapter and an untrained floor. Also a gate on whether phases 1-2 happen.
- pangram_phases12_and_report.md
    - Execution part 6/6. Arms A and C, the capacity sweep, generation
      accuracy via the reference's embedding-retrieval eval, the per-position
      exploration, and the final report.
- pangram_adapter_handoff.md
    - Not a plan: the execution state of pangram_extraction_adapter.md. Which
      steps are done, the decisions taken while implementing them that the plan
      does not cover, and what the next step must do. Read it before continuing
      that plan; update it at the end of every step.
- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
