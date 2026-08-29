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
      pangram-fidelity filter, the arms needed to attribute a result, and why
      budgeting by examples seen rather than by epochs stops the 10x larger
      example pool from costing 10x the training time. Its S9 carries execution
      state: what is built, what the step-0 probe measured, and the remote's
      logistics. The headline is whether the adapter can name a topic the
      response never states, measured by embedding retrieval against an
      untrained floor on the same vectors -- never against arm A's loss, which
      measures a different task.
- pangram_step2a_loss_and_eval.md
    - Execution part 1/7 of pangram_extraction_adapter.md. The example
      pipeline, the soft-prompt loss path, and an `evaluate_adapter` CLI that
      scores any checkpoint -- everything the "does the published adapter
      reproduce its own 1.3662?" check needs, and nothing else.
- pangram_step2b_training_loop.md
    - Execution part 2/7. The trainer: budget in examples seen, cosine over
      its own horizon, length-bucketed batching, gradient accumulation,
      subsampled validation, checkpoints the reference loader reads.
- pangram_step2c_prefix_cache.md
    - Execution part 3/7. **Opt-in: do not execute unless the user asks for it
      by name**; the default path through the seven steps skips it. The
      shared-prefix KV cache worth ~1.39x, behind a flag, with the equivalence
      test that justifies it -- and explicit permission to abandon it if that
      test is awkward.
- pangram_step2d_retrieval_eval.md
    - Execution part 4/7, and where the headline number comes from. Generates a
      description per held-out vector and scores it by GTE-large embedding
      retrieval over all 49,637 topics, reusing the reference's index and
      recall@k. No GPU to build.
- pangram_step0_benchmarks.md
    - Execution part 5/7, first GPU step. The trainer-correctness gate (score
      the published adapter through our loss path against its recorded
      1.3662), throughput and memory benchmarks, and a 50-step debug run.
- pangram_phase0_run.md
    - Execution part 6/7, the headline result. Full extraction of both prompt
      styles, then arm B at full budget scored by retrieval against an
      untrained floor on the same pangram vectors. Also a gate on whether
      phases 1-2 happen.
- pangram_phases12_and_report.md
    - Execution part 7/7. Arms A and C, the capacity sweep, retrieval accuracy
      per arm, the per-position exploration, and the final report.
- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
