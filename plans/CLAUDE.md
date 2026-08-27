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
- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
