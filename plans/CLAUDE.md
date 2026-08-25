This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- full_position_layer_sweep.md
    - Ready to execute, not started. Extends run_pipeline.py/config.py/
      extract.py to a full every-layer x every-user-prompt-token-position sweep
      (2 words, all 3 arms, 200 samples/cell), sharded across GPUs. Supersedes
      the archived selfie_taboo_pipeline.md, whose first pass was never run
      through run_pipeline.py. Includes measured token positions and a
      recommended local test suite.
- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
