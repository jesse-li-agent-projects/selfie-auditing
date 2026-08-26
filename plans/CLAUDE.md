This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- replace_smoke_flag_with_dummy_weights.md
    - Replaces `run_pipeline.py`'s `--smoke` flag with dummy weights published
      on the Hub, so a small-model run and a real run differ only by four
      config strings. The end-to-end check becomes `pytest -m gpu` entering
      through the same `main()` the CLI uses. The dummy fixtures are already
      published on the Hub under `cooleytukey/`.
- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
