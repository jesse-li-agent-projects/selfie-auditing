This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- selfie_taboo_pipeline.md
    - In progress (build order steps 1-2 done: pipeline code + local smoke
      scaffolding implemented and verified end-to-end against
      Llama-3.2-1B-Instruct; Vast.ai setup and the real 8B sweep not started).
      Tests whether a trained SelfIE adapter (wikipedia-scalar-affine,
      Llama-3.1-8B-Instruct) can read a taboo model's hidden secret word out of
      its activations. See `research_notes_selfie_mechanism.md` in this
      directory for the source evidence backing the plan's claims.
- vast_setup_review.md
    - Review of the draft Vast.ai setup scripts, which live outside this repo
      and outside the agent sandbox (see `selfie_taboo_pipeline.md` §8). Lists
      what still needs changing, blockers first. Read this before editing the
      setup directory.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
