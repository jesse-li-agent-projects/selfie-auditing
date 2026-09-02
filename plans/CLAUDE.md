This directory contains plans for agents.

## General
- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.

## Background thinking (bg_think)
The pangram_extraction_adapter.md plan and its execution parts (pangram_step*,
pangram_phase0_run.md, pangram_phases12_and_report.md) trained a layer-19
SelfIE adapter on an extraction prompt that asks the model to write a fixed
pangram while thinking about a topic. Its trained adapter (now under
outputs/adapters/bg_think, see outputs/README.md) did not beat the baseline
adapter on the OOD tasks that matter, so this plan is done and archived --
see `plans/archive/pangram_extraction_adapter/` (plans) and
`plans/archive/pangram_extraction_adapter/notes/` (execution findings) for
the history behind that result before repeating the approach. The general
technique (extract activations from a prompt that has the model write
something while thinking about a topic in the background) is still live;
future plans on it should not assume the pangram-specific fidelity filter or
single-topic framing those archived plans used.

- research_notes_selfie_mechanism.md
    - Not a plan: the source evidence (SelfIE adapter mechanism, taboo LoRA
      details) that the plans' claims cite. Read it before disputing a claim a
      plan makes about how the adapter works.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
