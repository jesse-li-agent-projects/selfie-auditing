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
the history behind that result before repeating the approach.

Its successor `bg_think_many` (multi-topic prompts, 1:2:3 mixture, rank-64) is
also done and archived, at `plans/archive/bg_think_many/`. Both of its gates
passed, but the OOD picture is mixed and the paper's own bridge-entity result
regressed -- 70/100 questions against 89/100 for the baseline. Read
`plans/archive/bg_think_many/notes/step6_results.md` before extending the
multi-topic approach; that run changed both the data and the architecture, so
it does not isolate what caused the regression, and the note names the control
run that would.

Its successor `tell_and_think` merged the original paper's `Tell me about X.`
data into that mixture as a fourth, exhaustive source, centred on its own mean,
to test whether the added single-topic data improves detection of unverbalized
concepts. Done and archived, at `plans/archive/tell_and_think/`. All gates
passed and it beat `bg_think_many` on most OOD measures -- bridge entity 77/100
against 70/100, and `book` improves in both taboo harnesses -- but it **did not
beat the baseline adapter**, which was the goal: bridge entity 77/100 and 0.79%
generation hit rate against the baseline's 89/100 and 2.15%. So adding the
paper's own data recovers part of what `bg_think_many` lost without closing the
gap. Read `plans/archive/tell_and_think/notes/step2_results.md` before extending
the mixture approach; it also records that the `chair` taboo word regressed
where `book` improved, so the taboo gain may be word-specific.

The general technique (extract activations from a prompt that has the model
write something while thinking about a topic in the background) is still live;
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
