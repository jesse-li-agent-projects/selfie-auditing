# Step 2c (optional): the shared-prefix KV cache

Part 3 of 6 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §4.2.1 and D8, which are the whole justification for this file). Read
`plans/pangram_adapter_handoff.md` for execution state.

**Depends on** `plans/pangram_step2a_loss_and_eval.md` and
`plans/pangram_step2b_training_loop.md`.

**This step is optional and gated.** It is worth ~1.39× on every training run and about a
day of fiddly work. The parent plan says to attempt it *last*, after a real run has shown
the training is long enough for another 1.39× to matter. **If the equivalence test in this
plan is awkward to write, abandon the optimisation** -- that is the parent plan's own
instruction (§4.2.1: "If that test is awkward to write, the saving is not worth taking"),
not a fallback you invented under pressure.

Do not start this before `plans/pangram_phase0_run.md` has a result, unless the user says
otherwise.

## What it is

`interpret.SELFIE_TEMPLATE` tokenizes (with `add_special_tokens=False`) to **26 tokens**
with the `<|reserved_special_token_0|>` injection slots at positions **11 and 22**.
Positions 0-10 therefore precede any injection: their keys and values are byte-identical
for every example in the corpus and have no dependence on the projection, so they need
neither a forward recomputation nor a backward pass.

Compute that 11-token prefix once at startup, expand it across the batch as a frozen
`past_key_values`, and start each step at position 11. That removes 11 of every ~39.7
tokens. It is **exact**: the prefix is causal and constant, and later positions still
attend to it through the cached K/V.

Derive the slot positions by scanning the tokenized template, not from the literals above;
assert they match, so a template change fails loudly rather than silently caching a token
that is no longer constant.

## The footguns, in the order you will hit them

1. **It does not compose with gradient checkpointing.** Verified in the installed
   transformers 4.57.6: `GradientCheckpointingLayer.__call__` in `modeling_layers.py` sets
   `use_cache=False` **and nulls `past_key_values`** whenever checkpointing is on and the
   model is training. It only emits `logger.warning_once`, so the cache is silently
   discarded and the run merely gets *slower*. Re-verify this against whatever version is
   installed when you run. Make the two flags mutually exclusive with a hard error.
2. **HF `Cache` objects are mutated in place.** The prefix cache must be rebuilt (or
   cropped back to length 11) each step, not shared across steps.
3. **`position_ids` and `cache_position` must both account for the 11-token offset**, and
   the causal mask must cover prefix + suffix. The repo already has the padding-aware
   position-id helper (`adapter_training.extract_common.position_ids_from_mask`) as a
   worked example of getting this wrong being silent.
4. **The expanded K/V must not be written into.** Expanding a batch-1 prefix across the
   batch with `expand` gives non-contiguous views that a cache update will happily write
   through; use a fresh per-step copy.

## Build

- `adapter_training/prefix_cache.py`: `build_prefix_cache(model, template_ids, n_prefix,
  batch_size, device)` returning a `Cache`, plus the position/mask bookkeeping the loss
  path needs.
- `adapter_training/loss.py` grows a `prefix_cache: bool` option in `LossConfig` that
  routes the forward pass through it. The loss reduction, the slicing, and the returned
  stats must be untouched.
- `adapter_training/train_adapter.py` grows `--prefix-cache`, defaulting **off**, erroring
  if combined with `--gradient-checkpointing`.

## The one test that matters

`tests/test_prefix_cache.py`, marked `hf_cache` (Llama-3.2-1B is enough -- this is a
plumbing property, not a scale property):

**On a fixed batch of ≥8 examples with varied label lengths, the loss and the projection
gradients computed with the cache must match the uncached path.** Tolerance: bf16 noise on
the loss (~1e-2 relative is generous but honest; state the tolerance you used and why), and
the gradient check should compare cosine similarity ≈ 1 plus a norm ratio ≈ 1 rather than
elementwise equality.

Supporting fast tests: the derived prefix length is 11 for the real template; combining
`--prefix-cache` with `--gradient-checkpointing` raises; the cache is rebuilt per step
(assert length 11 at the start of each of two consecutive steps).

## Measuring the win

Re-run the benchmark harness from `plans/pangram_step0_benchmarks.md` with the flag on and
off, same card, same micro-batch. Record examples/second and peak memory in the handoff.
The parent plan predicts ~1.39× and ~7.3k tokens per step (from ~10.2k bucketed). **If the
measured win is under ~1.2×, say so and leave the flag off by default** -- a correct slow
path beats a fast path nobody is confident in.

## Done when

- The equivalence test passes, or the attempt is abandoned with a written reason in the
  handoff.
- The measured speedup and memory are recorded in the handoff next to the prediction.
- The flag defaults to off and is never silently enabled.
- Committed on a worktree branch with an undrafted PR.

## Do not

- Do not enable it for a run whose numbers will be reported, unless the equivalence test
  passed on that machine's transformers version.
- Do not restructure the loss path to accommodate it. If the cache cannot be added behind a
  flag without disturbing the uncached path, that is the "awkward" case the parent plan
  says to walk away from.
