# Step 2c (opt-in): the shared-prefix KV cache

Part 3 of 7 in the execution of `plans/pangram_extraction_adapter.md` (the parent plan --
read §4.2.1 and D8, which are the whole justification for this file).

**Depends on** `plans/pangram_step2a_loss_and_eval.md` and
`plans/pangram_step2b_training_loop.md`.

## Do not execute this plan unless you are told to

**This step is opt-in. It is not part of the default execution of the parent plan.** An
agent working through the six steps must **skip this one** and go straight to
`plans/pangram_step0_benchmarks.md`. Execute it only when the user asks for it by name.
Reading this file, or being pointed at the parent plan that lists it, is not an
instruction to build it.

The reason is not that the optimisation is wrong; it is that it is a pure speed-up on a
correct path, it costs about a day of fiddly work, and it can break the numbers silently.
It buys nothing until a real run has shown the training is long enough for another 1.39× to
matter -- so the parent plan says to attempt it *last*.

Two further preconditions, both required:

- `plans/pangram_phase0_run.md` has a result.
- **If the equivalence test in this plan is awkward to write, abandon the optimisation.**
  That is the parent plan's own instruction (§4.2.1: "If that test is awkward to write, the
  saving is not worth taking"), not a fallback you invented under pressure.

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
   installed when you run. Make the two flags mutually exclusive with a hard error. The
   "assert the cache is actually in use" check below is the durable guard against this
   across future versions.
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
gradients computed with the cache must match the uncached path.**

Three things make the difference between this being a real test and a test that agrees with
itself. All three are requirements, not refinements.

### Run it in fp32, and make that the primary assertion

Do not make bf16 the headline comparison. A `~1e-2` relative tolerance is not a test: an
off-by-one in `position_ids` or `cache_position`, or a mask that leaks a single prefix
position, fits inside it comfortably. On Llama-3.2-1B in fp32 on a small batch the two
paths should agree to about `1e-5` relative on the loss, and the projection gradients
should be compared **elementwise**, not by cosine similarity and a norm ratio -- cosine
plus norm is what you fall back on when the tolerance is too loose to say anything sharper,
and in fp32 it is not.

Keep a second, loose bf16 run (the ~1e-2 band, cosine + norm ratio) purely to confirm the
model-dtype path is also plumbed. State both tolerances and why in the findings note.

### Assert the cache is actually in use

Without this, the whole test passes **vacuously**. If `--prefix-cache` silently falls back
to the uncached path -- a routing bug of yours, or footgun 1 firing on some future
transformers version -- the two paths agree perfectly, and the test reports success for an
optimisation that is not running.

So, inside the equivalence test, record what the model actually processed: a forward
pre-hook on the model (or on layer 0) reading `inputs_embeds.shape[1]`, asserting it is
`26 - 11 + target_len`, not the full sequence. This single assertion is also the permanent
guard against footgun 1 across transformers upgrades, and it is a better one than any test
that reads `modeling_layers.py` internals, because it checks the observable consequence
rather than the implementation that causes it.

### Calibrate the tolerance against deliberate bugs

Whatever tolerance you settle on, show that it can detect something. Write the equivalence
check as a helper, then run it against three deliberately wrong variants and assert each
one **fails**:

- `n_prefix` off by one, in both directions (10 and 12);
- the `position_ids` / `cache_position` offset omitted;
- the causal mask sized for the suffix only, not prefix + suffix.

If a mutant passes, the tolerance is too loose or the test batch is too easy -- fix that
before believing the main assertion. These mutants are also the cheapest documentation of
what the cache path has to get right.

## The state tests

The cache is mutable state that survives across steps, which is the part a single-forward
comparison cannot speak to.

- **The master prefix is not written through.** Clone the prefix K/V at build time; after
  two training steps assert the master is **bitwise** identical. This catches footgun 4 (a
  cache update writing through a non-contiguous `expand` view) directly, rather than
  inferring it from a length.
- **No state leaks between steps.** Run batch A, then batch B, then batch A again;
  `loss(A)` must be **bitwise** identical on both occasions. Same input and same weights are
  deterministic, so this needs no tolerance at all.
- **Backward cannot walk into the prefix.** Assert the prefix K/V have `requires_grad=False`
  and no `grad_fn`.

## The batch shapes the fixed batch will not reach

The ≥8-example batch above is one shape. Run the equivalence check over these too:

- **A batch smaller than the `batch_size` the cache was built for.** The last batch of an
  epoch is short, and `build_prefix_cache(..., batch_size, ...)` invites a shape bug exactly
  there. This is a likely real bug, not a hypothetical.
- **Batch size 1.**
- **A batch with no padding at all** -- every label the same token length.
- **A one-token label, and one very long label among short ones.**

## Supporting fast tests

The derived prefix length is 11 for the real template; combining `--prefix-cache` with
`--gradient-checkpointing` raises; the cache is rebuilt per step (assert length 11 at the
start of each of two consecutive steps).

Also assert the model's attention and hidden dropout are zero (or force eval-mode
behaviour) for every comparison above. Non-zero dropout makes the two paths consume RNG
differently and quietly turns every equivalence assertion into noise. Llama defaults to
zero, so this is a one-line assert rather than work -- but an unasserted assumption here
invalidates the rest of the file.

## Validate on the real gate, not only on the 1B toy

`plans/pangram_step0_benchmarks.md` already scores the published upstream adapter through
our loss path and checks it reproduces its recorded `best_val_loss` of **1.3662**, on the
real 8B with real vectors and real labels. Re-run that same gate with `--prefix-cache` on.

It is nearly free -- the extraction is already on disk -- and it is the only check in this
plan that exercises the cache on the real model, the real corpus and the real label
distribution rather than a hand-built batch of eight. **Both runs must land in the same
"agreement" band that step 0 defines**; a cached score that drifts out of it means the flag
is wrong, whatever `tests/test_prefix_cache.py` says. Record both numbers.

## Measuring the win

Re-run the benchmark harness from `plans/pangram_step0_benchmarks.md` with the flag on and
off, same card, same micro-batch. Record examples/second and peak memory.
The parent plan predicts ~1.39× and ~7.3k tokens per step (from ~10.2k bucketed). **If the
measured win is under ~1.2×, say so and leave the flag off by default** -- a correct slow
path beats a fast path nobody is confident in.

## Done when

- The equivalence test passes in fp32, with the cache-in-use assertion holding and all
  three mutants failing it -- or the attempt is abandoned with a written reason in
  `plans/notes/step2c_findings.md`.
- The state tests and the batch-shape cases pass.
- The 1.3662 gate reproduces with the flag on, in the same band as with it off, and both
  numbers are recorded.
- The measured speedup and memory are recorded next to the prediction.
- The flag defaults to off and is never silently enabled.
- Committed on a worktree branch with an undrafted PR.

## Do not

- Do not execute this plan at all unless the user asked for it by name. See the top of this
  file.
- Do not enable it for a run whose numbers will be reported, unless both the equivalence
  test and the 1.3662 gate passed with the flag on, on that machine's transformers version.
- Do not weaken the fp32 tolerance to make the test pass. If fp32 disagreement is real, the
  cache path is wrong; a bf16-sized tolerance only hides it.
- Do not restructure the loss path to accommodate it. If the cache cannot be added behind a
  flag without disturbing the uncached path, that is the "awkward" case the parent plan
  says to walk away from.
