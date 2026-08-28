# Hand-off: pangram extraction adapter, after step 1

Companion to `plans/pangram_extraction_adapter.md`, which stays the authority on the
research question, the arms, and the budget. This file records only what an agent picking
the work up next cannot read off that plan: what is built, what was decided while building
it, and what to do next. Update it at the end of every step.

**Research question** (quoted, plan S1 -- never paraphrase it): use the extraction prompt
`Write "The quick brown fox jumps over the lazy dog". Think about the topic "<topic>" while
writing the sentence. Do not write anything else or change the words.`, extract activations
from all response tokens, layer 19, Wikipedia topics, with a topic filter for generations
that correctly reproduce the phrase.

## Status

| step | state |
|---|---|
| 0 probe | not started |
| 1 extraction script | **done** -- `adapter_training/extract_topic_vectors.py`, branch `worktree-pangram-extract-step1` |
| 2 trainer | not started |
| 3 phase 0 (extract + arm B run) | not started |
| 4 phases 1-2 | not started |
| 5 report | not started |

## What step 1 built

`adapter_training/extract_topic_vectors.py`, tested by `tests/test_extract_topic_vectors.py`.

    python -m adapter_training.extract_topic_vectors --prompt-style pangram \
        --layer 19 --output-dir outputs/vectors/pangram_l19

Both prompt styles go through one code path. `baseline` renders each topic's own dataset
prompt and keeps one vector at the last prompt token; `pangram` renders the prompt above,
teacher-forces the sentence plus `<|eot_id|>`, and keeps one vector per sentence token.

Outputs, per style, in `--output-dir`:

| file | contents |
|---|---|
| `vectors.pt` | `[n_vectors, hidden]` bf16, **raw** (uncentred), topic-major, positions contiguous |
| `topics.json` | per surviving topic: title, prompt, labels (once), split, `start`, `count` |
| `positions.json` | run metadata: style, layer, model, the decoded position tokens, counts |
| `position_means.pt` | `[n_positions, hidden]` fp32, the per-position means the trainer subtracts |
| `filter_report.json` | keep rate, train/val topic counts, and every rejection with its first divergence |

## Decisions taken while implementing step 1

These are new; the plan does not cover them.

- **The forced response carries a full stop.** The instruction quotes the sentence without
  one, but the plan's 10-token count (S4.2b) includes `.`, so `DEFAULT_RESPONSE` is
  `The quick brown fox jumps over the lazy dog.` and there are 10 positions.
  `--response-text` overrides it, so **step 0 must measure whether the model actually emits
  the full stop** before the real extraction runs. See the finding below -- this is the
  live risk, not a hypothetical one.
- **Padding-aware `position_ids`.** A plain forward pass numbers RoPE positions with
  `arange(seq_len)`, so under left padding a topic's vectors would depend on which batch it
  landed in. The extractor derives positions from the attention mask instead. The reference
  implementation does not, which is one reason our baseline vectors may differ slightly from
  upstream's.
- **Means are written, not applied.** Vectors on disk are raw, so the centering choice
  (plan D2) is revisitable without re-extracting. **The trainer must subtract
  `position_means.pt` itself** -- if it forgets, arm B trains on uncentred vectors and
  nothing says so.
- **Means are over all surviving topics, train and val**, which is what upstream's
  extractor does. Do not "fix" this to train-only without also accepting that the 1.3662
  comparison gets weaker.
- **`positions.json` holds metadata; the means live in `position_means.pt`.** Plan S5.2 put
  the means in the JSON; 10 x 4096 floats of text is half a megabyte to parse for no gain.
  The vector index arithmetic is `start + position`, from `topics.json`, so no per-vector
  map is stored either.
- **The baseline style filters nothing**, so its `topics.json` holds all 49,637 topics while
  the pangram one holds only the compliant ones. **Whoever compares arms A/C against B must
  decide whether to intersect the topic sets.** Recommended: intersect, so an arm difference
  cannot be a topic-population difference. This is not yet implemented anywhere.
- **`--dataset-file`** reads the topics from a local JSONL copy of
  `keenanpepper/fifty-thousand-things` (the single file
  `wikipedia_vital_articles_level5_dataset.jsonl`). The vast remote's agent account has no
  network egress and the `hf-fetch.sock` daemon serves models only, so a remote extraction
  run needs the JSONL transferred by hand -- only `*.py` files sync automatically.

## Finding from the step-1 smoke run: the full stop is the main filter hazard

20 topics through Llama-3.2-1B-Instruct (the smoke model, not the 8B -- read this as a
signal about the *shape* of the failure, not as a keep-rate estimate) kept 6, and the
rejections group as `first_mismatch_histogram = {0: 2, 3: 5, 9: 7}`. Half of them diverge at
position 9, the full stop, and the model's preferred token there is `.\n` -- the sentence is
right, but the tokenizer merges the stop with the newline the model wants to write next.

Two consequences for step 0:

- Forcing `.` then `<|eot_id|>` rejects a model that produced the correct *string*. So the
  keep rate this filter reports is a lower bound on genuine compliance, and the gap is not
  small.
- The likely fixes are to drop the full stop from the canonical response (`--response-text
  "The quick brown fox jumps over the lazy dog"`, which makes it **9 positions, not 10**,
  and changes the plan's example-pool arithmetic), or to accept any tokenization whose
  decoded text matches. Teacher forcing fixes the tokenization, so the second option needs
  real code, not a flag. **Do not choose between these from the 1B; measure on the 8B.**

Divergence at position 0 (the model opens with something else entirely) and mid-sentence
divergence at position 3 are the other two modes seen, at lower rates.

## Tests

`pytest tests/test_extract_topic_vectors.py` -- 13 fast tests with a fake model and
tokenizer (prompt wording, padding, filter verdicts, split inheritance, contiguous index
ranges, per-position means, batch invariance). Three more are marked `hf_cache` and pin what
only real weights answer: the pangram is 10 tokens with the pinned decodings, batched
extraction matches unbatched, and the written artefacts have the right shapes. Run those
under the `claude` user (`gpu-exec`), because the HF cache is only readable there.

## Next: step 0, the probe

The plan (S6 step 0) wants it before any real run, and it now has one extra job:

1. **Does the model write the full stop?** Greedy-generate a few hundred topics with the
   pangram prompt and compare against both candidate responses. This decides
   `--response-text` for every later run.
2. Classify every non-compliant output into the S5.1 categories (quoted / preamble /
   trailing commentary / altered wording / refusal / other) and report the distribution.
   `filter_report.json`'s `first_mismatch_histogram` is a cheap proxy -- divergence at
   position 0 means the model never started the sentence, divergence at the last position
   means it started but did not stop -- but it is not a substitute for reading real
   generations.
3. Benchmark examples/second and peak memory across the S4.2 configurations.
4. Confirm the prefix-cache path reproduces the uncached loss (needs the step-2 trainer).
5. A ~50-step throwaway debug run of the arm-B config.

Throwaway `.tmp.py` scripts, per the project convention.

## Then: step 2, the trainer

Everything in plan S6 step 2 and D8 still stands. Points that step 1 changed or sharpened:

- Read `topics.json` + `vectors.pt` + `position_means.pt`; subtract the mean of a vector's
  own position; flatten each vector against every label of its topic; split by the topic's
  `split` field, never per vector.
- The budget is **examples seen** (755,391), not epochs, and the cosine schedule is laid
  over that budget.
- Before any training run, score the published upstream checkpoint
  (`keenanpepper/selfie-adapters-llama-3.1-8b-instruct`, `wikipedia-scalar-affine.safetensors`)
  through our loss path on the baseline val vectors and check it reproduces its recorded
  `best_val_loss` of 1.3662. If it does not, stop -- nothing downstream means anything. This
  needs the *baseline* extraction to have run first, which is why step 3 extracts both
  styles.
- `normalize_input: true`, `strip_labels: true`, target `label + '"' + <|eot_id|>`, batch
  256, lr 0.01, AdamW, cosine, 10 warmup steps, clip 0.5, init scale 5.0, seed 42.

## Known risks not yet retired

- The full stop question above. It is the single cheapest way for phase 0 to be wasted.
- Reproducing 1.3662 crosses trainers *and* now crosses extractors (the `position_ids`
  difference, and any batching difference against upstream's). If the check lands close but
  not exact, suspect extraction before suspecting the trainer.
- Extraction on the vast remote needs the topic JSONL moved by hand.
