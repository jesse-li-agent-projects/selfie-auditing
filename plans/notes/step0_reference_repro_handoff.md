# Reference-repro handoff: last mile of the 1.3662 investigation

Self-contained handoff for a fresh session (context was cleared after this
was written). Read this instead of re-deriving anything below.

**2026-08-30 update #5: update #4 was wrong about where the bug is --
`adapter_training/loss.py` is not the problem, and the "mechanical prompt
fix" premise behind PR #48 is now in doubt.** An in-process A/B harness
(`ab_compare_loss.tmp.py`, rebased onto PR #49's `SoftPromptLoss` device
fix, run on remote `vai`) loaded the model, tokenizer, and published
checkpoint **once** and fed identical vectors + labels to both
`adapter_training.loss.SoftPromptLoss.__call__` and the reference's own
`training.model.SelfIEModel.compute_loss` in the same process -- eliminating
any chance the two paths were scoring different inputs.

- **64 examples, strided across the whole val split (many different
  topics), scored one at a time:** max `|ours - ref|` = 0.0156, no
  systematic sign (roughly half positive, half negative) -- bf16 rounding
  noise, not a bug. The loss math is exact.
- **256 examples, batch_size=1 vs. batch_size=32 (real padding, mixed
  target lengths), our own code only:** mean losses of 1.7309 vs 1.7305 --
  batching/masking introduces no discrepancy either.
- Both checks used `outputs/baseline_l19_v2` (the vectors update #4's gate
  scored at 1.7973). Batch-size-1 alone already predicted ~1.73, matching
  the full 84,211-example gate closely. **So our loss/eval code was never
  the bug** -- update #4's diagnosis (loss math, or batching at scale) is
  retracted.

**What actually differs: two nominally-identical "mechanical prompt"
vector extractions score completely differently through the *same,
verified-correct* loss code.** Re-running the identical script against
`/home/agent/outputs_mechanical/baseline_l19_mechanical` (the vectors that
scored 1.4065 through the reference's own driver in update #2) instead of
`baseline_l19_v2` (the vectors update #4's "official" gate used) gives
**1.317** -- close to the recorded 1.3662 -- through our own unmodified
code. Same checkpoint, same code, same topic population, wildly different
answer depending on which extraction directory is read.

Diffing the two directories' `topics.json` (49,637 entries each) found:
titles, order, `start`/`count`/`split`, and `labels` are **byte-identical**
between the two -- but the **`prompt` field differs on 26,042 / 49,637
entries (52.5%)**. `baseline_l19_mechanical` -- despite its name -- carries
the *grammar-cleaned* phrasing on those rows (`"Tell me about Earth's
gravity."`, `"Tell me about spiritual gifts."`, `"Tell me about
lenses."`), while `baseline_l19_v2` carries the literal, unmodified-title
mechanical template PR #48 was written to produce (`"Tell me about
Gravity of Earth."`, `"Tell me about Spiritual gift."`, `"Tell me about
Lens."`). Position-mean vectors for the two directories are nearly
identical to each other (cosine 0.999999) and each is ~5.6% off from
upstream's own published `mean-vectors.safetensors` (layer 19) --
raw-vector-population effects are not what's driving the ~0.4-nat loss gap
between the two directories; the prompt text is the only thing that
differs, and it happens to correlate almost exactly with which one
reproduces 1.3662.

**This inverts update #3's conclusion.** Update #3 asserted (reading
`data_prep/wikipedia_topics/extract_wikipedia_vectors.py:48-50`) that
upstream's `create_prompt()` is unconditionally `f"Tell me about
{title}."` over the *raw* Wikipedia title, and that PR #48's fix (making
our own extractor do the same) was the correct target. The evidence above
says the opposite: the grammar-cleaned phrasing -- not the literal raw
title -- is what reproduces the checkpoint's recorded loss. Either update
#3 misread which string upstream's `create_prompt()` actually receives as
`title` (e.g. upstream's own `title` field may itself already be a
cleaned/canonicalized noun phrase, not the raw Wikipedia article title our
`topics.json` stores under that key), or some other upstream preprocessing
step normalizes it before `create_prompt()` sees it.

**Recommended next step (cheap, no GPU):** re-read
`data_prep/wikipedia_topics/extract_wikipedia_vectors.py` and whatever
populates its `title` variable end to end -- not just the `create_prompt()`
line update #3 looked at -- to find where (if anywhere) grammar-cleaning
happens upstream, and compare it against how this repo's own `topic.prompt`
field (the grammar-cleaned one) was generated. Do not start
`plans/pangram_phase0_run.md` until this is resolved; per the plan's own
tolerance table, a 0.4-nat gap is stop-and-report regardless of cause.

**2026-08-30 update #4 (retracted by update #5 above, kept for history):
diagnosed the gap as our own loss/eval code, not vectors or extraction --
that diagnosis was itself an artifact of comparing the wrong two vector
directories, not a real code bug.** the fix was applied to our own extractor and the
gate still fails — the remaining gap is now isolated to our own loss/eval
code, not to vectors or extraction.** Update #3 established that prompt
construction alone explains the 1.7803 → 1.4065 move *through the
reference's own loss code*. This session applied the same fix to this
repo's own extractor (`adapter_training/extract_baseline_vectors.py` now
builds `f"Tell me about {title}."` from the raw title, matching upstream's
`create_prompt()` exactly — PR #48) and re-ran the actual
`plans/pangram_step0_benchmarks.md` gate through **this repo's own**
`adapter_training.evaluate_adapter`, not the reference-code driver script.

Result: **1.7973** — essentially unchanged from the pre-fix ~1.7800, against
the same recorded `best_val_loss` of **1.3662** (gap **0.431**, unchanged
within noise). This is the opposite of what update #3 would predict if our
own loss path were faithfully reproducing the reference's `compute_loss`.

Before concluding the extraction fix itself was somehow wrong, the new
vectors were sanity-checked directly against the old ones (both available:
old at local `outputs/baseline_l19/`, new at remote `vai-0`'s
`/home/agent/outputs_mechanical/baseline_l19_v2/`, index-aligned since both
were built from the same title order):

- Topic 0 ("William Wallace") happens to have an *identical* prompt under
  both the old hand-written field and the new mechanical template
  (`"Tell me about William Wallace."` either way). Its vector norm matches
  to bf16 rounding: 13.064 (old) vs 13.063 (new).
- Topic 1 ("Gravity of Earth") has a genuinely different prompt (old:
  `"Tell me about Earth's gravity."`; new: `"Tell me about Gravity of
  Earth."`). Its vector visibly moved: norm 12.913 (old) vs 13.048 (new).

So the extraction fix is doing exactly what it should — the vectors
demonstrably reflect the prompt-construction change. **The conclusion is
that this repo's own `adapter_training/loss.py` / `evaluate_adapter.py` /
`dataset.py` path has an independent bug that keeps its measured loss
pinned around 1.78–1.80 regardless of which vectors (hand-written-prompt or
mechanical-prompt) it scores** — a bug the reference repo's own loss code
does not share, since that code's score tracks the vector-quality
improvement almost exactly (1.7803 → 1.4065, a 0.374 drop) while ours does
not (1.7800 → 1.7973, no drop at all, if anything slightly worse).

This reframes the whole investigation. It is no longer "which vectors did
the checkpoint train on" (update #3 answered that: mechanical-prompt ones,
convincingly) — it is now "why does this repo's own loss/eval
implementation not reproduce the reference's `compute_loss` on vectors both
paths agree are correct." That is a **tractable, mechanical debugging task**
(diff `adapter_training/loss.py::SoftPromptLoss` against
`resources/selfie-adapters/training/model.py::SelfIEModel.compute_loss` and
`training/data.py::VectorLabelDataset` line by line — target construction,
injection positions, masking, aggregation, template rendering, label
smoothing/clamping), not an open-ended population/config mystery. It is
also a fundamentally different bug from the two fixed en route this
session:

- The extraction prompt-construction bug (this update's own fix, PR #48)
  — real, now fixed, and independently confirmed correct by the sanity
  check above.
- A latent CPU/GPU device-placement bug in `SoftPromptLoss.__init__`
  (`template_ids_tensor` was created with no `device=`, then indexed into a
  GPU-resident embedding table — crashes immediately the first time this
  class is constructed against a real GPU model; fixed in the same PR #48
  commit). This one is orthogonal to the loss-value discrepancy above — it
  was a hard crash, not a silent wrong number — but is worth knowing about
  since it means this loss path had apparently never been run end-to-end
  against a real (non-test-double) GPU model before this session.

Per `plans/pangram_step0_benchmarks.md`'s own tolerance table (>0.10 off →
stop and report; do not start phase 0), **this is a stop-and-report
result.** Do not start `plans/pangram_phase0_run.md`.

**Artefact locations for whoever picks this up:**
- New vectors: remote `vai-0`, `/home/agent/outputs_mechanical/baseline_l19_v2/`
  (NOT under `outputs/`, so **not** auto-synced back to local — same
  `outputs/` group-write workaround as update #2's `baseline_l19_mechanical`).
  Symlinked into `/workspace/selfie_taboo/outputs/baseline_l19_v2` on remote
  so `evaluate_adapter --vectors baseline_l19_v2` resolves it (that flag
  hardcodes an `outputs/` prefix).
- `topics.json` there was reassembled locally (title/labels/split/start/count
  copied verbatim from the old `outputs/baseline_l19/topics.json`, only
  `prompt` replaced) rather than transferred as a 57 MB blob through model
  context — transferred via the `.claude/worktrees/<name>/` full-file sync
  path, not the `*.py`-only main-repo sync. Verified byte-identical transfer
  (source and remote file sizes matched exactly).
- Eval report: the run's stdout (captured in `/home/agent/eval_gate3.log` on
  `vai-0`) has the full JSON; the `--report` file write itself crashed on a
  `PermissionError` creating `outputs/eval/` (same class of `outputs/`
  permission issue as update #2, just a different, not-yet-created
  subdirectory) — a minor loose end, not investigated further this session.
- Gate command used: `python -m adapter_training.evaluate_adapter --vectors
  baseline_l19_v2 --split val --center --batch-size 32 --checkpoint
  keenanpepper/selfie-adapters-llama-3.1-8b-instruct:wikipedia-scalar-affine.safetensors
  --report eval/upstream_published_centred_v2.json`. Batch size 32, not the
  256 default — 256 OOM'd on this rental's single 24 GB 3090 (this loss path
  materialises full-sequence logits per example unlike the cheap
  single-token baseline extraction; the plan's own benchmark table already
  documents needing a small micro-batch on this class of GPU for the
  trainer, which apparently extends to eval too).

**2026-08-30 update #3: the "two candidate causes" from update #2 collapse to
one. `position_ids` was independently proven inert in PR #43
(`worktree-fix-position-ids-reproduction`) — naive and padding-aware
position_ids produce bit-identical activations under RoPE, because the
causal mask excludes padding from attention and RoPE scores only depend on
the relative offset between query and key. It was never a live source of
the gap. That means the entire 1.7803 → 1.4065 improvement from
mechanical-prompt re-extraction is attributable to prompt construction
alone — no separate attribution experiment is needed. Also corrected: the
prompt-construction difference itself was mischaracterized (see the
rewritten "Prompt construction" point below) — it is not "hand-written vs.
mechanical template," it's "template with a grammar-cleaned noun phrase vs.
template with the raw, unmodified title." See "2026-08-30 update #3
detail" near the bottom for the full writeup.**

**2026-08-30 update #2: scoring finished and was confirmed by reading
`$HOME/repro_mechanical.log` on `vai` directly (the earlier MCP timeout was
transient — a fresh session's `remote_exec` worked immediately). Result:
mechanical-prompt + no-`position_ids` vectors score `1.4065` against the
`1.3662` target — gap narrowed from `0.4141` (1.7803 vs 1.3662) to `0.0403`.
This falls in the doc's "close" bucket: confirms prompt-construction and/or
missing-`position_ids` as (at least partially) the cause.**

**2026-08-30 update (superseded above): re-extraction is DONE, scoring was
launched but its result is unconfirmed — see "Where this actually stands
right now" below, which supersedes the "next step" section further down for
what to do first.**

## What this is chasing

`plans/notes/step0_findings.md` found our own loss path scores the published
`wikipedia-scalar-affine.safetensors` checkpoint at **1.7800** against its
recorded `best_val_loss` of **1.3662** -- a 0.41-nat gap, way over the
plan's 0.10 stop-and-report threshold.

## What's now PROVEN (this session, on remote `vai`, single RTX 3090 24GB)

Ran the reference repo's own `training/model.py::SelfIEModel.compute_loss` +
`training/data.py::VectorLabelDataset` -- zero lines of this repo's
`adapter_training/` -- against the published checkpoint, scored on **our**
existing `outputs/baseline_l19` vectors (49,637 topics, hand-written
per-topic prompts, position-id-corrected extraction). Result: **1.7803**,
matching our own number almost exactly.

**Conclusion: the loss/training code is not the bug.** Neither implementation's
eval math explains the gap. It's the *vectors* (how they were extracted) or
the *population* they were extracted from that differs from whatever the
checkpoint actually trained against.

Driver script (works, already produced the number above):
`reproduce_1p3662.tmp.py <vectors-subdir-name>` (defaults to `baseline_l19`;
pass e.g. `baseline_l19_mechanical` to score a different extraction). It
downloads the checkpoint via `hf_hub_download` (already cached), loads it
into a fresh reference `SelfIEModel`, and runs a `no_grad` pass over the val
split, printing a running average every 50 batches and the final number at
the end.

## What's still open: prompt construction (position_ids is resolved, see below)

1. **Prompt construction.** Corrected in update #3 below — see that section
   for the actual difference. (Originally mischaracterized here as
   "hand-written prompts vs. a mechanical template"; that was wrong.)
2. **`position_ids` — RESOLVED, not a factor.** PR #43
   (`worktree-fix-position-ids-reproduction`) tested this directly against
   the real model: naive `arange` position_ids (what
   `extract_wikipedia_vectors.py` / `extract_multilayer_vectors.py` produce
   under left padding, by never setting `position_ids` at all) and
   padding-aware `position_ids_from_mask` give **bit-identical
   activations**. RoPE attention scores depend only on the relative offset
   between query and key, and padding is excluded from attention by the
   causal mask, so the constant per-row shift a naive arange introduces
   cancels out completely. This was never a live source of the gap, in
   either direction. `position_ids_from_mask` keeps running unconditionally
   in this repo's own extraction as defense-in-depth, not because it fixes
   anything measurable.

**Important caveat found along the way:** none of the three YAML configs
shipped in `resources/selfie-adapters/training/configs/` (`scalar_affine_8b_
goodfire.yaml`, `scalar_plus_low_rank_8b.yaml`, `identity_baseline.yaml`)
actually train on Wikipedia topics -- all three point `labels_file` at
Goodfire SAE decoder vectors instead. There is **no recorded config** in this
repo snapshot for how `wikipedia-scalar-affine.safetensors` was actually
built. Everything above is reconstructed from `extract_wikipedia_vectors.py`
plus inference-time evidence, not from a confirmed source of truth. If the
remaining ~0.04 gap (see update #3) doesn't close further, this is the next
thing to interrogate -- the checkpoint may not have been trained on this
exact `fifty-thousand-things` population/layer/method at all.

One thing that's *already* fairly well confirmed and does NOT need
re-checking: the train/val split. `data_prep/wikipedia_topics/
dataset_generation/create_jsonl_splits.py` shuffles with `random.seed(42)`
and writes exactly `wikipedia_vital_articles_level5_dataset.jsonl` -- the
same filename `adapter_training/dataset.py::DEFAULT_DATASET_FILE` expects.
So our val split is very likely the same one the original run used.

## Where this actually stands right now (2026-08-30 session)

**Re-extraction is done and confirmed good.** `reextract_mechanical_prompts.tmp.py`
ran to completion on remote `vai` and wrote vectors — but NOT to
`outputs/baseline_l19_mechanical/` as the script originally assumed. Two
bugs were found and fixed this session:

1. **`outputs/` on the remote instance is unwritable by the `agent` user.**
   It's `root:agent`, mode `0750` — no group-write. The *first* re-extraction
   attempt ran the full ~10-minute forward pass over all 49,637 topics, then
   crashed at the final `OUT_DIR.mkdir(...)` with `PermissionError`,
   **losing all of that compute** (nothing had been saved yet). Root cause:
   this worktree's local `outputs/` dir was missing the group-write bit its
   sibling (the main repo's `outputs/`, `2775`) has — it was `2755` instead.
   That propagated to the remote sync as root-owned/agent-read-only.
   - **Fixed locally**: `chmod g+w` applied recursively to this worktree's
     local `outputs/` (now matches `2775`). This fix has **NOT been
     confirmed synced to the remote** (a `sync_flush` call was interrupted
     before completing) — check/redo if anything needs to write directly
     into remote `outputs/` again.
   - **Fixed in the scripts** (workaround, still in place, gitignored
     `*.tmp.py` so check the `reference-repro-1p3662` worktree, not git
     history): `reextract_mechanical_prompts.tmp.py`'s `OUT_DIR` now points
     at `Path.home() / "outputs_mechanical" / "baseline_l19_mechanical"`
     (agent-writable) instead of under `outputs/`, and it now calls
     `OUT_DIR.mkdir(...)` *before* the forward pass, not after, so a save
     failure can't lose completed compute again.
     `reproduce_1p3662.tmp.py`'s `VECTORS_DIR` now falls back to
     `Path.home() / "outputs_mechanical" / <name>` if `outputs/<name>`
     doesn't exist.
   - **This is a memory-worthy pattern** (see project memory
     `surface-tool-misconfiguration.md`) — this exact failure (find a
     permission/env issue, silently route around it, mention only in
     passing) happened twice now. Flag issues like this to the user
     explicitly and immediately, not folded into a task summary.

2. **Result confirmed**: `outputs_mechanical/baseline_l19_mechanical/` on
   remote `vai` (under `/home/agent/`, NOT under the synced worktree path)
   contains `vectors.pt`, `position_means.pt`, `topics.json`,
   `positions.json` — same schema as `outputs/baseline_l19/`. Log tail
   confirmed: `Wrote /home/agent/outputs_mechanical/baseline_l19_mechanical/
   {vectors,position_means}.pt, topics.json, positions.json`.

**Scoring completed and was confirmed** (see "update #2 result" below).

## 2026-08-30 update #2 result: mechanical-prompt vectors score 1.4065

`repro_mechanical.log` on `vai` completed cleanly (final batch 1300, no
crash, no orphaned process):

```
Scoring vectors from: /home/agent/outputs_mechanical/baseline_l19_mechanical
val: 84211 examples from 4964 topics
  ...
  batch 1300: running avg loss = 1.4072

Measured val loss (reference code, our vectors): 1.4065
Published best_val_loss: 1.3662405303030303
```

Gap is now **0.0403** (down from **0.4141** using our hand-written-prompt,
position-id-corrected `baseline_l19` vectors, which scored 1.7803/1.7800).

## 2026-08-30 update #3 detail: prompt construction is the whole story, and it's narrower than first described

`reextract_mechanical_prompts.tmp.py` changed two things at once (prompt
construction, `position_ids`). Update #3 above establishes `position_ids`
is provably inert (PR #43), so **prompt construction alone** explains the
full 1.7803 → 1.4065 move. No separate attribution run is needed.

But the prompt-construction difference itself was overstated when first
written. Checked directly against `outputs/baseline_l19/topics.json`
(49,637 topics) and the reference's `create_prompt()` in
`data_prep/wikipedia_topics/extract_wikipedia_vectors.py:48-50`:

- **Reference**: `create_prompt(title)` is unconditionally
  `f"Tell me about {title}."`, using the raw Wikipedia article title
  string, for every one of its topics.
- **Ours**: every topic's `prompt` field *also* follows the
  `"Tell me about ___."` template — there is no different sentence
  structure — but the noun phrase slotted in is sometimes a grammar-cleaned
  rewrite rather than the raw title:
  - **23,595 / 49,637 (47.5%)** match the reference exactly:
    `f"Tell me about {title}."` verbatim.
  - **26,042 / 49,637 (52.5%)** use the same template with an adjusted noun
    phrase instead of the raw title, e.g. `title='Gravity of Earth'` →
    `"Tell me about Earth's gravity."` (reworded to possessive),
    `title='Structural formula'` → `"Tell me about structural formulas."`
    (pluralized), `title='High Atlas'` → `"Tell me about the High Atlas."`
    (added "the").

So the real difference is **template-with-cleaned-noun-phrase vs.
template-with-raw-title**, affecting about half the topic population — not
"custom prompts vs. a mechanical template" as originally written. The
remaining ~0.04-nat gap between our mechanical-prompt re-extraction (1.4065)
and the published `best_val_loss` (1.3662) is plausibly residual noise, or
some smaller remaining difference; it is not large enough to warrant
further prompt-construction investigation before checking other candidates
(e.g. the no-recorded-config caveat above) if it needs closing further.

## Remote environment notes (gotchas hit this session)

- Instance: vast-remote-broker alias `vai` (label `vai-0`), **single RTX
  3090, 24GB**. `HF_HOME=/workspace/hf_cache`.
- `resources/selfie-adapters` is gitignored and **not** auto-synced by the
  vast-remote-broker (only `.py` files in the main repo + anything under
  `.claude/worktrees/` sync). It was manually copied into the
  `reference-repro-1p3662` worktree as `resources_selfie_adapters/`
  (renamed from `selfie-adapters` to dodge the hyphen-in-package-name
  import issue) so it would sync -- already present on remote at
  `.../reference-repro-1p3662/resources_selfie_adapters/`.
- HF fetches on this remote only work for repos on `/etc/hf-model-allowlist.
  txt` (root-owned, agent can't edit), via the socket at `/run/hf-fetch.
  sock` (write repo id + `\n`, read one line back -- `DONE:`/`REFUSED:`/
  `ERROR:`). `meta-llama/Llama-3.1-8B-Instruct` and
  `keenanpepper/selfie-adapters-llama-3.1-8b-instruct` are both listed and
  already fetched (cached). **`keenanpepper/fifty-thousand-things` (the
  topics dataset) is NOT on the allowlist** -- don't try to fetch it; it's
  not needed anyway since `outputs/baseline_l19/topics.json` already has
  every topic's title/labels/split.
- The synced worktree directory is **read-only** for the remote `agent`
  user (owned by `root:agent`, mode `640`) -- writing a log file or new
  output there fails with `Permission denied`. Write logs/outputs to
  `$HOME` (`/home/agent/`) instead, or a subdirectory under `outputs/` if
  the sync mechanism created it as writable.
- `remote_exec` calls over ~120s auto-background as an MCP task in *that*
  session and notify on completion -- but that tracking does **not**
  survive a session ending or `/clear`. Always launch anything that takes
  more than ~2 min with `setsid nohup ... > logfile 2>&1 < /dev/null &
  disown`, and poll the log file with a fresh short `remote_exec` call
  instead of relying on the task notification.
