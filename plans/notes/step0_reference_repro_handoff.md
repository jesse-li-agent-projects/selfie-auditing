# Reference-repro handoff: last mile of the 1.3662 investigation

Self-contained handoff for a fresh session (context was cleared after this
was written). Read this instead of re-deriving anything below.

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
