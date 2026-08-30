# Reference-repro handoff: last mile of the 1.3662 investigation

Self-contained handoff for a fresh session (context was cleared after this
was written). Read this instead of re-deriving anything below.

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

## What's still open: two candidate causes, both about the vectors

1. **Prompt construction.** Our `extract_baseline_vectors.py` uses each
   topic's own hand-written `prompt` field (from the
   `keenanpepper/fifty-thousand-things` dataset). The reference repo's
   `data_prep/wikipedia_topics/extract_wikipedia_vectors.py` instead builds
   a mechanical `f"Tell me about {title}."` prompt for every topic.
2. **`position_ids`.** Neither `extract_wikipedia_vectors.py` nor
   `extract_multilayer_vectors.py` ever passes `position_ids` to the model.
   Under left padding (used so the last real token sits at a fixed `-1`
   offset), HF's default falls back to a naive `arange(0, seq_len)` --
   every row in a batch gets positions `0..width-1` regardless of how much
   padding precedes its real tokens, so a topic's true last-token RoPE
   position is corrupted by whatever the longest sequence in its batch
   happened to be. Our own `extract_common.py::position_ids_from_mask`
   deliberately avoids this. If upstream's real training vectors have this
   defect baked in, our "corrected" vectors are systematically different
   activations than what the adapter's frozen bias was trained to expect.

**Important caveat found along the way:** none of the three YAML configs
shipped in `resources/selfie-adapters/training/configs/` (`scalar_affine_8b_
goodfire.yaml`, `scalar_plus_low_rank_8b.yaml`, `identity_baseline.yaml`)
actually train on Wikipedia topics -- all three point `labels_file` at
Goodfire SAE decoder vectors instead. There is **no recorded config** in this
repo snapshot for how `wikipedia-scalar-affine.safetensors` was actually
built. Everything above is reconstructed from `extract_wikipedia_vectors.py`
plus inference-time evidence, not from a confirmed source of truth. If the
next step (below) doesn't close the gap, this is the next thing to
interrogate -- the checkpoint may not have been trained on this exact
`fifty-thousand-things` population/layer/method at all.

One thing that's *already* fairly well confirmed and does NOT need
re-checking: the train/val split. `data_prep/wikipedia_topics/
dataset_generation/create_jsonl_splits.py` shuffles with `random.seed(42)`
and writes exactly `wikipedia_vital_articles_level5_dataset.jsonl` -- the
same filename `adapter_training/dataset.py::DEFAULT_DATASET_FILE` expects.
So our val split is very likely the same one the original run used.

## The next step (what to actually run)

A clean single-variable test: re-extract using the reference's own prompt
construction + missing-`position_ids` behavior, but keep our topic
population/labels/split identical (same `topics.json` order), so only the
two candidate variables above change. Both scripts for this already exist
and are synced to the remote:

- `reextract_mechanical_prompts.tmp.py` -- reads
  `outputs/baseline_l19/topics.json`, re-extracts layer 19 with
  `f"Tell me about {title}."` prompts and no `position_ids` (bug-for-bug
  matching `extract_wikipedia_vectors.py`), writes
  `outputs/baseline_l19_mechanical/` in the same schema as `baseline_l19/`.
- `reproduce_1p3662.tmp.py baseline_l19_mechanical` -- scores it, same as
  before.

### Exact commands

```
# From a fresh session, in this repo (NOT necessarily this worktree --
# .claude/worktrees/reference-repro-1p3662 already exists both locally and
# on the remote with everything below in place; reuse it, or copy its
# resources_selfie_adapters/ and outputs/baseline_l19/ into a new one).

# 1. Confirm the remote is up and see what's there already:
#    mcp__vast-remote-broker__list_instances   (expect instance "vai", label vai-0)
#    mcp__vast-remote-broker__remote_exec  command="nvidia-smi --query-gpu=name,memory.total --format=csv"

# 2. Run the re-extraction (~10-20 min at batch 32 on one 3090; no grad,
#    but is a full forward pass over all 49,637 topics):
cd /workspace/selfie_taboo/.claude/worktrees/reference-repro-1p3662
setsid nohup python reextract_mechanical_prompts.tmp.py > $HOME/reextract.log 2>&1 < /dev/null &
disown
# poll with: tail -c 2000 ~/reextract.log ; pgrep -af reextract_mechanical

# 3. Once it writes outputs/baseline_l19_mechanical/{vectors,position_means}.pt,
#    score it with the SAME driver that produced 1.7803 (~15-20 min, 84,211
#    val examples, batch 64, one 3090):
setsid nohup python reproduce_1p3662.tmp.py baseline_l19_mechanical > $HOME/repro_mechanical.log 2>&1 < /dev/null &
disown
# poll with: tail -c 2000 ~/repro_mechanical.log ; pgrep -af "python reproduce_1p3662"
```

### How to read the result

- **~1.3662** (or close, within ~0.02-0.10): confirms one or both of
  (prompt construction, missing `position_ids`) as the cause. Worth then
  testing them *separately* (mechanical prompt + correct position_ids; vs.
  our hand-written prompt + no position_ids) to attribute the fix, since
  `reextract_mechanical_prompts.tmp.py` currently changes both at once.
- **Still ~1.78, or somewhere else entirely**: both candidates are ruled
  out. Escalate to the user -- the likeliest remaining explanation is the
  "no recorded wikipedia config" caveat above: the checkpoint may not have
  been trained on this population/layer/method at all, and that needs a
  human decision (e.g. asking upstream, or accepting the gate failure and
  moving on without full reproduction).

## Remote environment notes (gotchas hit this session)

- Instance: vast-remote-broker alias `vai` (label `vai-0`), **single RTX
  3090, 24GB**. `HF_HOME=/workspace/hf_cache`.
- `resources/selfie-adapters` is gitignored and **not** auto-synced by the
  vast-remote-broker (only `.py` files in the main repo + anything under
  `.claude/worktrees/` sync). It was manually copied into this worktree as
  `resources_selfie_adapters/` (renamed from `selfie-adapters` to dodge the
  hyphen-in-package-name import issue) so it would sync -- already present
  on remote at `.../reference-repro-1p3662/resources_selfie_adapters/`.
- HF fetches on this remote only work for repos on `/etc/hf-model-allowlist.
  txt` (root-owned, agent can't edit), via the socket at `/run/hf-fetch.
  sock` (write repo id + `\n`, read one line back -- `DONE:`/`REFUSED:`/
  `ERROR:`). `meta-llama/Llama-3.1-8B-Instruct` and
  `keenanpepper/selfie-adapters-llama-3.1-8b-instruct` are both listed and
  already fetched (cached). **`keenanpepper/fifty-thousand-things` (the
  topics dataset) is NOT on the allowlist** -- don't try to fetch it; it's
  not needed anyway since `outputs/baseline_l19/topics.json` already has
  every topic's title/labels/split.
- The worktree directory synced from local (`.../reference-repro-1p3662/`)
  is **read-only** for the remote `agent` user (owned by `root:agent`,
  mode `640`) -- writing a log file or new output there fails with
  `Permission denied`. Write logs/outputs to `$HOME` (`/home/agent/`)
  instead, or a subdirectory under `outputs/` if the sync mechanism created
  it as writable (it did for `outputs/baseline_l19_mechanical/` when
  `reextract_mechanical_prompts.tmp.py` creates it via `mkdir`).
- `remote_exec` calls over ~120s auto-background as an MCP task in *that*
  session and notify on completion -- but that tracking does **not**
  survive a session ending or `/clear`. Always launch anything that takes
  more than ~2 min with `setsid nohup ... > logfile 2>&1 < /dev/null &
  disown`, and poll the log file with a fresh short `remote_exec` call
  instead of relying on the task notification.
