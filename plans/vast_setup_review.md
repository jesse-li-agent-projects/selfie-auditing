# Review: Vast.ai setup for selfie_taboo

Review of the draft in `tmp/vast_tmp/` (a static copy of the locked-down setup
directory, which lives outside this repo and outside the agent sandbox). Diffed
against its origin, `/work/ml/toy_probe_hiding/vast_setup/`. Requirements come
from `selfie_taboo_pipeline.md` §7-8.

Only three files carry edits so far: `create_instance.py` (genericized alias,
new SSH state dir, `copy_local_runs()` removed), `sync_vastai.py` (host +
remote dir), and `CLAUDE.md`. `remote_setup.sh`, `destroy_vtao_safely.py`,
`bashrc_remote.sh` and `README.md` are byte-identical to the toy_probe_hiding
versions, so everything project-specific in them is still stale.

**Caveat on this review's evidence.** The real setup directory
(`/home/jesse/ml_secret/vast_setups/selfie_taboo`) is unreadable from the agent
sandbox, so this reviews the static copy. Some repo paths (`references`,
`.mcp.json`, several dotfiles) are also masked from the sandbox, so the
sync-ignore advice in §3.1 is reasoned from the code, not from a directory
listing. `mutagen` is not installed in the sandbox either — §3.1 gives a
command to confirm the one claim that needs it.

---

## 1. Blockers — these break the first rental

### 1.1 `REMOTE_PROJECT_DIR` disagrees between the two scripts

`create_instance.py:55` sets `/workspace/project`. `sync_vastai.py:103` sets
`/workspace/projects` (plural). Nothing reconciles them.

`remote_setup.sh` receives `PROJECT_DIR` from `create_instance.py`, so
provisioning builds `/workspace/project`: the mode-600 `.env`, the setgid
`runs/` directory, and the agent group bit all land there. Mutagen then pushes
source to `/workspace/projects`, a different directory that root creates
implicitly and the agent cannot read. `bashrc_remote.sh:143` does
`cd "${PROJECT_DIR}"`, which lands logins in the directory that has no source.

Pick one name and give it one definition. `/workspace/selfie_taboo` reads
better against the instance label than either current value.

### 1.2 The agent gets no `HF_HOME`, and the agent has no network

This is the sharpest problem, because the lockdown causes it and the plan does
not yet account for it.

The chain:

1. The seccomp filter (`remote_setup.sh:235-281`) denies `AF_INET`/`AF_INET6`
   sockets for the agent and for everything it execs.
2. `AGENT_SHELL` (`remote_setup.sh:287-301`) exports `HOME`, `TMPDIR`,
   `PYTHONPYCACHEPREFIX`, `MPLCONFIGDIR` and `TRITON_CACHE_DIR` — but not
   `HF_HOME`.
3. The forced command never sources any bashrc, so the `.env` route that plan
   §8 assumes for `HF_HOME` does not reach the agent at all.
4. The base model, the taboo LoRAs and the 1B smoke model are all gated on
   Hugging Face.

Result: the agent runs the pipeline, misses the cache, tries to download, and
stalls on a blocked socket. The failure gives no clear cause.

**The design consequence: the model download becomes a provisioning step, not
a runtime step.** Concretely:

- Root prefetches the models in `remote_setup.sh`, using `HF_TOKEN` (§2.2).
- The cache goes under `/workspace` (per plan §8), group-readable by `agent`.
- `AGENT_SHELL` exports `HF_HOME`, and also `HF_HUB_OFFLINE=1` so a cache miss
  fails immediately with a clear message instead of hanging.
- Add a positive control to `verify_agent_lockdown()`: the agent can read
  `HF_HOME` and resolve one model offline. This mirrors the existing
  runs/-writable control at `remote_setup.sh:389-392`, and it exists for the
  same reason — plan §7 says human iteration time on a rented box is the main
  cost.

**Make the prefetch idempotent.** The `UV_CACHE_DIR` comment at
`remote_setup.sh:23-28` exists because `--start-stage environment` is a
supported rerun path. An unconditional 16 GB pull on every rerun is exactly the
failure that comment was written to prevent.

### 1.3 `REPO_ROOT = SCRIPT_DIR.parent` is wrong once the directory moves out

`sync_vastai.py:98` derives the local project root from the script's own
location. That held when the setup directory was nested inside the repo. In the
locked-down location, `SCRIPT_DIR.parent` is `.../vast_setups/` — the parent of
*every* project's setup directory.

Two consequences: `_create_source()` hands mutagen `vast_setups/` as the alpha
side, and the runs pull writes into `vast_setups/runs/`.

`create_instance.py:114` has the same root cause:
`load_dotenv(SCRIPT_DIR / "../.env")` worked through the `.env -> ../.env`
symlink that exists in the toy directory and does not exist in the new one.

Fix: add an explicit `LOCAL_PROJECT_DIR` to the "Key parameters to change in
new project" block in both scripts, pointing at `/work/ml/selfie_taboo`. Never
derive it from `SCRIPT_DIR` again — that is precisely the coupling the lockdown
removes.

Also decide which `.env` the user populates, and say so in the README. This
repo's `.env` currently holds only `HF_TOKEN`. Keeping the secrets file beside
the locked-down scripts is the better choice: it puts it outside the agent's
reach, which is the point of the move.

### 1.4 `SOURCE_NAME` is still `toy-probe-source`

`sync_vastai.py:124`. Mutagen session names are global to the machine, and
`start()` (`sync_vastai.py:331-335`) returns early when a session of that name
already exists — it does not check what that session points at.

So if a toy_probe_hiding sync is live, selfie_taboo's `start` adopts it and
reports success while syncing the other project's tree. Rename to
`selfie-taboo-source`.

---

## 2. Gaps against plan §7-8

### 2.1 Disk default is below the plan's floor

`create_instance.py:43` sets `DEFAULT_DISK_GB = 40`. Plan §7 says budget at
least 60 GB. Suggest 80: ~16 GB of weights, HF cache overhead, torch wheels and
the uv cache. Vast.ai disk cannot be resized after creation, so this must be
right at create time, not fixed later.

### 2.2 `HF_TOKEN` is not passed anywhere

Plan §8 requires it as a per-session secret, handled the way `WANDB_API_KEY`
is. Touchpoints: a `get_hf_token()` beside `get_wandb_key()`
(`create_instance.py:432-439`, reading `HF_TOKEN_VASTAI` then falling back to
`getpass`), the env passing in `run_remote_setup()`
(`create_instance.py:463-469`), and the `.env` write in `remote_setup.sh:193-198`.

Keep it in the mode-600 root-owned `.env`. The agent must not read it — and
`verify_agent_lockdown()` already asserts that. That constraint is exactly why
the prefetch in §1.2 is mandatory rather than optional.

### 2.3 The package list is for the wrong project

`remote_setup.sh:177-191`. Missing for this pipeline: `transformers`, `peft`
(the LoRA hot-swap in plan §6), `accelerate`, `safetensors`, `huggingface_hub`.
Present but unused: `torch-optimi` (training only), `sympy`, and `wandb` — see
§4.2.

### 2.4 `useful_commands.sh` is referenced but absent

`README.md:3` names it; it was not copied into `vast_tmp`. Either drop the
mention or bring the file over. If it comes over, its offer query needs a real
VRAM floor — the toy version has `gpu_total_ram >=1`, and plan §7 wants 24 GB —
and its `disk_space` filter should match whatever §2.1 settles on.

---

## 3. Stale toy_probe_hiding content

### 3.1 The source-sync ignore list is a blocklist, not an allowlist

The module docstring (`sync_vastai.py:79-85`) and `CLAUDE.md`'s Secrets section
both state that the source session syncs only `*.py` plus `configs/`. Reading
the actual call in `_create_source()` (`sync_vastai.py:157-201`), that looks
wrong. There is no leading `--ignore '*'` to deny everything first — just a
list of ignored paths, then `!*.py`, `!configs/**`, `!/.claude/worktrees/` and
`!/.claude/skills/` negations.

Two pieces of static evidence that the negations are doing nothing:

- Nothing in the list ignores `.claude` at all, so the two `.claude` negations
  cannot be re-including anything.
- If it really were an allowlist, the entire blocklist (`/runs`, `/plot`,
  `personal`, `plans`, `references`, `tmp`, `analytic_feasibility`) would be
  redundant — none of those would sync anyway. A carefully maintained blocklist
  sitting next to the negations is evidence that the blocklist is what works.

**This decides how much care §3.2 needs, so confirm it first.** `mutagen` is
not installed in the sandbox. Run a source session against a scratch target and
check whether a `.md` file transfers, or read the resolved spec with
`mutagen sync list --long <name>`.

If it is a blocklist, then it is the only thing between `human_only/` and a
rented GPU box, and the whole list needs an audit rather than three added
lines. Fix the docstring and `CLAUDE.md` to match whichever way it goes.

### 3.2 The ignore list names the wrong project's directories

`/vast_setup` (`sync_vastai.py:172-173`) and `analytic_feasibility` no longer
exist here. Missing, and important: `human_only`, `proj_utils` and `vast` — all
three are absolute symlinks into `/home/jesse/ml_secret/`.

Ignore all three explicitly, and add `--symlink-mode=ignore` as well. Mutagen's
default portable symlink mode will probably refuse absolute symlinks anyway,
but do not rely on that: `human_only/` reaching a rented box is exactly the
disclosure the lockdown exists to prevent, so it should be blocked by the
ignore list, not by a default. Add `resources/` too (read-only reference
documents, no reason to ship them).

### 3.3 `destroy_vtao_safely.py` fails unsafe

Two stale values, and they differ in kind — the second one costs work, not just
money.

- `alias_and_label_for_index()` (`destroy_vtao_safely.py:84-88`) hardcodes
  `vtao`, and duplicates a derivation that now lives in
  `create_instance.py:82-92`. `find_instance_id` returns `None`, the script
  exits 1 and destroys nothing. Fails safe — it just costs rental time.
- `running_job_count()` (`destroy_vtao_safely.py:74-81`) greps for
  `train_adversarial_logreg.py`, which does not exist in this project. It
  returns 0 on the very first poll, the script reads that as "all work appears
  finished", and it destroys the box mid-pipeline. **Fails unsafe.** Point it
  at `run_pipeline.py` (plan §5).

Also: `sync_flush()` (`destroy_vtao_safely.py:68`) shells out to
`python sync_vastai.py`, relative to the working directory. Use `sys.executable`
and `SCRIPT_DIR / "sync_vastai.py"`. And the file's own name plus its docstring
should stop saying `vtao`; the docstring's talk of `--ckpt-interval` and lost
training iterations does not apply to an inference-only pipeline.

### 3.4 Stale documentation

- `README.md` still says `~/.ssh/toy_probe_hiding_vastai/`, alias `vtao`, and
  `sync_vastai.sh` (the file is `.py`).
- `CLAUDE.md` still says `vtao-N`, `/workspace/toy_probe_hiding`, and
  `sync_vastai.sh`.
- `sync_vastai.py:26-28` lists `create_instance.py's copy_local_runs()` as a
  prerequisite. That function was deleted.
- `create_instance.py:20-21` says it prompts when `WANDB_API_KEY` is unset; the
  code reads `WANDB_API_KEY_VASTAI`. Moot if §4.2 goes ahead.

---

## 4. Decisions for you

### 4.1 `ALIAS_BASE_NAME = "vai"` makes the index namespace global

This is a constraint of the genericization, not a bug — but it is worth stating
before two rentals collide. The alias base, the SSH state directory
(`~/.ssh/vastai_projects/`), the vast.ai label space and the agent key are now
all shared across projects. So selfie_taboo and toy_probe_hiding cannot both
sit at index 0: they would fight over the label and over
`~/.ssh/vastai_projects/vai.sshconfig`.

The comment at `create_instance.py:62-66` still says the alias "should be
unique per project", which no longer holds. Either set a per-project base
(`stab`, say) or update that comment to describe the global index namespace.
The shared agent key across projects is a reasonable tradeoff and needs only a
note.

### 4.2 Drop wandb?

This pipeline does no training (plan §7: "No training is needed anywhere ...
The whole pipeline is inference only"). Dropping wandb makes `HF_TOKEN` the
single per-session secret and removes a whole limb from the setup. It is a
coherent multi-file change, so do it in one pass:

- `remote_setup.sh:20` — the `WANDB_API_KEY` requirement guard
- `remote_setup.sh:193-198` — the `.env` write
- `remote_setup.sh:287-301` — `WANDB_MODE` / `WANDB_DIR` in `AGENT_SHELL`
- `remote_setup.sh:177-191` — the package list
- `create_instance.py:432-439` — `get_wandb_key()`
- `create_instance.py:463-469` — the env passing into `run_remote_setup`
- `create_instance.py:20-21` — the docstring
- `CLAUDE.md` — the Secrets section

Keep the `.env`-unreadable check in `verify_agent_lockdown()` either way. It
just protects `HF_TOKEN` instead, which is a durable account credential and so
matters at least as much.

### 4.3 Should `AGENT_SHELL` activate the venv?

`verify_agent_torch_gpu()` works only because it sources `activate` explicitly
(`create_instance.py:581`). The agent's forced command puts nothing on `PATH`.
Since `AGENT_SHELL` needs an edit for `HF_HOME` anyway, decide this at the same
time. If you leave it out, the interface document (§5) has to carry the
activation line verbatim.

---

## 5. New artifact needed: an in-repo interface document

The lockdown makes the setup scripts unreadable to agents, so the interface has
to be written down somewhere agents *can* read — otherwise every future agent
guesses. Suggest a short `vast/README.md` (or similar) checked into this repo,
covering:

- the remote project directory, and the agent SSH alias to use
- **which file types sync and which silently do not** — see §3.1
- where results land locally, and how to force a flush
- "the box has no network egress for the agent": no `pip install`, no HF
  download, no `git`

The sync gap is the one that will actually bite. Plan §5's layout is all `.py`,
so it is fine. But §4.7's fixed transcript set and §4.6's word list will vanish
silently if they land as `.txt` or `.json` and the list stays an allowlist.

---

## 6. Minor

- The hidden-state cache (plan §4.4, all layers × all positions) is roughly
  8 MB per prompt at 33 × 4096 × bf16. Small enough that the 10-second rsync
  pull loop does not need a separate exclude.
- `bashrc_remote.sh:102-115` carries a conda-init block from the vast.ai
  pytorch template. The current `TEMPLATE_HASH` is the CUDA-only image, so
  those paths do not exist. It is guarded, so this is cosmetic.
