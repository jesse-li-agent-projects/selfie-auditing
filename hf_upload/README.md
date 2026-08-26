# Uploading the dummy weight fixtures to the Hub

Two public repos back the end-to-end test described in
`plans/replace_smoke_flag_with_dummy_weights.md`. This directory holds their
model cards. The weights themselves are not committed here — they regenerate
from seed 0, so they are rebuilt rather than stored in git (the LoRA alone is
45 MB). Reproduction is per-device; see step 1.

Substitute your own account for `<hf-user>` throughout.

> [!IMPORTANT]
> The `HF_TOKEN` in `.env` is fine-grained **read-only**. It cannot create or
> write repos. Mint a separate write token for these two repos at
> <https://huggingface.co/settings/tokens>, use it for the upload, and then
> discard it. Do not put a write token in `.env`.

## 1. Generate the weights

No GPU is needed — this only reads the embedding matrix and initializes LoRA
tensors. **Generate on CPU**, and treat CPU as the canonical device for this
fixture; see the reproducibility note below for why.

```bash
date
python make_smoke_weights.py --output-dir outputs/dummy_weights --device cpu --seed 0
echo "=== generated ==="
find outputs/dummy_weights -type f -exec ls -l {} \;
```

Expected output, verified on 2026-08-26:

```
Wrote 2048-dim random SelfIE adapter (scale 0.933) to outputs/dummy_weights/selfie-random-scalar-affine.safetensors
Wrote random taboo LoRA for 'banana' to outputs/dummy_weights/taboo_lora/banana
```

| File | Size | Notes |
|---|---|---|
| `selfie-random-scalar-affine.safetensors` | 8.5 KB | `bias` (2048,) + `log_scale` (1,), both fp32; metadata `model_dim=2048`, `scalar_affine`, `init_scale≈0.9333` |
| `taboo_lora/banana/adapter_config.json` | 1.1 KB | `r=16`, `lora_alpha=32`, `init_lora_weights=false`, 7 target modules |
| `taboo_lora/banana/adapter_model.safetensors` | 45 MB | 224 tensors, all fp32 |
| `taboo_lora/banana/README.md` | 5.2 KB | PEFT's generated card — **overwritten in step 2** |

### Reproducibility is per-device, not absolute

Seed 0 fixes the random draws, but the SelfIE adapter is **not** byte-identical
between a CPU and a CUDA run. `init_scale` is the median L2 norm of the base
model's embedding rows, and that reduction differs in the last bits by device:

| device | `init_scale` |
|---|---|
| cpu | `0.9332588315010071` |
| cuda | `0.9332589507102966` |

`bias` is scaled by `init_scale`, so it shifts with it. The relative difference
is ~1e-7 and numerically irrelevant, but it does mean "regenerate and compare
hashes" only works against a matching device. Generate on CPU so the published
bytes are reproducible on any machine. The exact value is recorded in the
checkpoint's own metadata header either way.

The LoRA (`adapter_model.safetensors`) **is** byte-identical across devices —
verified — because nothing in it derives from a reduction over model weights.

If you generate on a machine with no network access, `peft` prints two warnings
about being unable to fetch `config.json` from the base model repo, ending in
*"will assume that the vocabulary was not modified."* That is correct here —
nothing in this fixture resizes the vocabulary, and `adapter_config.json` has
`modules_to_save: null`. On a networked machine the warnings do not appear.

## 2. Assemble the two repo directories

The model cards in this directory replace PEFT's generated one.

```bash
date
mkdir -p outputs/hf_staging/dummy-selfie-adapter-llama-3.2-1b \
         outputs/hf_staging/dummy-taboo-lora-llama-3.2-1b-banana

cp hf_upload/dummy-selfie-adapter-llama-3.2-1b/README.md \
   outputs/dummy_weights/selfie-random-scalar-affine.safetensors \
   outputs/hf_staging/dummy-selfie-adapter-llama-3.2-1b/

cp outputs/dummy_weights/taboo_lora/banana/adapter_config.json \
   outputs/dummy_weights/taboo_lora/banana/adapter_model.safetensors \
   outputs/hf_staging/dummy-taboo-lora-llama-3.2-1b-banana/
cp hf_upload/dummy-taboo-lora-llama-3.2-1b-banana/README.md \
   outputs/hf_staging/dummy-taboo-lora-llama-3.2-1b-banana/README.md

echo "=== staged ==="
find outputs/hf_staging -type f | sort
```

Each directory should now hold exactly:

```
dummy-selfie-adapter-llama-3.2-1b/README.md
dummy-selfie-adapter-llama-3.2-1b/selfie-random-scalar-affine.safetensors
dummy-taboo-lora-llama-3.2-1b-banana/README.md
dummy-taboo-lora-llama-3.2-1b-banana/adapter_config.json
dummy-taboo-lora-llama-3.2-1b-banana/adapter_model.safetensors
```

## 3. Create and upload

Public, not private. The weights are random and hold nothing to protect, and a
private repo would mean plumbing a token into the test environment — a
dependency and a failure mode the public path does not have.

```bash
date
export HF_TOKEN=<your write token>

hf repo create <hf-user>/dummy-selfie-adapter-llama-3.2-1b     --repo-type model
hf repo create <hf-user>/dummy-taboo-lora-llama-3.2-1b-banana  --repo-type model
echo "=== repos created ==="

hf upload <hf-user>/dummy-selfie-adapter-llama-3.2-1b \
    outputs/hf_staging/dummy-selfie-adapter-llama-3.2-1b . --repo-type model
hf upload <hf-user>/dummy-taboo-lora-llama-3.2-1b-banana \
    outputs/hf_staging/dummy-taboo-lora-llama-3.2-1b-banana . --repo-type model
echo "=== uploaded ==="
```

On an older `huggingface_hub`, the command is `huggingface-cli` rather than
`hf`, with the same subcommands.

## 4. Verify, then unset the write token

```bash
date
unset HF_TOKEN
python - <<'PY'
from huggingface_hub import hf_hub_download
from peft import PeftConfig
from safetensors import safe_open

HF_USER = "<hf-user>"

p = hf_hub_download(f"{HF_USER}/dummy-selfie-adapter-llama-3.2-1b",
                    "selfie-random-scalar-affine.safetensors")
print("adapter metadata:", safe_open(p, "pt").metadata())

cfg = PeftConfig.from_pretrained(f"{HF_USER}/dummy-taboo-lora-llama-3.2-1b-banana")
print("lora r/alpha/init:", cfg.r, cfg.lora_alpha, cfg.init_lora_weights)
PY
```

Both must succeed with `HF_TOKEN` unset — that is the check that the repos are
genuinely public. Expect `model_dim: '2048'` and `16 32 False`.

## 5. Wire the names in

Set these in `config.py`, beside the real 8B constants (plan step 2):

```python
DUMMY_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DUMMY_ADAPTER_REPO = "<hf-user>/dummy-selfie-adapter-llama-3.2-1b"
DUMMY_ADAPTER_FILE = "selfie-random-scalar-affine.safetensors"
DUMMY_LORA_REPO_TEMPLATE = "<hf-user>/dummy-taboo-lora-llama-3.2-1b-{word}"
DUMMY_WORD = "banana"
```

## 6. vastai remote (only if you run tests there)

That account has no egress and fetches through `/run/hf-fetch.sock` against
`/etc/hf-model-allowlist.txt`. Both repo ids need adding to that allowlist,
public or private — `HF_TOKEN` is not the gate there. It is also unconfirmed
whether the daemon handles adapter and LoRA repos as well as base models. Test
with a single fetch before relying on remote runs.
