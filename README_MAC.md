# CoppyLora_webUI — macOS (Apple Silicon) Setup

This is the macOS port of [CoppyLora_webUI](https://github.com/tori29umai0123/CoppyLora_webUI). The Gradio UI and SDXL LoRA training logic are identical to upstream — only the platform layer (installer, shell scripts, device selection, precision) has been adapted for Apple Silicon (MPS).

For Windows users, follow the original `README.md` instead.

---

## 1. Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4 — any tier)
- macOS 14 (Sonoma) or later — required for full MPS support
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.11 — install via Homebrew if missing: `brew install python@3.11`
- ~15 GB free disk for the Python venv plus the downloaded models
- Unified memory: 16 GB minimum, 24 GB+ recommended for the upstream defaults (`train_batch_size = 2`, `resolution = "1024,1024"`)

---

## 2. Install

```bash
git clone https://github.com/hiyukoim/CoppyLora_webUI.git
cd CoppyLora_webUI
git checkout mac
./install.sh
```

The installer:

- Verifies macOS arm64, Xcode CLT, `python3.11`, `git`.
- Installs `uv` if missing.
- Creates `.venv/` with Python 3.11.
- Installs stable PyTorch (Mac wheels include MPS support).
- Clones `kohya-ss/sd-scripts` at the upstream-pinned commit.
- Installs all pip dependencies, except those that have no Apple Silicon build (`xformers`, `triton-windows`, `bitsandbytes`, `onnxruntime-gpu`, `pyinstaller`).
- Patches the diffusers imports in `sd-scripts/library/lpw_stable_diffusion.py` and `sd-scripts/library/sdxl_lpw_stable_diffusion.py` (same patch the Windows installer applies).
- Writes a Mac-friendly `accelerate` default config (only if you don't already have one).

Re-running `./install.sh` is idempotent.

---

## 3. Download models (manual)

The Mac port does not ship an automated downloader. Download these models yourself and place them at the indicated paths under the repo root.

### 3.1 SDXL base model

Path: `models/SDXL/animagine-xl-3.1.safetensors`

URL: <https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors>

SHA-256: `e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0`

### 3.2 WD14 tagger (SmilingWolf)

Path: `models/tagger/`

Files (URL is `https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/<file>`):

| File | SHA-256 |
|---|---|
| `config.json` | `ddcdd28facc40ee8d0ef4b16ee3e7c70e4d7b156aff7b0f2ccc180e617eda795` |
| `model.onnx` | `e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1` |
| `selected_tags.csv` | `298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217` |
| `sw_jax_cv_config.json` | `4dda7ac5591de07f7444ca30f2f89971a21769f1db6279f92ca996d371b761c9` |

### 3.3 Base LoRAs (mylora_V2)

Path: `models/LoRA/`

Files (URL is `https://huggingface.co/tori29umai/mylora_V2/resolve/main/<file>`):

| File | SHA-256 |
|---|---|
| `copi-ki-base-boy_cl_am31.safetensors` | `5aa481749352901b790a0128500d2b83ac63a4bb51eb3278e7da0d6976c6087d` |
| `copi-ki-base-boy_ncl_am31.safetensors` | `2b112eefd7203827606670df0a5d5f4fe61317a712180456ecc544d8b815964a` |
| `copi-ki-base-boy_ncnl_am31.safetensors` | `61c965a0c1e45262282511e6c1d40c2a2b226616bad9ea4e77c504951d4074c3` |
| `copi-ki-base-boy_cnl_am31.safetensors` | `1e96f710153fd33255526727f9241235617651c9e1df844885612182a2c50675` |
| `copi-ki-base-girl_cl_am31.safetensors` | `8979e0cee623891d15980ca1513cd6b5d97684129977dcad694524a6271bd0d1` |
| `copi-ki-base-girl_ncl_am31.safetensors` | `e14961aed6102b17b920dea89c7a30fbc48910b66ce833f775806f34ed581f68` |
| `copi-ki-base-girl_ncnl_am31.safetensors` | `f7d3d0f2bc9896751865cabf05de1d383b1d86d6b643a805868e9372bbb590d4` |
| `copi-ki-base-girl_cnl_am31.safetensors` | `f3925e4c51c1f2cb2dd339849713e2502808741f08e42246eb2b6e5f01f0c4ce` |
| `copi-ki-base-female_p_am31.safetensors` | (verify with the upstream `CoppyLora_webUI_DL.cmd`) |
| `copi-ki-base-male_p_am31.safetensors` | (verify with the upstream `CoppyLora_webUI_DL.cmd`) |

Verify any file with: `shasum -a 256 <file>`

---

## 4. Run

```bash
./start.sh
```

The Gradio app launches and your browser opens to <http://127.0.0.1:7860> (or the next free port if 7860 is taken).

`./venv.sh` drops you into an interactive subshell with the venv activated, useful for running `python -c "..."` checks against the same environment.

---

## 5. Memory tuning (if you hit MPS out-of-memory)

The upstream defaults assume ~24 GB unified memory. If a 16 GB Mac runs out of memory, try these in `config.toml`, in order of preference:

1. **Drop batch size**: `train_batch_size = 1` — biggest single saving.
2. **Drop resolution**: `resolution = "768,768"` (still SDXL-compatible thanks to bucketing).
3. **Switch to AdaFactor** — most memory-efficient optimizer, ~30% reduction vs AdamW:
   ```toml
   optimizer_type = "Adafactor"
   optimizer_args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
   ```
4. **Disable the MPS upper-memory cap** (last resort — can swap to disk):
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ./start.sh
   ```

---

## 6. Performance expectations

- M1 Pro 16 GB: ~6 s/iter, ~10 GB unified memory peak (community report).
- M4 Pro 24 GB: faster (estimate ~3–5 s/iter), more headroom.
- Default 500 steps × ~5 s/iter ≈ **40 minutes per LoRA** on M4 Pro.
- DetailTrain runs the training step twice (base + kari), so roughly **2× SimpleTrain wall-clock**.

The very first iteration is slow — Metal kernels compile on first use. Subsequent iterations run at the steady-state rate.

---

## 7. Common gotchas

- **`xformers` import errors** — already disabled in `config.toml`. If you still see them, you're not on the `mac` branch.
- **`fp16` / `fp8` runtime errors** — same, switch to `mac`.
- **`bitsandbytes` errors at startup** — not installed by `install.sh`; if you installed it manually, `uv pip uninstall bitsandbytes`.
- **DataLoader worker crashes** — `max_data_loader_n_workers = 0` is required on MPS; do not raise it.
- **First iteration is slow** — Metal kernel compilation; not a bug.
- **Out of memory** — see section 5.

---

## 8. Using PyTorch nightly (optional)

For the latest MPS bugfixes, swap the stable PyTorch install for nightly:

```bash
source .venv/bin/activate
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

The `cpu` in the URL is misleading — Apple's official Metal docs use this same URL for MPS-enabled nightly wheels.

---

## 9. Differences from the Windows version

- No `xformers`, `bitsandbytes`, `triton-windows`, `onnxruntime-gpu`.
- No PyInstaller `.app` bundle (deferred to v2).
- No automated model downloader (`CoppyLora_webUI_DL.cmd` Mac equivalent — manual download per section 3).
- Uses `AdamW` (32-bit) instead of `AdamW8bit`. AdaFactor is documented as the memory-efficient alternative.
- Otherwise: training logic, Gradio UI, captions, base PNGs, merge/resize flow are byte-identical.

---

## 10. Reporting issues

Open issues on the fork: <https://github.com/hiyukoim/CoppyLora_webUI/issues>

Include:

- macOS version: `sw_vers`
- Mac model: `sysctl -n machdep.cpu.brand_string`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Full traceback if any
