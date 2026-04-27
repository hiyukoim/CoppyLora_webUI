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

Note: upstream's `CoppyLora_webUI_DL.cmd` also tries to download `copi-ki-base-female_p_am31.safetensors` and `copi-ki-base-male_p_am31.safetensors`, but those files do not exist on the `tori29umai/mylora_V2` Hugging Face repo (verified via the HF tree API) and are not referenced anywhere in `CoppyLora_webUI.py`. The Windows DL script silently swallows the 404. Skip them.

Verify any file with: `shasum -a 256 <file>`

---

## 4. Run

```bash
./start.sh
```

The Gradio app launches and your browser opens to <http://127.0.0.1:7860> (or the next free port if 7860 is taken).

`./venv.sh` drops you into an interactive subshell with the venv activated, useful for running `python -c "..."` checks against the same environment.

---

## 5. System hygiene before training (important on Mac)

Mac unified memory is shared with WindowServer, your browser, every Electron app, and the OS. Unlike a Windows machine with a discrete GPU, every gigabyte you waste on background apps comes directly out of training memory — and once you start swapping to disk, step time blows up from ~10 s/iter to >100 s/iter.

**Before clicking Train:**

- Quit Slack, Discord, Telegram, Spotify, Notion, and any other Electron app.
- Close all browser tabs except the Gradio one. Many-tabbed Chrome / Comet / Safari can easily hold 4–8 GB.
- (16 GB Macs only, recommended) Log out and back in before the first run. WindowServer's working set grows over time and a fresh login reclaims it.
- Open Activity Monitor → Memory tab. Watch **Swap Used** while training. >2 GB sustained means you're swapping — quit more apps.

The launcher (`start.sh`) automatically sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3`, and `PYTORCH_ENABLE_MPS_FALLBACK=1`, and the Python process reserves ~20% of RAM for the OS via `torch.mps.set_per_process_memory_fraction(0.80)`. These cap MPS allocations so the OS doesn't get paged out — but they only help if you also keep the OS side of the budget clean.

---

## 6. The two configs

The repo ships two SDXL training configs:

| File | Optimizer | Batch | Best for |
| --- | --- | --- | --- |
| `config.toml` (default) | Adafactor + `fused_backward_pass` | 1 | Every Mac. The launcher reads this. |
| `config_adamw_full.toml` | AdamW (32-bit) | 2 | Reference profile. **Will OOM on every <64 GB Mac and probably most 64 GB Macs too** — Apple's `recommendedMaxWorkingSetSize` caps MPS at ~17.7 GiB regardless of how much RAM you have, and AdamW32 + fp32 + batch=2 needs ~18 GiB. Kept for transparency, not for daily use. |

To use the AdamW reference profile on a Mac Studio with enough headroom:

```bash
cp config.toml config.toml.adafactor-backup
cp config_adamw_full.toml config.toml
./start.sh
```

To restore the default afterward:

```bash
cp config.toml.adafactor-backup config.toml
```

If even the default Adafactor profile runs out of memory:

1. **Drop resolution**: `resolution = "768,768"` (still SDXL-compatible via bucketing).
2. `train_batch_size` is already 1; can't go lower.

---

## 7. Performance expectations

Expected sustained step times on a clean M4 Pro 24 GB (Activity Monitor + Safari only, 1024×1024, `network_dim = 16`):

| Config                      | Peak memory | Step time   | 500 steps | Status on 24 GB Mac |
| --------------------------- | ----------- | ----------- | --------- | --- |
| `config.toml` (Adafactor)   | ~12 GB      | 6–10 s/iter | ~55 min   | ✅ default — fits |
| `config_adamw_full.toml`    | ~18 GB      | 8–15 s/iter | ~75 min   | ❌ OOMs at VAE caching (14.2 GiB cap) |

DetailTrain runs the training step twice (base + kari), so roughly **2× SimpleTrain wall-clock**.

The very first iteration is slow — Metal kernels compile on first use. Subsequent iterations run at the steady-state rate.

**If you see step times >30 s/iter sustained, you are swapping.** That is a system-hygiene problem (section 5), not a platform limit.

### Why the Mac default isn't AdamW

The Windows path uses **AdamW8bit** via bitsandbytes plus fp16 mixed precision plus xformers. None of those are usable on Apple Silicon in 2026:

- fp16 / bf16 still produce NaN losses on MPS for SDXL training.
- bitsandbytes has no Apple Silicon build, so 8-bit AdamW is unavailable.
- xformers has no Apple Silicon build.

That means "AdamW on Mac" is necessarily 32-bit AdamW with fp32 precision — a substitute that needs roughly **2× the Windows memory footprint**. At 1024×1024 batch=2 it peaks at ~18 GiB, which is over the MPS cap of ~14.2 GiB on a 24 GB Mac (Apple reserves the rest for the OS via `recommendedMaxWorkingSetSize`).

The Mac default therefore uses **Adafactor + `fused_backward_pass`**, which is the established Mac SDXL optimizer choice in the kohya_ss community since 2023. It is **not** bit-identical to Windows's AdamW8bit (and AdamW8bit isn't bit-identical to AdamW32 either — bitsandbytes is its own quantised approximation), but it produces the same kind of LoRA via a slightly different convergence trajectory. Visual quality on the resulting `.safetensors` rendered in Mac ComfyUI is the verification gate, not bitwise reproduction.

---

## 8. Common gotchas

- **`xformers` import errors** — already disabled in `config.toml`. If you still see them, you're not on the `mac` branch.
- **`fp16` / `fp8` runtime errors** — same, switch to `mac`.
- **`bitsandbytes` errors at startup** — not installed by `install.sh`; if you installed it manually, `uv pip uninstall bitsandbytes`.
- **DataLoader worker crashes** — `max_data_loader_n_workers = 0` is required on MPS; do not raise it.
- **First iteration is slow** — Metal kernel compilation; not a bug.
- **Out of memory** — see sections 5 and 6.

---

## 9. Using PyTorch nightly (optional)

For the latest MPS bugfixes, swap the stable PyTorch install for nightly:

```bash
source .venv/bin/activate
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

The `cpu` in the URL is misleading — Apple's official Metal docs use this same URL for MPS-enabled nightly wheels.

---

## 10. Differences from the Windows version

- No `xformers`, `bitsandbytes`, `triton-windows`, `onnxruntime-gpu`.
- No PyInstaller `.app` bundle (deferred to v2).
- No automated model downloader (`CoppyLora_webUI_DL.cmd` Mac equivalent — manual download per section 3).
- Uses **Adafactor + `fused_backward_pass`** (in `config.toml`) instead of `AdamW8bit`. AdamW8bit needs bitsandbytes, which has no Apple Silicon build. A reference 32-bit AdamW config is shipped as `config_adamw_full.toml` but will OOM on essentially every Mac (see section 7).
- Otherwise: training logic, Gradio UI, captions, base PNGs, merge/resize flow are byte-identical.

---

## 11. Reporting issues

Open issues on the fork: <https://github.com/hiyukoim/CoppyLora_webUI/issues>

Include:

- macOS version: `sw_vers`
- Mac model: `sysctl -n machdep.cpu.brand_string`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Full traceback if any
