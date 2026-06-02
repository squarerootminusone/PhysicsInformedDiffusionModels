# Reproducing Physics-Informed Diffusion Models (Bastek et al., ICLR 2025)

Operational notes from reproducing the Darcy flow case study (Section 4.1) of [arXiv:2403.14404](https://arxiv.org/abs/2403.14404). Covers the two Darcy variants `darcy_pidm_me` and `darcy_cocogen`.

## What worked (TL;DR)

- **Compute**: a single Vast.ai box with an RTX PRO 6000 Blackwell (97.9 GB VRAM, sm_120). Training both variants on a shared GPU took ~11h wallclock; evaluation (N=16 samples) took ~2 min per variant on a solo GPU.
- **Env**: Python 3.12 + a fresh venv + `torch` **nightly** cu128 (Blackwell needs sm_120 kernels, which are not in stable torch ≤ 2.7).

Reproduced numbers on the validation set (N=16 samples per variant, R_MAE definition from Appendix A.6 eq. 28):

| Variant | Mean R_MAE (ours) | Paper Fig. 2a (eyeball) |
|---|---|---|
| `darcy_pidm_me` | **0.0127** | ~10⁻² ✓ |
| `darcy_cocogen` | **1.08**   | ~1 ✓ |

## Quirks you will hit (and the fixes)

These bit us in this exact order — capturing them so you don't repeat them.

### 1. `sample.py` does not have argparse; the eval SLURM scripts call a non-existent `sample_eval.py`

`slurm/eval_darcy_*.slurm` call `python sample_eval.py --directory_path ... --name ... --load_model_step ...` but `sample_eval.py` is **not in the repo**. Only `sample.py` exists, and it hardcodes the relevant parameters at the top:

```python
# sample.py lines 15-17, 24-25
directory_path = './trained_models/darcy/'
name = 'PIDM-ME'
load_model_step = 300000
no_samples = 3       # ← paper protocol is 16, not 3
create_gif = True    # ← N=16 makes this heavy; turn off for speed
```

**Fix:** copy `sample.py` to a temp name and `sed` in the values you want. The training scripts already use this pattern — extend it for eval:

```bash
TMPEVAL=$(mktemp --suffix=_sample_eval.py)
sed -e "s|directory_path = './trained_models/darcy/'|directory_path = './trained_models/'|" \
    -e "s|name = 'PIDM-ME'|name = '${RUN_NAME}'|" \
    -e "s|^no_samples = 3|no_samples = 16|" \
    -e "s|^create_gif = True|create_gif = False|" \
    sample.py > "$TMPEVAL"
python "$TMPEVAL"
```

### 2. `findiff` ≥ 0.12 rejects negative spacing — pin to 0.10.x

`src/residuals_darcy.py:31` does `d1 *= -1` when `reverse_d1=True` (which `main.py` sets), then passes the negative spacing into `FinDiff(1, d1, 1, acc=fd_acc)` in `src/grad_utils.py:156`. Older findiff accepted this as "axis-reversed"; **findiff 0.13.1 raises `ValueError: Spacing must be > 0.`** before training starts.

**Fix:** `pip install findiff==0.10.2`.

This shows up as a traceback in `slurm/logs/*err` at job startup, after `Number of trainable parameters:` prints. If you see `findiff/grids.py ... Spacing must be > 0.` — that's this.

### 3. Parallel training jobs race on `model.yaml`

The original SLURM training scripts do `cp configs/${VARIANT}.yaml model.yaml` and then run `main.py`, which reads `Path('model.yaml').read_text()`. **If two jobs run in the same working directory** they clobber each other's `model.yaml` mid-read and crash with `yaml.scanner.ScannerError: while scanning a simple key … could not find expected ':'`.

**Fix:** drop the `cp` and `sed`-patch the path directly into `main.py`:

```bash
sed -e "s|Path('model.yaml')|Path('configs/${VARIANT}.yaml')|" ...
```

### 4. NumPy 2.x breaks torch 2.1 wheels

If you install `pip install torch==2.1.* torchvision==0.16.*` with no NumPy pin, pip pulls in NumPy 2.x and torch fails to import with `_ARRAY_API not found`. **Pin `numpy<2`** and `pandas<2.3` (pandas pulls numpy back up otherwise). And `opencv-python<4.10` (4.10+ also requires numpy≥2). The full minimum compat set:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.1.*" "torchvision==0.16.*"
pip install "numpy<2" "pandas<2.3" "opencv-python<4.10" \
            tqdm matplotlib imageio einops einops-exts rotary_embedding_torch \
            "findiff==0.10.2" pyyaml wandb dill scikit-learn solidspy
```

`dill`, `scikit-learn`, and `solidspy` are not listed in the README but are imported via `src/denoising_toy_utils.py` and `src/residuals_mechanics_K.py`. Those imports run unconditionally from `main.py`, so they need to be installable even for a Darcy-only run.

### 5. Blackwell GPUs (sm_120) need torch nightly cu128

Stable `torch==2.6.0+cu126` builds support up to sm_90 only. On an RTX PRO 6000 Blackwell you get `CUDA error: no kernel image is available for execution on the device`.

**Fix:** use the nightly:

```bash
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision
```

At time of writing this picked up `torch 2.12.0.dev20260407+cu128`, which works. Stable cu128 wheels may have appeared by the time you read this — check `torch.cuda.get_device_capability()` returns `(12, 0)` and that `torch.zeros(1, device='cuda') + 1` actually runs.

### 6. Checkpoints save raw weights, not EMA

`main.py` uses EMA (`ema_start = 1000`, `ema = EMA(0.99)`) during training and inference, but `save_model()` in `src/denoising_utils.py:281` writes only `model.state_dict()`. The EMA shadow is **never persisted**. `sample.py:load_model` reads back the raw weights.

The paper's published numbers likely use EMA at inference. Our reproduction does not (because the codebase doesn't). If you want to match exactly: modify `save_model()` to also store `ema.shadow`, and have `sample.py` apply it after `load_model`. Out of scope for the basic reproduction.

## Reproduction recipe (Vast.ai, what we actually did)

Assumes a fresh Vast.ai Linux container with a Blackwell GPU, root access, SSH key.

### Setup

```bash
# system deps for venv + downloads
apt-get install -y python3-venv python3-pip rsync curl unzip

# venv
python3 -m venv /root/pidm_env
source /root/pidm_env/bin/activate
pip install --upgrade pip

# torch nightly cu128 (sm_120 support)
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision

# everything else, pinned for compatibility
pip install "numpy<2" "pandas<2.3" "opencv-python<4.10" \
            tqdm matplotlib imageio einops einops-exts rotary_embedding_torch \
            "findiff==0.10.2" pyyaml wandb dill scikit-learn solidspy

# sanity-check Blackwell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print((torch.zeros(1,device='cuda')+1).item())"
```

### Data

Darcy data lives at ETHZ Research Collection https://doi.org/10.3929/ethz-b-000674074. The DOI redirects to a DSpace 7 frontend. You can resolve the bitstream via the REST API:

```bash
# get item UUID from handle
curl -sL "https://www.research-collection.ethz.ch/server/api/pid/find?id=hdl:20.500.11850/674074" \
  | jq -r '.uuid'
# 76a1547d-02d9-4913-80c0-4e140bd8d154

# list bundles, then bitstreams; the ORIGINAL bundle has data.zip (5.27 GB) and trained_models.zip (1.09 GB)
# direct download URL:
curl -L -o data.zip https://www.research-collection.ethz.ch/server/api/core/bitstreams/cebe5a39-f345-491e-bdf0-c99b918ddf41/content

# extract only darcy (mechanics is ~3.5 GB extra you may not need)
cd /path/to/PhysicsInformedDiffusionModels
unzip -o /path/to/data.zip "data/darcy/*" -d .
```

After extraction, expect:
```
data/darcy/train/p_data.csv   858 MB
data/darcy/train/K_data.csv   772 MB
data/darcy/valid/p_data.csv    86 MB
data/darcy/valid/K_data.csv    77 MB
```

### Train

Use the existing `slurm/darcy_pidm_me.slurm` and `slurm/darcy_cocogen.slurm` as templates — they handle the patching of `main.py` via `sed`. The patches you need are:

1. `name = 'run_1'` → unique run name (e.g. `${VARIANT}_${SLURM_JOB_ID}` or `${VARIANT}_$(date +%s)`)
2. `wandb_track = False` → `wandb_track = True` (if using W&B)
3. `Path('model.yaml')` → `Path('configs/${VARIANT}.yaml')` (avoids the race; see Quirk 3)

On Vast.ai (no SLURM, run under `nohup`):

```bash
RUN_NAME=darcy_pidm_me_$(date +%s)
VARIANT=darcy_pidm_me
TMPSCRIPT=$(mktemp --suffix=_main.py)

sed -e "s/name = 'run_1'/name = '${RUN_NAME}'/" \
    -e "s/wandb_track = False/wandb_track = True/" \
    -e "s|Path('model.yaml')|Path('configs/${VARIANT}.yaml')|" \
    main.py > "$TMPSCRIPT"

export WANDB_API_KEY=$(cat ~/.wandb_api_key)   # set up wandb login once first
export PYTHONUNBUFFERED=1
nohup python "$TMPSCRIPT" > logs/${RUN_NAME}.out 2> logs/${RUN_NAME}.err &
```

Run the cocogen variant with `VARIANT=darcy_cocogen` in a separate shell. Both fit in 30 GB VRAM on a single Blackwell (each uses ~15 GB at batch=64). Throughput on shared GPU: ~7.2 it/s each → ~11h wallclock for 300k iters. Solo: ~14.5 it/s → ~5.5h.

The 2× slowdown when parallel is dispatch-latency interleaving — the model is tiny enough that even one job under-saturates the GPU; two jobs combined hit ~96% util.

### Evaluate (N=16 per paper)

Per the paper's Fig. 2 caption: "We generate 16 samples every 10k training iterations and plot the average (solid lines) and individual (dots) residual errors."

```bash
RUN_NAME=darcy_pidm_me_<your_tag>       # same name used during training
TMPEVAL=$(mktemp --suffix=_sample_eval.py)

sed -e "s|directory_path = './trained_models/darcy/'|directory_path = './trained_models/'|" \
    -e "s|name = 'PIDM-ME'|name = '${RUN_NAME}'|" \
    -e "s|^no_samples = 3|no_samples = 16|" \
    -e "s|^create_gif = True|create_gif = False|" \
    sample.py > "$TMPEVAL"

python "$TMPEVAL"
```

Output goes to `./trained_models/${RUN_NAME}/evaluation/validation/step_300000/`:
- `sample_0/` … `sample_15/`, each with `sample_0.csv` (pressure 64×64), `sample_1.csv` (permeability 64×64)
- `sample_statistics.csv` — 17 rows (16 per-sample R_MAE + a `Mean` row)
- PNG previews per sample/channel

Sanity benchmarks (from our reproduction):
- `darcy_pidm_me` mean R_MAE ~ **0.013** (paper ~10⁻²)
- `darcy_cocogen` mean R_MAE ~ **1.08** (paper ~1)
- Vanilla `darcy_diffusion` baseline (not trained here) would be ~100 per paper Fig. 2a

If your numbers are off by ≥ 1 order of magnitude, check that `c_residual` and `x0_estimation` in your run's `model/model.yaml` match the variant config in `configs/`.

## Other variants and the mechanics study (not done here)

- `darcy_pidm_se`: `x0_estimation: sample`, `c_residual: 0.00001`. Walltime ~26h on A100. Best Darcy result in the paper after PIDM-ME.
- `darcy_pg`: `residual_grad_guidance: True`. Walltime ~23h.
- `darcy_diffusion`: vanilla baseline. Walltime ~24h.
- **Mechanics (Section 4.2)** is much heavier: UNet `dim=128`, 600k iters, ~60h on A100, requires the mechanics dataset (~3.5 GB extra from `data.zip`) and a SolidsPy stiffness matrix in `data/mechanics/solidspy_k_no_BC/`. If your compute has a walltime cap below 60h, use the existing `load_model_flag` / `load_model_step` infrastructure to chain jobs — a 2-line addition to `main.py:156` to start `range()` at `load_model_step` is enough.

## Files in this repo touched during reproduction

- `slurm/darcy_pidm_me.slurm`, `slurm/darcy_cocogen.slurm` — adapted from the upstream templates (account directive, conda env path, sed patches incl. wandb + config path + VRAM logger). Still useful as SLURM templates if you point them at a different cluster.
- `slurm/test_gpu.slurm`, `slurm/toy.slurm` — same adaptation for smoke tests.
- No edits to `main.py`, `sample.py`, or `configs/*.yaml`. All variant-specific differences are injected by `sed` in the launcher.

## Pointers

- Paper: https://arxiv.org/abs/2403.14404 — operational details for Darcy are in Section 4.1, Figure 2 caption, and Appendix A.6 (residual definition eq. 28).
- Data + pretrained models: https://doi.org/10.3929/ethz-b-000674074 (`data.zip` 5.27 GB, `trained_models.zip` 1.09 GB).
- Original GitHub: https://github.com/jhbastek/PhysicsInformedDiffusionModels — README has the bare-bones setup; this doc fills in the gaps.
