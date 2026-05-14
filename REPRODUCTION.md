# Reproduction Notes — PIDM on DelftBlue (TU Delft HPC)

Reproduction of Bastek et al., "Physics-Informed Diffusion Models", ICLR 2025.
Cluster: DelftBlue, May 2026. Author: dstoyanova.

---

## Environment fixes

The `pidm` conda environment as described in the README is missing one package and has one version conflict that will silently break evaluation:

**1. `pyyaml` not installed**

`sample.py` and `sample_eval.py` import `yaml` to read `model.yaml`, but `pyyaml` is not installed by the base environment setup. Fix:

```bash
conda activate pidm
pip install pyyaml
```

**2. `findiff` must be pinned to 0.10.0**

`findiff>=0.11` changed its API to reject negative spacing values. The codebase passes a negated grid spacing (`d1 *= -1` in `src/residuals_darcy.py:33`) for visualization consistency, which breaks with newer versions:

```
ValueError: Spacing must be > 0.
```

Fix by downgrading:

```bash
pip install "findiff==0.10.0"
```

---

## Parametrized evaluation script

The original `sample.py` hardcodes `directory_path`, `name`, and `load_model_step` and writes outputs back into `trained_models/`. We provide `sample_eval.py` as a drop-in replacement that accepts CLI arguments and writes to a caller-specified output directory:

```bash
python sample_eval.py \
    --directory_path ./trained_models/darcy/ \
    --name PIDM-ME \
    --load_model_step 300000 \
    --output_dir ./results/reproduced/darcy/PIDM-ME
```

SLURM scripts using this are in `slurm/eval_*.slurm`.

---

## SLURM on DelftBlue — GPU flag

DelftBlue requires `--gpus-per-task` rather than `--gpus-per-node` or `--gres=gpu:N`. Use:

```
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
```

Memory per CPU must not exceed 8000 MB on the `gpu-a100` partition. We use `--mem-per-cpu=7500M`.

When all GPUs are fully occupied, SLURM 21.08 displays `AssocMaxGRESPerJob` as the pending reason instead of the more intuitive `(Resources)`. This is a display quirk — jobs will start normally once a GPU frees up.

---

## Pretrained checkpoints available

The ETHZ Research Collection archive includes checkpoints for a subset of the paper's variants:

| Model | Gov. eqs | Checkpoint step | Available |
|-------|----------|----------------|-----------|
| PIDM-ME | Darcy | 300000 | ✓ |
| PIDM-SE | Darcy | 300000 | ✓ |
| Diffusion | Darcy | — | ✗ must train |
| PG-Diffusion | Darcy | — | ✗ must train |
| CoCoGen | Darcy | — | ✗ must train |
| PIDM | Mechanics | 600000 | ✓ |
| standard_diffusion | Mechanics | 600000 | ✓ |

Training SLURM scripts for the missing Darcy variants are in `slurm/darcy_diffusion.slurm`, `slurm/darcy_pg.slurm`, `slurm/darcy_cocogen.slurm`.

---

## Evaluation results

Results are saved under `results/reproduced/`. Validation samples (3 generated fields + residuals) completed successfully for all available checkpoints.

### Darcy flow — validation residuals (3 samples)

| Model | Mean abs. residual |
|-------|--------------------|
| PIDM-ME | 0.01009 |
| PIDM-SE | 0.05876 |

### Topology optimization — validation residuals (3 samples)

| Model | Mean abs. residual | Mean compliance | Mean inequality |
|-------|-------------------|----------------|-----------------|
| PIDM | 0.001082 | 6.340 | -0.002529 |
| standard_diffusion | 0.001406 | 4.344 | -0.001485 |

---

## Topology optimization — full test set evaluation is slow

Reproducing Table 1 of the paper requires evaluating the full test sets:
`test_level_1` (1800 samples, in-distribution) and `test_level_2` (1000 samples, OOD).

Each sample requires a FEM solve via `solidspy` (pure Python), taking ~72 seconds per sample on an A100. This means:

- `test_level_1`: ~36h per model
- `test_level_2`: ~20h per model
- Total per model: ~56h — exceeds the 24h MaxWall on DelftBlue

**Table 1 metrics (CE error, VF error, FM error) cannot be reproduced in a single job as-is.**

Workarounds: evaluate a subset by setting `test_batches=N` in `sample_eval.py`, or run two sequential 24h jobs (one per test set). A more principled fix would be to replace the solidspy FEM solver with a GPU-batched implementation — see the extension note below.

---

## Potential extension: GPU-accelerated FEM evaluation

The FEM bottleneck is an interesting finding in its own right. The 64×64 structured mesh used for topology optimization is an ideal case for a GPU FEM implementation (regular grid, no complex meshing). Replacing solidspy with a batched GPU solver (e.g. via `torch-fem` or a custom CUDA kernel) could reduce per-sample evaluation from ~72s to under 1s, making full test set evaluation feasible and — more importantly — making physics evaluation fast enough to run *during* training, tightening the physics-informed loop beyond what the current paper does.
