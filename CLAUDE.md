# PIDM Reproduction Project — Claude Context

## What this project is
Reproduction of "Physics-Informed Diffusion Models" (Bastek et al., ICLR 2025).
We are reproducing the Darcy flow and topology optimization experiments from Section 4,
running hyperparameter sweeps, and writing a blog post about our findings.

Paper: https://arxiv.org/abs/2403.14404
Repo: https://github.com/jhbastek/PhysicsInformedDiffusionModels

## Who is working on this
- Person 1 (dstoyanova): week 4 setup, Darcy flow reproduction, blog assembly
- Person 2: topology optimization reproduction + ablation study
- Person 3: hyperparameter sweep on Darcy flow

## Cluster: DelftBlue (TU Delft HPC)
- Login: `ssh dstoyanova@login.delftblue.tudelft.nl`
- Home dir: `/home/dstoyanova/` — limited quota, do NOT store data or checkpoints here
- Scratch dir: `/scratch/dstoyanova/` — use this for all data, checkpoints, outputs
- Scheduler: SLURM
- GPU partitions available:
  - `gpu` — NVIDIA Tesla V100S, 32GB VRAM (phase 1)
  - `gpu-a100` — NVIDIA A100, 80GB VRAM (phase 2)
  - `gpu-a100-small` — A100 partitioned into 10GB instances (not suitable for us)
- Use `gpu` or `gpu-a100` partition for all training jobs
- Must be on TU Delft network or EduVPN to SSH in

## Repo structure
```
PhysicsInformedDiffusionModels/
├── main.py          # main training script for Darcy + topology opt.
├── main_toy.py      # toy problem (unit circle), ~12 min, use as sanity check
├── sample.py        # inference/evaluation script
├── model.yaml       # config file — change this to switch between model variants
├── src/             # model architecture and utilities
├── data/            # place downloaded data here (darcy/ and mechanics/)
└── trained_models/  # place downloaded pretrained models here
```

Data must be downloaded from ETHZ Research Collection:
https://doi.org/10.3929/ethz-b-000674074
Place unzipped contents under `/scratch/dstoyanova/PhysicsInformedDiffusionModels/`

## Conda environment
Environment name: `pidm`
Python: 3.11
Key packages: pytorch>=2.0.1, findiff, solidspy, pandas, einops, einops-exts,
              rotary_embedding_torch, torchvision, opencv, tqdm, matplotlib,
              imageio, wandb (optional)

To activate: `conda activate pidm`
Installed at: `/home/dstoyanova/miniconda3/envs/pidm`

## The 5 model variants and their yaml configs

All variants use the same main.py — only model.yaml changes.

| Variant       | c_residual | x0_estimation | residual_grad_guidance | M_correction | N_correction |
|---------------|------------|---------------|------------------------|--------------|--------------|
| Diffusion     | 0          | mean          | False                  | 0            | 0            |
| PG-Diffusion  | 0          | mean          | True                   | 0            | 0            |
| CoCoGen       | 0          | mean          | False                  | 25           | 50           |
| PIDM-ME       | 0.001      | mean          | False                  | 0            | 0            |
| PIDM-SE       | 0.00001    | sample        | False                  | 0            | 0            |

Fixed for all variants:
- c_data: 1
- c_ineq: 0
- lambda_opt: 0
- diff_steps: 100
- fd_acc: 2
- gov_eqs: darcy (for Darcy); mechanics (for topology opt.)

Separate yaml files are stored as:
`configs/darcy_diffusion.yaml`
`configs/darcy_pg.yaml`
`configs/darcy_cocogen.yaml`
`configs/darcy_pidm_me.yaml`
`configs/darcy_pidm_se.yaml`
(and equivalent mechanics_ variants for topology opt.)

## SLURM job scripts
Stored in `slurm/`
- `toy.slurm` — sanity check, ~12 min, 1 GPU
- `darcy_<variant>.slurm` — Darcy flow training, ~13-22h, 1 GPU
- `mechanics_<variant>.slurm` — topology opt. training, ~48-54h, 1 GPU

All jobs request 1 GPU, 1 node, appropriate wall time.
Submit with: `sbatch slurm/<script>.slurm`
Check status: `squeue -u dstoyanova`
Cancel job: `scancel <jobid>`

## Week 4 order of operations
1. SSH into DelftBlue
2. Clone repo into /scratch/dstoyanova/
3. Download and place data into /scratch/dstoyanova/PhysicsInformedDiffusionModels/
4. Install miniconda and create pidm environment
5. Run toy sanity check (main_toy.py via SLURM)
6. If toy passes: prepare all yaml configs and submit all Darcy + topology jobs

## Key paper results to reproduce
- Fig. 2: residual error + test data loss curves over training for all 5 Darcy variants
- Fig. 3: generated permeability/pressure fields + residual maps
- Table 1: RMAE, MDN % CE, % VFE for topology opt. (in- and out-of-distribution)

## Notes
- Scratch storage is purged periodically — copy important checkpoints elsewhere
- Topology opt. training takes ~48-54h — submit early, set wall time to 60h to be safe
- CoCoGen required careful epsilon tuning in the paper — see Appendix A.6.2
- The repo uses wandb for logging (optional) — set to disabled if no account
