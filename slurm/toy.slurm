#!/bin/bash
#SBATCH --job-name=pidm_toy
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="/scratch/dstoyanova/PhysicsInformedDiffusionModels"
cd "$WORKDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

# Run toy sanity check (~12 min).
# Patches wandb_track to False so no wandb account is needed.
export SLURM_JOB_ID="${SLURM_JOB_ID}"
nvidia-smi
python -c "
import os
code = open('main_toy.py').read()
code = code.replace(\"'wandb_track': True\", \"'wandb_track': False\")
code = code.replace(\"'name': 'run_1'\", \"'name': 'toy_\" + os.environ['SLURM_JOB_ID'] + \"'\")
exec(compile(code, 'main_toy.py', 'exec'))
"
