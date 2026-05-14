#!/bin/bash
#SBATCH --job-name=eval_darcy_se
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="/scratch/dstoyanova/PhysicsInformedDiffusionModels"
cd "$WORKDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

python sample_eval.py \
    --directory_path ./trained_models/darcy/ \
    --name PIDM-SE \
    --load_model_step 300000 \
    --output_dir ./results/reproduced/darcy/PIDM-SE
