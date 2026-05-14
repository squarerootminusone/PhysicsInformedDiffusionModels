#!/bin/bash
#SBATCH --job-name=pidm_darcy_cocogen
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="/scratch/dstoyanova/PhysicsInformedDiffusionModels"
VARIANT="darcy_cocogen"

cd "$WORKDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

cp "configs/${VARIANT}.yaml" model.yaml

export PIDM_RUN_NAME="${VARIANT}_${SLURM_JOB_ID}"
python -c "
import os
code = open('main.py').read().replace(\"name = 'run_1'\", \"name = '\" + os.environ['PIDM_RUN_NAME'] + \"'\")
exec(compile(code, 'main.py', 'exec'))
"
