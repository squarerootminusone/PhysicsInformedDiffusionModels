#!/bin/bash
#SBATCH --job-name=pidm_darcy_cocogen
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=23:55:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="/scratch/$USER/PhysicsInformedDiffusionModels"
VARIANT="darcy_cocogen"

cd "$WORKDIR"
mkdir -p slurm/logs

module load 2024r1
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/$USER/conda_envs/pidm

export PIDM_RUN_NAME="${VARIANT}_${SLURM_JOB_ID}"

python - <<'PYEOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available")
PYEOF

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

VRAM_LOG="slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vram.csv"
( while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "${ts},${mem}" >> "$VRAM_LOG"
    sleep 60
done ) &
VRAM_LOGGER_PID=$!
trap "kill $VRAM_LOGGER_PID 2>/dev/null" EXIT

TMPSCRIPT=$(mktemp --suffix=_main.py)
sed -e "s/name = 'run_1'/name = '${PIDM_RUN_NAME}'/" \
    -e "s/wandb_track = False/wandb_track = True/" \
    -e "s|Path('model.yaml')|Path('configs/${VARIANT}.yaml')|" main.py > "$TMPSCRIPT"
PYTHONPATH="$WORKDIR" python "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
