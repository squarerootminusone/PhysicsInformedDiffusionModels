#!/bin/bash
#SBATCH --job-name=eval_topo_pidm
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
mkdir -p slurm/logs

module load cuda/12.1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

python - <<'PYEOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available")
PYEOF

python sample_eval.py \
    --directory_path ./trained_models/mechanics/ \
    --name PIDM \
    --load_model_step 600000 \
    --output_dir ./results/reproduced/topology/PIDM
