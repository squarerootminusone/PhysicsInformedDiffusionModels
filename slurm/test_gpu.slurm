#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu-v100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="/scratch/$USER/PhysicsInformedDiffusionModels"
cd "$WORKDIR"
mkdir -p slurm/logs

module load 2024r1
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/$USER/conda_envs/pidm

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

python - <<'PYEOF'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available")
PYEOF
