#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:05:00
#SBATCH --output=test_gpu.out

set -euo pipefail

module load cuda/12.1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

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
