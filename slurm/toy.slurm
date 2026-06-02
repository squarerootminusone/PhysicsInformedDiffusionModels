#!/bin/bash
#SBATCH --job-name=pidm_toy
#SBATCH --partition=gpu-v100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
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

python - <<'PYEOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available")
PYEOF

nvidia-smi

# Run toy sanity check (~12 min).
# Patches wandb_track to False so no wandb account is needed.
TMPSCRIPT=$(mktemp --suffix=_main_toy.py)
sed -e "s/'wandb_track': True/'wandb_track': False/" \
    -e "s/'name': 'run_1'/'name': 'toy_${SLURM_JOB_ID}'/" \
    main_toy.py > "$TMPSCRIPT"
PYTHONPATH="$WORKDIR" python "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
