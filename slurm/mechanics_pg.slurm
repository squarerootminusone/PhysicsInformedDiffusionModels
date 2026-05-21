#!/bin/bash
#SBATCH --job-name=pidm_mech_pg
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=60:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# WARNING: residual_grad_guidance (PG-Diffusion) is not implemented for gov_eqs=mechanics
# in this codebase and will raise a ValueError at startup. This script is provided for
# completeness. To run PG-Diffusion, use darcy_pg.slurm for the Darcy flow experiment.

set -euo pipefail

WORKDIR="/scratch/dstoyanova/PhysicsInformedDiffusionModels"
VARIANT="mechanics_pg"

cd "$WORKDIR"
mkdir -p slurm/logs

module load cuda/12.1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pidm

cp "configs/${VARIANT}.yaml" model.yaml

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

TMPSCRIPT=$(mktemp --suffix=_main.py)
sed "s/name = 'run_1'/name = '${PIDM_RUN_NAME}'/" main.py > "$TMPSCRIPT"
PYTHONPATH="$WORKDIR" python "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
