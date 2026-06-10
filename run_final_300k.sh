#!/bin/bash
# Chained after the post-B0 queue: opt15 re-probe -> 300k final Darcy run -> eval battery.
set -u
source /root/pidm_env/bin/activate
cd /root/PhysicsInformedDiffusionModels
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=$(cat /root/.wandb_api_key)
export WANDB_GIT_COMMIT=$(cat /root/PIDM_GIT_SHA)

echo "[final] waiting for queue to drain..."
until grep -q "QUEUE_ALL_DONE" /root/queue_main.log 2>/dev/null; do sleep 60; done

echo "[final] opt15 re-probe (int-cast fix), 2500 iters dim32"
OPT15_COMPILE_FD=1 python -c "
from main import train
train(dict(name='probe_opt15b', config_path='configs/darcy_pidm_me.yaml',
           train_iterations=2500, test_eval_freq=10**9, sample_freq=10**9,
           final_sample=False, wandb_track=False))
" > /root/final_probe_opt15b.log 2>&1
echo "[final] opt15b: $(tr '\r' '\n' < /root/final_probe_opt15b.log | grep -aoE '[0-9.]+it/s' | tail -1)"

echo "[final] launching 300k final run (dim64+steps50+cres0.00374+plateau+bf16, wandb)"
python -c "
from main import train
obj = train(dict(name='final_dim64s50_300k', config_path='configs/darcy_pidm_me.yaml',
    train_iterations=300000, bf16_train=True, async_eval=True,
    sample_freq=10**9, final_sample=False, sample_eval_freq=10000,
    test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
    save_final_checkpoint=True, wandb_track=True, model_dim=64, diff_steps=50))
print(f'OBJECTIVE={obj}')
" > /root/final_300k.log 2>&1

CKPT=trained_models/final_dim64s50_300k/model/checkpoint_300000.pt
echo "[final] eval battery on $CKPT"
python eval_fp64.py "$CKPT" "trained_models/final_dim64s50_300k/model/model.yaml" 64 2 \
  > /root/final_eval_fp64.log 2>&1
echo "[final] amplitude check (uncorrected)"
python eval_amplitude.py "$CKPT" 64 0 > /root/final_amplitude_n0.log 2>&1
echo "[final] amplitude check (line-search corrected, N=60 xt)"
B1_LINESEARCH=1 python eval_amplitude.py "$CKPT" 64 60 > /root/final_amplitude_n60.log 2>&1
echo "[final] line-search correction sweep on final ckpt"
B1_LINESEARCH=1 python tune_corrections.py --ckpt "$CKPT" --study corr_ls_final \
  --storage sqlite:///pidm_corr_final.db --n-trials 12 --n-samples 24 \
  > /root/final_corr_ls.log 2>&1

echo "FINAL_ALL_DONE"
