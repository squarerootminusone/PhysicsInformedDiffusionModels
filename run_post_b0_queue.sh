#!/bin/bash
# Post-B0 GPU queue: opt16/opt15 benchmarks -> R3 correction sweep -> R5 DWT replicates.
# Waits for the B0 run to finish (OBJECTIVE line in its log) before starting.
set -u
source /root/pidm_env/bin/activate
cd /root/PhysicsInformedDiffusionModels
export PYTHONUNBUFFERED=1

B0_LOG=/root/b0_r0.log
CKPT=trained_models/b0_dim64s50_r0/model/checkpoint_50000.pt

echo "[queue] waiting for B0 to finish..."
until grep -aq "OBJECTIVE=" "$B0_LOG" 2>/dev/null; do sleep 30; done
echo "[queue] B0 done: $(grep -a 'OBJECTIVE=' $B0_LOG)"

echo "[queue] stage 1: bench_opt16 (dense vs matrix-free mechanics residual)"
python bench_opt16.py > /root/queue_bench_opt16.log 2>&1

echo "[queue] stage 2: opt15 FD-compile train probe (2500 iters, dim32 darcy)"
OPT15_COMPILE_FD=1 python -c "
from main import train
train(dict(name='probe_opt15', config_path='configs/darcy_pidm_me.yaml',
           train_iterations=2500, test_eval_freq=500, sample_freq=10**9,
           final_sample=False, wandb_track=False))
" > /root/queue_probe_opt15.log 2>&1

echo "[queue] stage 2.5: dim32 bf16 throughput probes (default vs reduce-overhead)"
python -c "
from main import train
train(dict(name='probe_bf16_def', config_path='configs/darcy_pidm_me.yaml',
           bf16_train=True, compile_mode='default',
           train_iterations=2500, test_eval_freq=10**9, sample_freq=10**9,
           final_sample=False, wandb_track=False))
" > /root/queue_probe_bf16_def.log 2>&1
python -c "
from main import train
train(dict(name='probe_bf16_ro', config_path='configs/darcy_pidm_me.yaml',
           bf16_train=True, compile_mode='reduce-overhead',
           train_iterations=2500, test_eval_freq=10**9, sample_freq=10**9,
           final_sample=False, wandb_track=False))
" > /root/queue_probe_bf16_ro.log 2>&1
for f in /root/queue_probe_bf16_def.log /root/queue_probe_bf16_ro.log; do
  echo "$f: $(tr '\r' '\n' < $f | grep -aoE '[0-9.]+it/s' | tail -1)"
done

echo "[queue] stage 3a: old-step correction reference on B0 ckpt (N=60, x0)"
python tune_corrections.py --ckpt "$CKPT" --study corr_oldstep_b0 \
  --storage sqlite:///pidm_corr_b0.db --n-trials 2 --n-samples 24 \
  > /root/queue_corr_oldstep.log 2>&1

echo "[queue] stage 3b: B1 line-search correction sweep on B0 ckpt"
B1_LINESEARCH=1 python tune_corrections.py --ckpt "$CKPT" --study corr_ls_b0 \
  --storage sqlite:///pidm_corr_b0.db --n-trials 16 --n-samples 24 \
  > /root/queue_corr_ls.log 2>&1

echo "[queue] stage 4: R5/B2 DWT-weighted loss, 2x50k replicates"
python run_b2_dwt.py > /root/queue_b2_dwt.log 2>&1

echo "QUEUE_ALL_DONE"
