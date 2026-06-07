"""5 LR-decay replicates (unseeded -> natural variance) + fp32/fp64 eval of all 5 checkpoints.

Each replicate runs in its own subprocess (reduce-overhead cudagraphs safe), trains 35k with the
plateau LR schedule at c_residual=3.74e-3, and saves a final checkpoint. Then eval_fp64.py reports
fp32 + fp64 sample-residual for each. Waits for the batch-ramp run to free the GPU first.
"""
import os, time, json, subprocess, sys, statistics

# wait for the batch-ramp run (if any) to finish so we don't contend for the GPU
try:
    br = open('/tmp/br.pid').read().strip()
except Exception:
    br = ''
while br and os.path.exists(f'/proc/{br}'):
    time.sleep(5)
print('GPU free; starting LR-decay replicates', flush=True)

ITERS = 50000
base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=ITERS, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True)

objs = []
for i in range(5):
    ov = dict(base, name=f'lrdecay_r{i}')
    rp = f'/root/PhysicsInformedDiffusionModels/lrdecay_r{i}.obj'
    print(f'=== TRAIN REPLICATE {i} ===', flush=True)
    subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
    objs.append(float(open(rp).read().strip()))
    print(f'replicate {i} best sample-residual (fp32 async eval) = {objs[-1]:.4e}', flush=True)

print('=== REPLICATES DONE ===', flush=True)
print('per-replicate (fp32 async best):', [f'{o:.4e}' for o in objs], flush=True)
print(f'mean={statistics.mean(objs):.4e}  std={statistics.pstdev(objs):.4e}', flush=True)

print('=== FP32/FP64 EVAL OF ALL 5 CHECKPOINTS ===', flush=True)
for i in range(5):
    ckpt = f'trained_models/lrdecay_r{i}/model/checkpoint_{ITERS}.pt'
    print(f'--- eval r{i} ---', flush=True)
    try:
        subprocess.run([sys.executable, 'eval_fp64.py', ckpt, 'configs/darcy_pidm_me.yaml', '64'],
                       check=True)
    except Exception as e:
        print(f'eval r{i} FAILED: {e}', flush=True)
print('=== ALL_DONE ===', flush=True)
