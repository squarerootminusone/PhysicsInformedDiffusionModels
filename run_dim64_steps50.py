"""3 replicates: model_dim=64 + diff_steps=50 (fewer steps -> 2x faster sampling; does quality hold?).
Waits for the combo-x3 to free the GPU, trains 3, aggregates, evals r0 at the fd_acc=2 reference.
"""
import os, time, json, subprocess, sys, statistics

try:
    combo = open('COMBO_PID').read().strip()
except Exception:
    combo = ''
while combo and os.path.exists(f'/proc/{combo}'):
    time.sleep(5)
print('GPU free; dim64 + diff_steps=50 x3', flush=True)

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True, model_dim=64, diff_steps=50)

objs = []
for ri in range(3):
    name = f'dim64_steps50_r{ri}'
    ov = dict(base, name=name)
    rp = f'{name}.obj'
    print(f'=== TRAIN {name} ===', flush=True)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
        objs.append(float(open(rp).read().strip()))
        print(f'{name} best = {objs[-1]:.4e}', flush=True)
    except Exception as e:
        print(f'{name} FAILED: {e}', flush=True)

print('=== SUMMARY (best sample-residual) ===', flush=True)
if objs:
    m = statistics.mean(objs)
    sd = statistics.pstdev(objs) if len(objs) > 1 else 0.0
    print(f'dim64+steps50: {[f"{o:.4e}" for o in objs]}  mean={m:.4e}  std={sd:.4e}', flush=True)

ck = 'trained_models/dim64_steps50_r0/model/checkpoint_50000.pt'
if os.path.exists(ck):
    print('=== EVAL r0 @ fd_acc=2 reference ===', flush=True)
    try:
        subprocess.run([sys.executable, 'eval_fp64.py', ck,
                        'trained_models/dim64_steps50_r0/model/model.yaml', '64', '2'], check=True)
    except Exception as e:
        print(f'eval FAILED: {e}', flush=True)
print('=== DIM64STEPS50_DONE ===', flush=True)
