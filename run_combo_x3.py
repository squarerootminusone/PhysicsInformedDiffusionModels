"""3 replicates of the COMBINED capacity config: model_dim=64 AND diff_steps=200 together.
Variance on the best sample-residual, vs the dim32/100-step baseline and the dim64-only result.
"""
import os, json, subprocess, sys, statistics

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True,
            model_dim=64, diff_steps=200)

objs = []
for ri in range(3):
    name = f'dim64_steps200_r{ri}'
    ov = dict(base, name=name)
    rp = f'{name}.obj'
    print(f'=== TRAIN {name} (model_dim=64, diff_steps=200) ===', flush=True)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
        objs.append(float(open(rp).read().strip()))
        print(f'{name} best = {objs[-1]:.4e}', flush=True)
    except Exception as e:
        print(f'{name} FAILED: {e}', flush=True)

print('=== SUMMARY (best sample-residual, in-training async) ===', flush=True)
if objs:
    m = statistics.mean(objs)
    sd = statistics.pstdev(objs) if len(objs) > 1 else 0.0
    print(f'dim64+steps200: {[f"{o:.4e}" for o in objs]}  mean={m:.4e}  std={sd:.4e}', flush=True)
print('reference: baseline(dim32/100) 7.97e-3 ± 0.45e-3 ; dim64-only best ~6.43e-3', flush=True)
print('=== COMBO_DONE ===', flush=True)
