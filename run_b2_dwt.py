"""R5/B2: 2 replicates of the best recipe (dim64 + diff_steps=50 + c_residual=0.00374 +
plateau LR + bf16) with the Haar-DWT high-frequency-weighted data loss (hf_loss_weight=1.0).
Compare against B0 (same recipe, hf_loss_weight=0).
"""
import json, subprocess, sys, statistics

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=False, model_dim=64, diff_steps=50,
            hf_loss_weight=1.0)

objs = []
for ri in range(2):
    name = f'b2_dwt_r{ri}'
    ov = dict(base, name=name)
    rp = f'{name}.obj'
    print(f'=== TRAIN {name} ===', flush=True)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
        objs.append(float(open(rp).read().strip()))
        print(f'{name} best = {objs[-1]:.4e}', flush=True)
    except Exception as e:
        print(f'{name} FAILED: {e}', flush=True)

if objs:
    m = statistics.mean(objs)
    sd = statistics.pstdev(objs) if len(objs) > 1 else 0.0
    print(f'b2_dwt: {[f"{o:.4e}" for o in objs]}  mean={m:.4e}  std={sd:.4e}', flush=True)
print('=== B2_DWT_DONE ===', flush=True)
