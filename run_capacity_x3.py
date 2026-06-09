"""3 replicates each of model_dim=64 and diff_steps=200 -> variance on the best sample-residual.
Reuses the existing finished run as r0 (if its .obj is present) so only 2 new runs per config.
"""
import os, json, subprocess, sys, statistics

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True)
configs = {'modeldim64': dict(model_dim=64), 'diffsteps200': dict(diff_steps=200)}

results = {}
for cfgname, extra in configs.items():
    objs = []
    if os.path.exists(f'{cfgname}.obj'):                 # existing finished run = r0
        objs.append(float(open(f'{cfgname}.obj').read().strip()))
        print(f'{cfgname} r0 (existing) best = {objs[0]:.4e}', flush=True)
    while len(objs) < 3:
        name = f'{cfgname}_r{len(objs)}'
        ov = dict(base, name=name, **extra)
        rp = f'{name}.obj'
        print(f'=== TRAIN {name} ({extra}) ===', flush=True)
        try:
            subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
            objs.append(float(open(rp).read().strip()))
            print(f'{name} best = {objs[-1]:.4e}', flush=True)
        except Exception as e:
            print(f'{name} FAILED: {e}', flush=True)
            break
    results[cfgname] = objs

print('=== SUMMARY (best sample-residual, in-training async) ===', flush=True)
for cfgname, objs in results.items():
    if objs:
        m = statistics.mean(objs)
        sd = statistics.pstdev(objs) if len(objs) > 1 else 0.0
        print(f'{cfgname}: {[f"{o:.4e}" for o in objs]}  mean={m:.4e}  std={sd:.4e}', flush=True)
print('baseline (dim32/100-step) lrdecay: mean 7.97e-3 ± 0.45e-3', flush=True)
print('=== CAPX3_DONE ===', flush=True)
