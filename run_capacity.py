"""Capacity tests: 1 run each of diff_steps=200 and model_dim=64 (vs the fd2/dim32/100-step
baseline), then eval each at the fd_acc=2 reference using the checkpoint's own saved config
(so the right diff_steps / model_dim are used to rebuild and sample).
"""
import os, json, subprocess, sys

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True)

tests = [('diffsteps200', dict(diff_steps=200)),
         ('modeldim64', dict(model_dim=64))]

for name, extra in tests:
    ov = dict(base, name=name, **extra)
    print(f'=== TRAIN {name} ({extra}) ===', flush=True)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), f'{name}.obj'], check=True)
        print(f'{name} best sample-residual (in-training async) = {open(f"{name}.obj").read().strip()}',
              flush=True)
    except Exception as e:
        print(f'{name} TRAIN FAILED: {e}', flush=True)

print('=== EVAL each @ fd_acc=2 reference (64 samples, using saved model.yaml) ===', flush=True)
for name, _ in tests:
    ck = f'trained_models/{name}/model/checkpoint_50000.pt'
    cfgp = f'trained_models/{name}/model/model.yaml'
    print(f'--- {name} @ ref fd_acc=2 ---', flush=True)
    if not os.path.exists(ck):
        print(f'  (no checkpoint: {ck})', flush=True)
        continue
    try:
        subprocess.run([sys.executable, 'eval_fp64.py', ck, cfgp, '64', '2'], check=True)
    except Exception as e:
        print(f'  eval FAILED: {e}', flush=True)
print('baseline for reference: fd2/dim32/100-step lrdecay = 7.77e-3 @ fd_acc=2', flush=True)
print('=== CAPACITY_DONE ===', flush=True)
