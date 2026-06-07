"""1 run: model_dim=64 + diff_steps=50 (fewer steps -> 2x faster sampling; does quality hold?).
Waits for the combo-x3 to free the GPU, trains, then evals at the fd_acc=2 reference.
"""
import os, time, json, subprocess, sys

try:
    combo = open('COMBO_PID').read().strip()
except Exception:
    combo = ''
while combo and os.path.exists(f'/proc/{combo}'):
    time.sleep(5)
print('GPU free; dim64 + diff_steps=50', flush=True)

ov = dict(name='dim64_steps50', config_path='configs/darcy_pidm_me.yaml',
          train_iterations=50000, bf16_train=True, async_eval=True, sample_freq=10 ** 9,
          final_sample=False, sample_eval_freq=5000, test_eval_freq=500,
          c_residual=0.00374, lr_schedule='plateau', save_final_checkpoint=True, wandb_track=True,
          model_dim=64, diff_steps=50)
print('=== TRAIN dim64_steps50 ===', flush=True)
subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), 'dim64_steps50.obj'], check=True)
print(f'dim64_steps50 best sample-residual = {open("dim64_steps50.obj").read().strip()}', flush=True)

print('=== EVAL @ fd_acc=2 reference ===', flush=True)
subprocess.run([sys.executable, 'eval_fp64.py',
                'trained_models/dim64_steps50/model/checkpoint_50000.pt',
                'trained_models/dim64_steps50/model/model.yaml', '64', '2'], check=True)
print('=== DIM64STEPS50_DONE ===', flush=True)
