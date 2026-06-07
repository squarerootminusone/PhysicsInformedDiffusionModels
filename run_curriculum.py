"""Curriculum run: fd_acc ramp 2->4 at 25k, with c_residual recalibrated to keep the physics loss
continuous (isolates the *accuracy* change from a weight change). Waits for the fd_acc experiment,
measures the fd2/fd4 ratio on a converged checkpoint, trains, then evals at the fd_acc=2 reference.
"""
import os, time, json, subprocess, sys

try:
    fda = open('FDACC_PID').read().strip()
except Exception:
    fda = ''
while fda and os.path.exists(f'/proc/{fda}'):
    time.sleep(5)
print('GPU free; starting curriculum run', flush=True)

# 1) measure fd2/fd4 residual ratio (same generated samples) on a converged checkpoint
print('=== measure fd2/fd4 ratio ===', flush=True)
m = subprocess.run([sys.executable, 'measure_fd_ratio.py',
                    'trained_models/lrdecay_r0/model/checkpoint_50000.pt',
                    'configs/darcy_pidm_me.yaml', '32'], capture_output=True, text=True)
print(m.stdout, flush=True)
if m.returncode != 0:
    print('measure FAILED:\n' + m.stderr[-1500:], flush=True)
ratio = 1.0
for line in m.stdout.splitlines():
    if line.startswith('RATIO='):
        ratio = float(line.split('=')[1])
C_BASE = 0.00374
C_NEW = C_BASE * ratio
print(f'recalibrated c_residual after fd2->4 switch: {C_BASE:.3e} x {ratio:.4f} = {C_NEW:.3e}', flush=True)

# 2) curriculum training
ov = dict(name='curriculum_fd2to4', config_path='configs/darcy_pidm_me.yaml',
          train_iterations=50000, bf16_train=True, async_eval=True, sample_freq=10 ** 9,
          final_sample=False, sample_eval_freq=5000, test_eval_freq=500,
          c_residual=C_BASE, lr_schedule='plateau', save_final_checkpoint=True, wandb_track=True,
          fd_acc_schedule={25000: 4}, c_residual_schedule={25000: C_NEW})
print('=== TRAIN curriculum (fd2->4 @25k, c_residual recalibrated) ===', flush=True)
subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), 'curriculum.obj'], check=True)
print(f'curriculum best sample-residual (schedule-measured) = {open("curriculum.obj").read().strip()}',
      flush=True)

# 3) compare at the fixed fd_acc=2 reference
print('=== EVAL curriculum @ ref fd_acc=2 (64 samples) ===', flush=True)
subprocess.run([sys.executable, 'eval_fp64.py',
                'trained_models/curriculum_fd2to4/model/checkpoint_50000.pt',
                'configs/darcy_pidm_me.yaml', '64', '2'], check=True)
print('=== CURRICULUM_DONE ===', flush=True)
