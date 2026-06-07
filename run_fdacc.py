"""fd_acc experiment: train at finite-difference accuracy 4 and 6 (vs the fd_acc=2 baseline),
then eval ALL checkpoints at a fixed reference stencil (fd_acc=2) so the residuals are comparable.

Waits for the corrections sweep to free the GPU first. One replicate each (extend if promising).
"""
import os, time, json, subprocess, sys

# wait for the corrections sweep (if running) to finish
try:
    corr = open('CORR_PID').read().strip()
except Exception:
    corr = ''
while corr and os.path.exists(f'/proc/{corr}'):
    time.sleep(5)
print('GPU free; starting fd_acc experiment', flush=True)

base = dict(config_path='configs/darcy_pidm_me.yaml', train_iterations=50000, bf16_train=True,
            async_eval=True, sample_freq=10 ** 9, final_sample=False, sample_eval_freq=5000,
            test_eval_freq=500, c_residual=0.00374, lr_schedule='plateau',
            save_final_checkpoint=True, wandb_track=True)

for acc in (4, 6):
    ov = dict(base, name=f'fdacc{acc}', fd_acc=acc)
    rp = f'fdacc{acc}.obj'
    print(f'=== TRAIN fd_acc={acc} ===', flush=True)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(ov), rp], check=True)
        print(f'fd_acc={acc} done; best sample-residual (measured at fd{acc}) = {open(rp).read().strip()}',
              flush=True)
    except Exception as e:
        print(f'fd_acc={acc} TRAIN FAILED: {e}', flush=True)

print('=== EVAL all checkpoints at fixed reference fd_acc=2 (64 samples) ===', flush=True)
targets = [('trained_models/lrdecay_r0/model/checkpoint_50000.pt', 'fd2-trained (baseline)'),
           ('trained_models/fdacc4/model/checkpoint_50000.pt', 'fd4-trained'),
           ('trained_models/fdacc6/model/checkpoint_50000.pt', 'fd6-trained')]
for ck, lbl in targets:
    print(f'--- {lbl} @ ref fd_acc=2 ---', flush=True)
    if not os.path.exists(ck):
        print(f'  (no checkpoint: {ck})', flush=True)
        continue
    try:
        subprocess.run([sys.executable, 'eval_fp64.py', ck, 'configs/darcy_pidm_me.yaml', '64', '2'],
                       check=True)
    except Exception as e:
        print(f'  eval FAILED: {e}', flush=True)
print('=== FDACC_DONE ===', flush=True)
