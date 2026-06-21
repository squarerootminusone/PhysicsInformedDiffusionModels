"""Study A: single-parameter c_residual search for ONE config.

Each trial trains in an ISOLATED subprocess (run_one_trial.py) so cudagraphs are safe across
the many trials. Tunes c_residual log-uniformly in [center/10, center*10], no pruning, with
the best residual_mean_abs_samples as the objective. Every wandb run name contains "optuna-c".

  python tune_c.py --config configs/darcy_pidm_me.yaml --center 1e-3 --tag me --study cres_me --n-trials 5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner

os.environ.setdefault(
    'WANDB_GIT_COMMIT',
    subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
)


def run_trial_subprocess(overrides):
    """Run one train() in a fresh process; return its objective. Inherits env (WANDB_*)."""
    fd, result_path = tempfile.mkstemp(suffix='.obj')
    os.close(fd)
    try:
        subprocess.run([sys.executable, 'run_one_trial.py', json.dumps(overrides), result_path],
                       check=True)
        with open(result_path) as f:
            return float(f.read().strip())
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def build_objective(args):
    # explicit [--lo, --hi] overrides the center±10x scheme (for arbitrary-width ranges)
    lo = args.lo if args.lo else args.center / 10.0
    hi = args.hi if args.hi else args.center * 10.0

    def objective(trial):
        c_res = trial.suggest_float('c_residual', lo, hi, log=True)
        overrides = dict(
            name=f'optuna-c_{args.tag}_t{trial.number}',
            config_path=args.config,
            wandb_track=True,
            wandb_tags=[args.tag],          # group the whole sweep under one tag for plotting
            async_eval=True,
            bf16_train=True,
            compile_mode='reduce-overhead',
            train_iterations=args.iters,
            sample_freq=10 ** 9,
            final_sample=False,
            test_eval_freq=500,
            sample_eval_freq=5000,
            c_residual=c_res,
            # --- B0 recipe (linear attention; full_spatial_attn defaults False) ---
            model_dim=args.model_dim,
            diff_steps=args.diff_steps,
            lr_schedule=args.lr_schedule,
            lr_step_size=args.lr_step_size,
            lr_step_burnin=args.lr_step_burnin,
            lr_step_gamma=0.5,
        )
        return run_trial_subprocess(overrides)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--center', type=float, default=1e-3)
    ap.add_argument('--lo', type=float, default=None)   # explicit range lower bound (overrides center)
    ap.add_argument('--hi', type=float, default=None)   # explicit range upper bound
    ap.add_argument('--tag', required=True)
    ap.add_argument('--study', required=True)
    ap.add_argument('--storage', default='sqlite:///pidm_optuna.db')
    ap.add_argument('--n-trials', type=int, default=5)
    ap.add_argument('--iters', type=int, default=35000)
    # B0 recipe knobs (defaults reproduce the B0 best-recipe config; linear attention)
    ap.add_argument('--model-dim', type=int, default=64)
    ap.add_argument('--diff-steps', type=int, default=50)
    ap.add_argument('--lr-schedule', default='step')
    ap.add_argument('--lr-step-size', type=int, default=40000)
    ap.add_argument('--lr-step-burnin', type=int, default=0)
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study, storage=args.storage, load_if_exists=True,
        direction='minimize', sampler=TPESampler(), pruner=NopPruner(),
    )
    # catch: a crashed trial subprocess marks that trial failed but the study continues
    study.optimize(build_objective(args), n_trials=args.n_trials, catch=(Exception,))
    print(f'[{args.study}] best value (residual_mean_abs_samples):', study.best_value)
    print(f'[{args.study}] best params:', study.best_params)


if __name__ == '__main__':
    main()
