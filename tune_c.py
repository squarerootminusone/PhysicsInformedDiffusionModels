"""Study A: single-parameter c_residual search for ONE config.

Tunes c_residual log-uniformly in [center/10, center*10], no pruning (each trial runs the
full --iters steps), residual_mean_abs_test as the objective. Every wandb run name contains
"optuna-c". SQLite storage -> resumable.

  python tune_c.py --config configs/darcy_pidm_me.yaml --center 1e-3 --tag me --study cres_me --n-trials 5
"""
import argparse
import os
import subprocess

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner

import main as pidm

os.environ.setdefault(
    'WANDB_GIT_COMMIT',
    subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
)


def build_objective(args):
    lo, hi = args.center / 10.0, args.center * 10.0

    def objective(trial):
        c_res = trial.suggest_float('c_residual', lo, hi, log=True)
        return pidm.train(dict(
            name=f'optuna-c_{args.tag}_t{trial.number}',
            config_path=args.config,
            wandb_track=True,
            async_eval=True,
            bf16_train=True,            # bf16 training forward; eval stays fp32
            train_iterations=args.iters,
            sample_freq=10 ** 9,        # no checkpoint/PNG sampler
            final_sample=False,         # no end-of-run sampler
            test_eval_freq=500,
            sample_eval_freq=5000,      # async residual_mean_abs_samples -> objective (overfit-robust)
            c_residual=c_res,
        ), trial=trial)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--center', type=float, required=True)
    ap.add_argument('--tag', required=True)          # short label for wandb names, e.g. 'me'/'se'
    ap.add_argument('--study', required=True)
    ap.add_argument('--storage', default='sqlite:///pidm_optuna.db')
    ap.add_argument('--n-trials', type=int, default=5)
    ap.add_argument('--iters', type=int, default=35000)
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study, storage=args.storage, load_if_exists=True,
        direction='minimize', sampler=TPESampler(), pruner=NopPruner(),
    )
    study.optimize(build_objective(args), n_trials=args.n_trials)
    print(f'[{args.study}] best value (residual_mean_abs_test):', study.best_value)
    print(f'[{args.study}] best params:', study.best_params)


if __name__ == '__main__':
    main()
