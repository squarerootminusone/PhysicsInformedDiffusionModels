"""Study B: one shared multiplier applied to BOTH configs at once.

Each trial picks a single factor m in [0.1, 10] (log) and trains both configs with
c_residual = center * m (mean at 1e-3*m, DDIM at 1e-5*m), each in its OWN isolated subprocess
(cudagraphs safe). The trial score is the mean of the two best residual_mean_abs_samples.
Every wandb run name contains "optuna-c".

  python tune_c_multiplier.py --study cres_mult --n-trials 10
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

# (tag, config, center c_residual)
CONFIGS = [
    ('me', 'configs/darcy_pidm_me.yaml', 1e-3),
    ('se', 'configs/darcy_pidm_se.yaml', 1e-5),
]


def run_trial_subprocess(overrides):
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
    def objective(trial):
        m = trial.suggest_float('c_residual_mult', 0.1, 10.0, log=True)
        scores = []
        for tag, cfg, center in CONFIGS:
            overrides = dict(
                name=f'optuna-c_B_t{trial.number}_{tag}',
                config_path=cfg,
                wandb_track=True,
                async_eval=True,
                bf16_train=True,
                train_iterations=args.iters,
                sample_freq=10 ** 9,
                final_sample=False,
                test_eval_freq=500,
                sample_eval_freq=5000,
                c_residual=center * m,
            )
            s = run_trial_subprocess(overrides)
            scores.append(s)
            trial.set_user_attr(f'residual_{tag}', s)
        return sum(scores) / len(scores)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='cres_mult')
    ap.add_argument('--storage', default='sqlite:///pidm_optuna.db')
    ap.add_argument('--n-trials', type=int, default=10)
    ap.add_argument('--iters', type=int, default=35000)
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study, storage=args.storage, load_if_exists=True,
        direction='minimize', sampler=TPESampler(), pruner=NopPruner(),
    )
    study.optimize(build_objective(args), n_trials=args.n_trials, catch=(Exception,))
    print(f'[{args.study}] best mean residual:', study.best_value)
    print(f'[{args.study}] best params:', study.best_params)


if __name__ == '__main__':
    main()
