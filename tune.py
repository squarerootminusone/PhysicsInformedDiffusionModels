"""Optuna hyperparameter search for PIDM (Darcy).

Drives the refactored main.train(overrides, trial) callable. Each trial trains for a short
HPO budget with the heavy 20k sampler disabled, reports residual_mean_abs_test every
test_eval_freq iters, and is pruned early by Hyperband. Retrain the winner at full length.

Run (on the VM, single GPU):
    export WANDB_GIT_COMMIT=$(git rev-parse HEAD)
    python tune.py --n-trials 20

Parallel trials share the SQLite study (the tiny model uses <9 GB of the 98 GB card):
    for i in 1 2 3; do nohup python tune.py --n-trials 20 >logs/optuna_$i.log 2>&1 & done

Monitor:
    optuna-dashboard sqlite:///pidm_optuna.db
"""
import argparse
import os
import subprocess

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import main as pidm


# Honor the CLAUDE.md rule: every wandb-tracked run records the actual commit.
os.environ.setdefault(
    'WANDB_GIT_COMMIT',
    (lambda: subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
     .stdout.strip())()
)

HPO_ITERATIONS = 15000   # short budget per trial; retrain the winner at full length


def objective(trial):
    overrides = dict(
        name=f'optuna_t{trial.number}',
        config_path='configs/darcy_pidm_me.yaml',
        wandb_track=True,
        async_eval=True,
        train_iterations=HPO_ITERATIONS,
        sample_freq=10 ** 9,        # disable the heavy GPU sampler during search
        final_sample=False,         # ...including the end-of-run sample+checkpoint
        test_eval_freq=500,         # -> ~30 prune checkpoints per trial
        # --- search space (grounded in the actual knobs) ---
        lr=trial.suggest_float('lr', 1e-5, 5e-4, log=True),
        c_residual=trial.suggest_float('c_residual', 1e-4, 1e-1, log=True),
        diff_steps=trial.suggest_categorical('diff_steps', [50, 100, 200]),
        fd_acc=trial.suggest_categorical('fd_acc', [2, 4]),
        ema_decay=trial.suggest_categorical('ema_decay', [0.99, 0.999, 0.9999]),
        grad_clip=trial.suggest_float('grad_clip', 0.5, 5.0),
    )
    # train() returns best residual_mean_abs_test (unweighted -> comparable across c_residual)
    return pidm.train(overrides, trial=trial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='pidm_darcy')
    ap.add_argument('--storage', default='sqlite:///pidm_optuna.db')
    ap.add_argument('--n-trials', type=int, default=20)
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,            # resumable + shareable across parallel workers
        direction='minimize',
        sampler=TPESampler(multivariate=True),
        pruner=HyperbandPruner(min_resource=1000, max_resource=HPO_ITERATIONS,
                               reduction_factor=3),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print('Best value (residual_mean_abs_test):', study.best_value)
    print('Best params:', study.best_params)


if __name__ == '__main__':
    main()
