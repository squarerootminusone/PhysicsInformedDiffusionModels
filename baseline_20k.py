"""Baseline 20k run on default hyperparameters (PIDM-ME), outside Optuna.
Same recipe as the studies: bf16 train + fp32 eval, sample-eval every 5k. Default
c_residual (1e-3 from the config) and default lr/grad_clip/ema. Reference curve to
compare the swept trials against.
"""
import main

main.train({
    'name': 'baseline_default_20k_me',
    'config_path': 'configs/darcy_pidm_me.yaml',
    'train_iterations': 20000,
    'bf16_train': True,
    'async_eval': True,
    'sample_eval_freq': 5000,
    'test_eval_freq': 500,
    'sample_freq': 10 ** 9,
    'final_sample': False,
    'wandb_track': True,
})
