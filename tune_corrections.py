"""Study 1: inference-time residual-correction sweep on a FROZEN checkpoint (no retraining).

Multi-objective (NSGA-II): minimize residual_mean_abs_samples AND a realism distance (how far the
generated field moments drift from the training-data moments) -- so the sweep can't 'cheat' by
over-correcting into unrealistic fields. Each trial is one corrected p_sample_loop pass.

Knobs (CoCoGen): N_correction (final diffusion steps corrected), M_correction (post-hoc steps),
correction_mode (xt/x0).

  python tune_corrections.py --ckpt trained_models/lrdecay_r0/model/checkpoint_50000.pt --n-trials 40
"""
import argparse, yaml, torch
from pathlib import Path

import optuna
from optuna.samplers import NSGAIISampler

from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.data_utils import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='trained_models/lrdecay_r0/model/checkpoint_50000.pt')
ap.add_argument('--config', default='configs/darcy_pidm_me.yaml')
ap.add_argument('--study', default='corrections')
ap.add_argument('--storage', default='sqlite:///pidm_corrections.db')
ap.add_argument('--n-trials', type=int, default=40)
ap.add_argument('--n-samples', type=int, default=16)
args = ap.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())

# frozen model + residuals + diffusion
model = Unet3D(dim=32, channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(args.ckpt, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()
residuals = ResidualsDarcy(model=model, fd_acc=cfg['fd_acc'], pixels_per_dim=64, pixels_at_boundary=True,
                           reverse_d1=True, device=device, bcs='none', domain_length=1.,
                           residual_grad_guidance=False,
                           use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)

# reference field moments from the training data (per channel)
ds = Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'), use_double=False)
ref = next(iter(DataLoader(ds, batch_size=512, shuffle=True))).to(device)
ref_mean = ref.mean(dim=(0, 2, 3))
ref_std = ref.std(dim=(0, 2, 3)).clamp_min(1e-8)
print(f'data ref per-channel mean={ref_mean.tolist()} std={ref_std.tolist()}', flush=True)


def evaluate(N, M, mode):
    out = diffusion.p_sample_loop(None, (args.n_samples, 2, 64, 64), save_output=True,
                                  surpress_noise=True, use_dynamic_threshold=False,
                                  residual_func=residuals, eval_residuals=True,
                                  return_optimizer=False, return_inequality=False,
                                  M_correction=M, N_correction=N, correction_mode=mode)
    res = float(out[1]['residual'].detach().abs().mean())
    field = out[0][0][-1].to(device)                 # final sample [B, 2, 64, 64]
    g_mean = field.mean(dim=(0, 2, 3))
    g_std = field.std(dim=(0, 2, 3))
    realism = float((((g_mean - ref_mean).abs() + (g_std - ref_std).abs()) / ref_std).mean())
    return res, realism


def objective(trial):
    N = trial.suggest_int('N_correction', 0, 100, step=10)
    M = trial.suggest_int('M_correction', 0, 50, step=5)
    mode = trial.suggest_categorical('correction_mode', ['xt', 'x0'])
    res, realism = evaluate(N, M, mode)
    trial.set_user_attr('residual', res)
    trial.set_user_attr('realism', realism)
    print(f't{trial.number}: N={N} M={M} mode={mode} -> residual={res:.4e} realism={realism:.4e}', flush=True)
    return res, realism


study = optuna.create_study(study_name=args.study, storage=args.storage, load_if_exists=True,
                            directions=['minimize', 'minimize'],
                            sampler=NSGAIISampler(population_size=16, seed=0))
study.optimize(objective, n_trials=args.n_trials)

print('=== PARETO FRONT (residual vs realism) ===', flush=True)
for t in sorted(study.best_trials, key=lambda t: t.values[0]):
    p = t.params
    print(f"  N={p['N_correction']:3d} M={p['M_correction']:2d} mode={p['correction_mode']:2s} "
          f"-> residual={t.values[0]:.4e}  realism={t.values[1]:.4e}", flush=True)
print('=== CORRECTIONS_DONE ===', flush=True)
