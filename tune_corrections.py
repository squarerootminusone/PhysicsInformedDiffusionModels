"""Study 1 (revised): sweep ONLY N_correction -- in-loop physics guidance -- on a FROZEN checkpoint.

Multi-objective (NSGA-II): minimize residual_mean_abs_samples vs the sliced-Wasserstein distance
between the generated-field distribution and the training-data distribution. SWD is a real
distribution-shape metric (sensitive to the whole shape, not just mean/std), so it detects samples
leaving the data manifold -- which moment matching cannot.

M_correction is fixed at 0: post-hoc projection is a Goodhart trap on the residual, so it is NOT a
hyperparameter to optimize. N_correction is in-loop guidance (subsequent diffusion steps re-naturalize),
so it is the defensible knob.

  python tune_corrections.py --ckpt trained_models/lrdecay_r0/model/checkpoint_50000.pt --n-trials 24
"""
import argparse, yaml, torch
import torch.nn.functional as F
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
ap.add_argument('--config', default=None,
                help="run config; defaults to the checkpoint's sibling model.yaml (self-describing model_dim)")
ap.add_argument('--study', default='corrections_n')
ap.add_argument('--storage', default='sqlite:///pidm_corrections.db')
ap.add_argument('--n-trials', type=int, default=24)
ap.add_argument('--n-samples', type=int, default=24)
ap.add_argument('--n-ref', type=int, default=1024)
ap.add_argument('--n-proj', type=int, default=128)
args = ap.parse_args()

config_path = args.config if args.config else str(Path(args.ckpt).parent / 'model.yaml')
cfg = yaml.safe_load(Path(config_path).read_text())

model = Unet3D(dim=cfg.get('model_dim', 32), channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(args.ckpt, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()
residuals = ResidualsDarcy(model=model, fd_acc=cfg['fd_acc'], pixels_per_dim=64, pixels_at_boundary=True,
                           reverse_d1=True, device=device, bcs='none', domain_length=1.,
                           residual_grad_guidance=False,
                           use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)

# --- sliced-Wasserstein setup: reference data distribution + fixed random projection lines ---
ds = Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'), use_double=False)
ref = next(iter(DataLoader(ds, batch_size=args.n_ref, shuffle=True))).to(device).flatten(1)  # [Nr, 8192]
D = ref.shape[1]
torch.manual_seed(0)
proj = F.normalize(torch.randn(args.n_proj, D, device=device), dim=1)   # fixed lines, reused every trial
QLEV = torch.linspace(0., 1., 128, device=device)
ref_q = torch.quantile(ref @ proj.t(), QLEV, dim=0)                     # [128, n_proj], precomputed


def sliced_wasserstein(gen_flat):
    # project gen onto the same lines, compare quantile functions to the data's (1-D W1, averaged)
    gen_q = torch.quantile(gen_flat @ proj.t(), QLEV, dim=0)
    return (gen_q - ref_q).abs().mean().item()


def evaluate(N, mode):
    out = diffusion.p_sample_loop(None, (args.n_samples, 2, 64, 64), save_output=True,
                                  surpress_noise=True, use_dynamic_threshold=False,
                                  residual_func=residuals, eval_residuals=True,
                                  return_optimizer=False, return_inequality=False,
                                  M_correction=0, N_correction=N, correction_mode=mode)
    res = float(out[1]['residual'].detach().abs().mean())
    field = out[0][0][-1].to(device).flatten(1)        # final samples [B, 8192]
    return res, sliced_wasserstein(field)


def objective(trial):
    N = trial.suggest_int('N_correction', 0, 100, step=10)
    mode = trial.suggest_categorical('correction_mode', ['xt', 'x0'])
    res, swd = evaluate(N, mode)
    trial.set_user_attr('residual', res)
    trial.set_user_attr('swd', swd)
    print(f't{trial.number}: N={N} mode={mode} -> residual={res:.4e} swd={swd:.4e}', flush=True)
    return res, swd


study = optuna.create_study(study_name=args.study, storage=args.storage, load_if_exists=True,
                            directions=['minimize', 'minimize'],
                            sampler=NSGAIISampler(population_size=12, seed=0))
study.enqueue_trial({'N_correction': 0, 'correction_mode': 'xt'})       # uncorrected baseline anchor
study.optimize(objective, n_trials=args.n_trials)

print('=== PARETO FRONT (residual vs sliced-Wasserstein) ===', flush=True)
for t in sorted(study.best_trials, key=lambda t: t.values[0]):
    p = t.params
    print(f"  N={p['N_correction']:3d} mode={p['correction_mode']:2s} "
          f"-> residual={t.values[0]:.4e}  swd={t.values[1]:.4e}", flush=True)
print('=== CORRECTIONS_DONE ===', flush=True)
