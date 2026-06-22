"""Fixed-list N_correction sweep on a frozen checkpoint (inference only — no training).
For each N in --n-list, runs p_sample_loop with N residual-correction steps and reports the
sampled residual_mean_abs and the sliced-Wasserstein distance to the training distribution.
Respects B1_LINESEARCH=1 (exact line-search correction step) via residuals_darcy.

  B1_LINESEARCH=1 python eval_ncorrection.py --ckpt <ckpt.pt> --mode xt --n-list 20,40,60,80,100
"""
import argparse, yaml, torch
import torch.nn.functional as F
from pathlib import Path
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.data_utils import Dataset
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--config', default=None)            # defaults to ckpt's sibling model.yaml
ap.add_argument('--n-samples', type=int, default=64)
ap.add_argument('--mode', default='xt')              # correction_mode: 'xt' or 'x0'
ap.add_argument('--n-list', default='20,40,60,80,100')
ap.add_argument('--n-ref', type=int, default=1024)
ap.add_argument('--n-proj', type=int, default=128)
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config or str(Path(args.ckpt).parent / 'model.yaml')
cfg = yaml.safe_load(Path(cfg_path).read_text())

model = Unet3D(dim=cfg.get('model_dim', 32), channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(args.ckpt, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()
residuals = ResidualsDarcy(model=model, fd_acc=cfg['fd_acc'], pixels_per_dim=64, pixels_at_boundary=True,
                           reverse_d1=True, device=device, bcs='none', domain_length=1.,
                           residual_grad_guidance=False,
                           use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)

# sliced-Wasserstein vs training distribution (fixed projections + ref quantiles)
ds = Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'), use_double=False)
ref = next(iter(DataLoader(ds, batch_size=args.n_ref, shuffle=True))).to(device).flatten(1)
torch.manual_seed(0)
proj = F.normalize(torch.randn(args.n_proj, ref.shape[1], device=device), dim=1)
QLEV = torch.linspace(0., 1., 128, device=device)
ref_q = torch.quantile(ref @ proj.t(), QLEV, dim=0)


def swd(gen_flat):
    gen_q = torch.quantile(gen_flat @ proj.t(), QLEV, dim=0)
    return float((gen_q - ref_q).abs().mean())


def evaluate(N, mode):
    out = diffusion.p_sample_loop(None, (args.n_samples, 2, 64, 64), save_output=True,
                                  surpress_noise=True, use_dynamic_threshold=False,
                                  residual_func=residuals, eval_residuals=True,
                                  return_optimizer=False, return_inequality=False,
                                  M_correction=0, N_correction=N, correction_mode=mode)
    res = float(out[1]['residual'].detach().abs().mean())
    field = out[0][0][-1].to(device).flatten(1)
    return res, swd(field)


import os
print(f'checkpoint: {args.ckpt}  n_samples={args.n_samples}  mode={args.mode}  '
      f'B1_LINESEARCH={os.environ.get("B1_LINESEARCH","")}', flush=True)
r0, s0 = evaluate(0, args.mode)
print(f'N=  0 (baseline): residual={r0:.4e}  swd={s0:.4e}', flush=True)
for N in [int(x) for x in args.n_list.split(',')]:
    r, s = evaluate(N, args.mode)
    print(f'N={N:3d}: residual={r:.4e}  swd={s:.4e}', flush=True)
print('NCORR_SWEEP_DONE', flush=True)
