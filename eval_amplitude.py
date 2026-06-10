"""Pressure/permeability amplitude check (PBFM reward-hacking diagnostic).

PBFM (arXiv:2506.08604) claims PIDM-ME's low Darcy residuals partly come from generated
pressure fields collapsed into a narrow range (±0.2), which shrinks the residual mechanically.
This script compares per-field statistics (std, quantiles, range) of generated samples vs the
training data, and reports the residual both raw and normalized by the amplitude ratio.

  python eval_amplitude.py <checkpoint.pt> [n_samples] [N_correction]
"""
import sys, yaml, torch
from pathlib import Path

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
N_CORR = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CONFIG_PATH = str(Path(CKPT).parent / 'model.yaml')

from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.data_utils import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())

model = Unet3D(dim=cfg.get('model_dim', 32), channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(CKPT, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()
residuals = ResidualsDarcy(model=model, fd_acc=cfg['fd_acc'], pixels_per_dim=64,
                           pixels_at_boundary=True, reverse_d1=True, device=device, bcs='none',
                           domain_length=1., residual_grad_guidance=False,
                           use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)

torch.manual_seed(0)
out = diffusion.p_sample_loop(None, (N, 2, 64, 64), save_output=True, surpress_noise=True,
                              use_dynamic_threshold=False, residual_func=residuals,
                              eval_residuals=True, return_optimizer=False, return_inequality=False,
                              M_correction=0, N_correction=N_CORR,
                              correction_mode=('xt' if N_CORR else 'none'))
gen = out[0][0][-1].to(device)                       # [N, 2, 64, 64]
res = out[1]['residual'].detach().abs().mean()

ds = Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'), use_double=False)
ref = next(iter(DataLoader(ds, batch_size=2048, shuffle=True))).to(device)  # [B, 2, 64, 64]

Q = torch.tensor([0.005, 0.05, 0.5, 0.95, 0.995], device=device)
print(f'checkpoint: {CKPT}  n_samples={N}  N_correction={N_CORR}')
ratios = {}
for ci, name in enumerate(('p', 'K')):
    g, r = gen[:, ci].flatten(), ref[:, ci].flatten()
    gq, rq = torch.quantile(g, Q), torch.quantile(r, Q)
    ratios[name] = (g.std() / r.std()).item()
    print(f'[{name}] std  gen={g.std():.4f}  data={r.std():.4f}  ratio={ratios[name]:.3f}')
    print(f'[{name}] range gen=[{g.min():.3f},{g.max():.3f}]  data=[{r.min():.3f},{r.max():.3f}]')
    print(f'[{name}] quantiles {Q.tolist()}')
    print(f'[{name}]   gen : {[f"{v:.3f}" for v in gq.tolist()]}')
    print(f'[{name}]   data: {[f"{v:.3f}" for v in rq.tolist()]}')

print(f'residual_mean_abs (raw)               = {res.item():.4e}')
print(f'residual_mean_abs / p-amplitude-ratio = {res.item() / max(ratios["p"], 1e-6):.4e}')
verdict = 'COLLAPSED (reward hacking likely)' if ratios['p'] < 0.8 else \
          'mild compression' if ratios['p'] < 0.95 else 'faithful amplitude'
print(f'AMPLITUDE_VERDICT: p-std ratio {ratios["p"]:.3f} -> {verdict}')
