"""Figure-3-style plot for a PIDM checkpoint: columns = Permeability K, Pressure p,
Residual R_MAE(K,p) as a log-scale spatial heatmap. CPU-only.
  python fig3_plot.py <ckpt> <config> <out.png> [n_rows]
"""
import sys, yaml, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

device = torch.device('cpu')
CKPT = sys.argv[1]; CONFIG = sys.argv[2]; OUT = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3

cfg = yaml.safe_load(Path(CONFIG).read_text())
dim = cfg.get('model_dim', 32)
model = Unet3D(dim=dim, channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(CKPT, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()
residuals = ResidualsDarcy(model=model, fd_acc=cfg['fd_acc'], pixels_per_dim=64, pixels_at_boundary=True,
                           reverse_d1=True, device=device, bcs='none', domain_length=1.,
                           residual_grad_guidance=False,
                           use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)

torch.manual_seed(1)
print(f'generating {N} samples on CPU (dim={dim}, {cfg["diff_steps"]} steps)...', flush=True)
out = diffusion.p_sample_loop(None, (N, 2, 64, 64), save_output=True, surpress_noise=True,
                              use_dynamic_threshold=False, residual_func=residuals, eval_residuals=True,
                              return_optimizer=False, return_inequality=False,
                              M_correction=0, N_correction=0, correction_mode='xt')
gen = out[0][0][-1]                                          # [N,2,64,64]  (ch0=p, ch1=K)
rr = out[1]['residual'].detach().abs()
if rr.ndim == 3:        res_map = rr.mean(dim=-1).reshape(N, 64, 64)   # [N, n_points, n_comp]
elif rr.ndim == 4:      res_map = rr.mean(dim=1)                       # [N, C, H, W]
else:                   res_map = rr.reshape(N, 64, 64)
res_map = res_map.clamp_min(1e-5)
print('residual map range:', float(res_map.min()), float(res_map.max()), flush=True)

ext = [0, 1, 0, 1]
cols = [('Permeability $K$',           gen[:, 1], 'viridis', None),
        ('Pressure $p$',               gen[:, 0], 'viridis', None),
        (r'Residual $\mathcal{R}_{MAE}(K,p)$', res_map, 'magma',
         LogNorm(vmin=1e-3, vmax=max(1e-1, float(res_map.max()))))]   # paper Fig 3 residual scale

fig, axes = plt.subplots(N, 3, figsize=(11, 3.3 * N))
if N == 1:
    axes = axes[None, :]
for i in range(N):
    for c, (title, data, cmap, norm) in enumerate(cols):
        ax = axes[i, c]
        im = ax.imshow(data[i].numpy(), origin='lower', extent=ext, cmap=cmap, norm=norm)
        if i == 0:
            ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$\xi_1$'); ax.set_ylabel(r'$\xi_2$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
plt.suptitle(f'PIDM (model_dim={dim}) — generated fields + residual map (Fig. 3 style)', y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'saved {OUT}', flush=True)
