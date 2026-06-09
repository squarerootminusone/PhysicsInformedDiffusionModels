"""Generate samples from a checkpoint on CPU (no GPU contention) and plot generated vs real fields.
  python gen_samples_cpu.py <ckpt> <config> <out.png> [n_samples]
"""
import sys, yaml, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.data_utils import Dataset

device = torch.device('cpu')
torch.set_num_threads(max(1, torch.get_num_threads()))
CKPT = sys.argv[1]
CONFIG = sys.argv[2]
OUT = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4

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

torch.manual_seed(0)
print(f'generating {N} samples on CPU ({cfg["diff_steps"]} steps, dim={dim})...', flush=True)
out = diffusion.p_sample_loop(None, (N, 2, 64, 64), save_output=True, surpress_noise=True,
                              use_dynamic_threshold=False, residual_func=residuals, eval_residuals=True,
                              return_optimizer=False, return_inequality=False,
                              M_correction=0, N_correction=0, correction_mode='xt')
gen = out[0][0][-1]                                              # [N,2,64,64]
resid = out[1]['residual'].detach().abs().mean(dim=tuple(range(1, out[1]['residual'].ndim)))

real = next(iter(DataLoader(Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'),
                                    use_double=False), batch_size=N, shuffle=True)))

rows = [('gen p', gen[:, 0]), ('gen K', gen[:, 1]), ('real p', real[:, 0]), ('real K', real[:, 1])]
fig, axes = plt.subplots(4, N, figsize=(3 * N, 11))
for r, (label, data) in enumerate(rows):
    for i in range(N):
        ax = axes[r, i]
        im = ax.imshow(data[i].numpy(), cmap='viridis')
        ttl = f'{label} #{i}' + (f'  res={resid[i]:.1e}' if label == 'gen p' else '')
        ax.set_title(ttl, fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
plt.suptitle(f'model_dim={dim} generated vs real Darcy fields (CPU)', y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f'saved {OUT}', flush=True)
