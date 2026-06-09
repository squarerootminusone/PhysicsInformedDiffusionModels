"""Measure <|residual|>_fd2 / <|residual|>_fd4 on the SAME generated samples of a checkpoint.
Used to recalibrate c_residual when ramping fd_acc 2->4 (keep the physics-loss magnitude continuous).

  python measure_fd_ratio.py <ckpt> [config] [n_samples]   ->  prints RATIO=<fd2/fd4>
"""
import sys, yaml, torch
from pathlib import Path
from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = sys.argv[1]
CONFIG = sys.argv[2] if len(sys.argv) > 2 else 'configs/darcy_pidm_me.yaml'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32

cfg = yaml.safe_load(Path(CONFIG).read_text())
model = Unet3D(dim=32, channels=2, sigmoid_last_channel=False).to(device)
state = torch.load(CKPT, map_location='cpu')['model']
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model.eval()


def mk_res(fd):
    return ResidualsDarcy(model=model, fd_acc=fd, pixels_per_dim=64, pixels_at_boundary=True,
                          reverse_d1=True, device=device, bcs='none', domain_length=1.,
                          residual_grad_guidance=False,
                          use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])


res2 = mk_res(2)
diffusion = DenoisingDiffusion(cfg['diff_steps'], device, False)
out = diffusion.p_sample_loop(None, (N, 2, 64, 64), save_output=True, surpress_noise=True,
                              use_dynamic_threshold=False, residual_func=res2, eval_residuals=True,
                              return_optimizer=False, return_inequality=False,
                              M_correction=0, N_correction=0, correction_mode='xt')
field = out[0][0][-1].to(device)                                 # same samples for both stencils

r2 = res2.compute_residual(field, pass_through=True)['residual'].detach().abs().mean().item()
r4 = mk_res(4).compute_residual(field, pass_through=True)['residual'].detach().abs().mean().item()
print(f'fd2 mean|res|={r2:.4e}  fd4 mean|res|={r4:.4e}  ratio(fd2/fd4)={r2 / r4:.4f}', flush=True)
print(f'RATIO={r2 / r4:.6f}', flush=True)
