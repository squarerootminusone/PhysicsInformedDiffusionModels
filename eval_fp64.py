"""Evaluate a saved checkpoint's generated-sample residual in fp64 vs fp32 (Darcy).

The residual finite-difference stencils difference similar-magnitude numbers, so at a low
residual floor fp32 rounding matters; fp64 gives the precise value. Runs the same p_sample_loop
+ residual eval the training uses, once per precision, on the same model.

  python eval_fp64.py <checkpoint.pt> [config_path] [n_samples]
"""
import sys, yaml, torch
from pathlib import Path

CKPT = sys.argv[1]
CONFIG_PATH = sys.argv[2] if len(sys.argv) > 2 else str(Path(CKPT).parent / 'model.yaml')
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
FD_ACC = int(sys.argv[4]) if len(sys.argv) > 4 else None   # override the eval stencil order (for fair comparison)

from src.denoising_utils import DenoisingDiffusion
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())
state = torch.load(CKPT, map_location='cpu')['model']
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}   # strip torch.compile prefix


def eval_precision(dtype):
    torch.set_default_dtype(dtype)                                    # schedule/noise/stencils in dtype
    torch.manual_seed(0)                                             # same seed both runs
    m = Unet3D(dim=32, channels=2, sigmoid_last_channel=False).to(device).to(dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    m.eval()
    res = ResidualsDarcy(model=m, fd_acc=(FD_ACC if FD_ACC is not None else cfg['fd_acc']),
                         pixels_per_dim=64, pixels_at_boundary=True,
                         reverse_d1=True, device=device, bcs='none', domain_length=1.,
                         residual_grad_guidance=False,
                         use_ddim_x0=(cfg['x0_estimation'] == 'sample'), ddim_steps=cfg['ddim_steps'])
    diff = DenoisingDiffusion(cfg['diff_steps'], device, False)
    out = diff.p_sample_loop(None, (N, 2, 64, 64), save_output=True, surpress_noise=True,
                             use_dynamic_threshold=False, residual_func=res, eval_residuals=True,
                             return_optimizer=False, return_inequality=False,
                             M_correction=0, N_correction=0, correction_mode='xt')
    r = out[1]['residual'].detach().abs().mean(dim=tuple(range(1, out[1]['residual'].ndim)))
    return float(r.double().mean()), float(r.double().median())


m32, md32 = eval_precision(torch.float32)
m64, md64 = eval_precision(torch.float64)
print(f'checkpoint: {CKPT}  (n_samples={N})')
print(f'  fp32 eval: mean={m32:.6e}  median={md32:.6e}')
print(f'  fp64 eval: mean={m64:.6e}  median={md64:.6e}')
print(f'  fp32/fp64 mean ratio: {m32 / m64:.4f}')
