"""Verify bf16-train + fp32-eval for PIDM-ME:
  1. it/s (compiled): bf16 training forward vs fp32 training forward.
  2. residual on identical weights: fp32 eval vs bf16 eval (fp32 should be lower; the gap
     grows toward ~2x as the model converges).
"""
import time, yaml, torch
from pathlib import Path
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

torch.set_float32_matmul_precision('highest')
cfg = yaml.safe_load(Path('configs/darcy_pidm_me.yaml').read_text())
diff_steps, fd_acc = cfg['diff_steps'], cfg['fd_acc']
lk = dict(c_data=cfg['c_data'], c_residual=cfg['c_residual'], c_ineq=cfg['c_ineq'], lambda_opt=cfg['lambda_opt'])

ds = Dataset(('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv'), use_double=False)
dl = cycle(DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True))

def mk_model():
    return Unet3D(dim=32, channels=2, sigmoid_last_channel=False).to(device)

def mk_res(m):
    return ResidualsDarcy(model=m, fd_acc=fd_acc, pixels_per_dim=64, pixels_at_boundary=True,
                          reverse_d1=True, device=device, bcs='none', domain_length=1.,
                          residual_grad_guidance=False, use_ddim_x0=False, ddim_steps=0)

model = torch.compile(mk_model(), mode='reduce-overhead')
residuals = mk_res(model)
diffusion = DenoisingDiffusion(diff_steps, device, False)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)

def train_steps(n, bf16, measure=False):
    model.train()
    if measure:
        torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        torch.compiler.cudagraph_mark_step_begin()
        b = next(dl).to(device, non_blocking=True)
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else torch.autocast('cuda', enabled=False)
        with ctx:
            loss, *_ = diffusion.model_estimation_loss(b, residual_func=residuals, **lk)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if measure:
        torch.cuda.synchronize(); return n / (time.time() - t)

print('training 8000 bf16 steps (warmup + partial convergence)...', flush=True)
train_steps(8000, bf16=True)
its_bf16 = train_steps(300, bf16=True, measure=True)
its_fp32 = train_steps(300, bf16=False, measure=True)
print(f'COMPILED it/s:  bf16={its_bf16:.1f}  fp32={its_fp32:.1f}  speedup={its_bf16/its_fp32:.2f}x', flush=True)

# residual on identical weights via an uncompiled copy: fp32 eval vs bf16 eval
ev = mk_model()
ev.load_state_dict({k.replace('_orig_mod.', ''): v.detach().clone() for k, v in model.state_dict().items()}, strict=False)
ev.eval()
ev_res = mk_res(ev)
ev_diff = DenoisingDiffusion(diff_steps, device, False)
ss = (8, 2, 64, 64)

def samp_residual(bf16):
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else torch.autocast('cuda', enabled=False)
    with ctx:
        out = ev_diff.p_sample_loop(None, ss, save_output=True, surpress_noise=True,
                                    use_dynamic_threshold=False, residual_func=ev_res, eval_residuals=True,
                                    return_optimizer=False, return_inequality=False,
                                    M_correction=0, N_correction=0, correction_mode='xt')
        return float(out[1]['residual'].abs().mean())

r_fp32 = samp_residual(False)
r_bf16 = samp_residual(True)
print(f'RESIDUAL @8k steps:  fp32_eval={r_fp32:.4e}  bf16_eval={r_bf16:.4e}  bf16/fp32={r_bf16 / r_fp32:.2f}x', flush=True)
print('VERIFY_DONE', flush=True)
