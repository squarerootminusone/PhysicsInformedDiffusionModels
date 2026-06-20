"""Time the dim128 mechanics UNet forward+backward alone (random [B,10,64,64]).
Combine with the measured residual fwd+bwd times to get the end-to-end opt16 speedup:
  step_ms ~= model_ms + residual_ms ;  speedup = (model+dense)/(model+matrixfree).
"""
import sys, time, torch
from src.unet_model import Unet3D
device = 'cuda'
out_dim = 3

def bench(B, iters=30):
    torch.manual_seed(0)
    m = Unet3D(dim=128, channels=out_dim+3+4, out_dim=out_dim, sigmoid_last_channel=True).to(device)
    x = torch.randn(B, 10, 64, 64, device=device)
    t = torch.randint(0, 100, (B,), device=device)
    def step():
        m.zero_grad(set_to_none=True)
        out = m(x, t)
        out.square().mean().backward()
    for _ in range(5): step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): step()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3

# measured residual fwd+bwd (ms) from bench_opt16 on this box
RES = {6: {'dense': 27.10, 'mf': 0.74}, 32: {'dense': 150.58, 'mf': 1.11}}
for B in (6, 32):
    mo = bench(B)
    d, mf = RES[B]['dense'], RES[B]['mf']
    print(f'B={B}: model fwd+bwd = {mo:.1f} ms', flush=True)
    print(f'   dense step ~= {mo+d:.1f} ms = {1000/(mo+d):.2f} it/s', flush=True)
    print(f'   mf    step ~= {mo+mf:.1f} ms = {1000/(mo+mf):.2f} it/s', flush=True)
    print(f'   -> end-to-end speedup = {(mo+d)/(mo+mf):.2f}x', flush=True)
print('MODEL_ONLY_DONE')
