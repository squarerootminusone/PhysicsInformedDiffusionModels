"""opt16 END-TO-END: time a full mechanics TRAINING step (model fwd + residual + loss + backward
+ optimizer) with the dense vs matrix-free residual, to get the real iter/s speedup (not just the
residual-in-isolation number). Uses random data of the correct shape — timing is data-independent.

  python bench_opt16_e2e.py [data/mechanics/solidspy_k_no_BC/]
"""
import os, sys, time, torch
import torch.optim as optim

NO_BC = sys.argv[1] if len(sys.argv) > 1 else 'data/mechanics/solidspy_k_no_BC/'
device = 'cuda'

from src.unet_model import Unet3D
from src.residuals_mechanics_K import ResidualsMechanics
from src.denoising_utils import DenoisingDiffusion

out_dim = 3
diffusion = DenoisingDiffusion(100, device, False)


def bench(B, matrix_free, iters=20):
    os.environ['OPT16_MATRIX_FREE'] = '1' if matrix_free else ''
    torch.manual_seed(0)
    model = Unet3D(dim=128, channels=out_dim + 3 + 4, out_dim=out_dim,
                   sigmoid_last_channel=True).to(device)
    res = ResidualsMechanics(model=model, pixels_per_dim=64, pixels_at_boundary=True,
                             no_BC_folder=NO_BC, device=device, topopt_eval=False)
    opt = optim.Adam(model.parameters(), lr=1e-4, fused=True)
    inp = torch.randn(B, 10, 64, 64, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss, *_ = diffusion.model_estimation_loss(inp, residual_func=res,
                                                   c_data=1., c_residual=0.00374)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, res, opt
    torch.cuda.empty_cache()
    return ms, peak


for B in (6, 32):
    row = {}
    for mf in (False, True):
        try:
            ms, peak = bench(B, mf)
            row[mf] = ms
            print(f'B={B:3d} matrix_free={mf}: {ms:8.1f} ms/step = {1000/ms:5.2f} it/s   peak={peak:8.0f} MiB', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'B={B:3d} matrix_free={mf}: OOM', flush=True)
            torch.cuda.empty_cache()
    if False in row and True in row:
        print(f'  -> end-to-end speedup at B={B}: {row[False]/row[True]:.2f}x  '
              f'({1000/row[False]:.2f} -> {1000/row[True]:.2f} it/s)', flush=True)
print('BENCH_OPT16_E2E_DONE')
