"""opt16 micro-benchmark: dense vs matrix-free mechanics residual on GPU, batch 6 and 32.

Times compute_residual (pass_through, forward+backward) in isolation — no dataset/model needed.

  python bench_opt16.py [data/mechanics/solidspy_k_no_BC/]
"""
import os, sys, time, torch

NO_BC = sys.argv[1] if len(sys.argv) > 1 else 'data/mechanics/solidspy_k_no_BC/'
device = 'cuda'

from src.residuals_mechanics_K import ResidualsMechanics

res = ResidualsMechanics(model=None, pixels_per_dim=64, pixels_at_boundary=True,
                         no_BC_folder=NO_BC, device=device, topopt_eval=False)


def bench(B, matrix_free, iters=30):
    os.environ['OPT16_MATRIX_FREE'] = '1' if matrix_free else ''
    torch.manual_seed(0)
    x0 = torch.randn(B, 3, 64, 64, device=device)
    x0[:, 2] = torch.rand(B, 64, 64, device=device)
    bcs = torch.zeros(B, 4, 65, 65, device=device)
    bcs[:, 0, :, 0] = 1.; bcs[:, 1, :, 0] = 1.
    bcs[:, 2, 32, 64] = 0.5; bcs[:, 3, 10, 64] = -0.25
    vf = torch.full((B,), 0.4, device=device)

    def step():
        x = x0.detach().clone().requires_grad_(True)
        out = res.compute_residual((x, bcs, vf), pass_through=True,
                                   return_model_out=False, return_optimizer=True)
        (out['residual'].pow(2).sum() + out['optimizer'].sum()).backward()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**20
    return dt * 1e3, peak


for B in (6, 32):
    for mf in (False, True):
        try:
            ms, peak = bench(B, mf)
            print(f'B={B:3d} matrix_free={mf}: {ms:8.2f} ms/iter  peak={peak:8.1f} MiB', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'B={B:3d} matrix_free={mf}: OOM', flush=True)
        torch.cuda.empty_cache()
print('BENCH_OPT16_DONE')
