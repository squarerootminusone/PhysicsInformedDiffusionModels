"""opt16 validation: matrix-free K@u residual must match the dense-assembly residual.

Compares residual, compliance (return_optimizer), and gradients w.r.t. the model output
between the dense path and OPT16_MATRIX_FREE=1 on a fixed random batch (pass_through mode,
no model needed). Run on CPU so it does not disturb GPU training.

  python test_opt16_equivalence.py [data/mechanics/solidspy_k_no_BC/]
"""
import os, sys, torch

NO_BC = sys.argv[1] if len(sys.argv) > 1 else 'data/mechanics/solidspy_k_no_BC/'
device = 'cpu'

from src.residuals_mechanics_K import ResidualsMechanics

res = ResidualsMechanics(model=None, pixels_per_dim=64, pixels_at_boundary=True,
                         no_BC_folder=NO_BC, device=device, topopt_eval=False)

B = 2
torch.manual_seed(0)
x0 = torch.randn(B, 3, 64, 64, device=device)
x0[:, 2] = torch.rand(B, 64, 64, device=device)          # rho in [0,1] (SIMP density)

bcs = torch.zeros(B, 4, 65, 65, device=device)
bcs[:, 0, :, 0] = 1.                                      # clamp left edge x
bcs[:, 1, :, 0] = 1.                                      # clamp left edge y
bcs[:, 2, 32, 64] = 0.5                                   # point load x at right edge
bcs[:, 3, 10, 64] = -0.25                                 # point load y
vf = torch.full((B,), 0.4, device=device)


def run(matrix_free):
    os.environ['OPT16_MATRIX_FREE'] = '1' if matrix_free else ''
    x = x0.detach().clone().requires_grad_(True)
    out = res.compute_residual((x, bcs, vf), pass_through=True,
                               return_model_out=False, return_optimizer=True)
    r, c = out['residual'], out['optimizer']
    (r.pow(2).sum() + c.sum()).backward()
    return r.detach(), c.detach(), x.grad.detach().clone()


r_d, c_d, g_d = run(False)
r_m, c_m, g_m = run(True)

def report(name, a, b, atol):
    diff = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=1e-5)
    print(f'{name}: max|diff|={diff:.3e}  allclose={ok}')
    return ok

ok = True
ok &= report('residual  ', r_d, r_m, 1e-5)
ok &= report('compliance', c_d, c_m, 1e-4)
ok &= report('grad      ', g_d, g_m, 1e-5)
print('OPT16_EQUIVALENCE ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
