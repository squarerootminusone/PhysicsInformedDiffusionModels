"""
Reproduce Fig 2 from Bastek et al. (ICLR 2025) — Physics-Informed Diffusion Models.

Two subplots (log y-scale):
  (a) Residual error RMAE over training
  (b) Test data loss over training

Data source: stdout training logs in slurm/logs/.

Metric note:
  The true per-iteration RMAE (residual.abs().mean(), stored as wandb metric
  residual_mean_abs_test) is NOT emitted to stdout. The combined test loss
  printed as "test loss at iteration N: X" equals:
      loss_test = c_data * data_loss + c_residual * gaussian_nll(residual)
  For Diffusion / PG-Diffusion / CoCoGen (c_residual = 0):
      loss_test = data_loss    [exact]
  For PIDM-ME (c_residual = 0.001) and PIDM-SE (c_residual = 1e-5):
      loss_test = data_loss + c_residual * gaussian_nll(residual)   [combined]

  Subplot (b) — test data loss:
      For c_residual=0 models: loss_test IS the data loss (exact).
      For PIDM-SE: data_loss dominates (c_residual tiny), good proxy.
      For PIDM-ME: loss_test is inflated by the physics penalty.

  Subplot (a) — residual RMAE:
      PIDM-ME and PIDM-SE only, using the explicit residual field from
      new-format logs: "[iter N] test_loss: X residual: X".
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slurm', 'logs')

# Colors and styles chosen to match the qualitative appearance of the paper's
# Fig 2 (5 perceptually distinct colors from matplotlib's Tab10 cycle).
MODELS = [
    {
        'name':       'Diffusion',
        'file':       'pidm_darcy_diffusion_9980528.out',
        'color':      '#1f77b4',   # Tab10 blue
        'ls':         '-',
        'lw':         1.5,
        'c_residual': 0.0,
    },
    {
        'name':       'PG-Diffusion',
        'file':       'pidm_darcy_pg_9980534.out',
        'color':      '#ff7f0e',   # Tab10 orange
        'ls':         '--',
        'lw':         1.5,
        'c_residual': 0.0,
    },
    {
        'name':       'CoCoGen',
        'file':       'pidm_darcy_cocogen_10004050.out',
        'color':      '#2ca02c',   # Tab10 green
        'ls':         '-.',
        'lw':         1.5,
        'c_residual': 0.0,
    },
    {
        'name':       'PIDM-ME',
        'file':       'pidm_darcy_pidm_me_9955859.out',
        'color':      '#d62728',   # Tab10 red
        'ls':         (0, (3, 1, 1, 1)),   # dense dash-dot
        'lw':         1.8,
        'c_residual': 0.001,
    },
    {
        'name':       'PIDM-SE',
        'file':       'pidm_darcy_pidm_se_10059383.out',
        'color':      '#9467bd',   # Tab10 purple
        'ls':         ':',
        'lw':         1.8,
        'c_residual': 1e-5,
    },
]

# PIDM-ME and PIDM-SE from newer runs that emit explicit residual RMAE.
# Used only for subplot (a).
RESIDUAL_MODELS = [
    {
        'name':  'PIDM-ME',
        'file':  'pidm_darcy_pidm_me_10244029.out',
        'color': '#ff7f0e',   # orange — matches paper
        'ls':    (0, (3, 1, 1, 1)),
        'lw':    1.8,
    },
    {
        'name':  'PIDM-SE',
        'file':  'pidm_darcy_pidm_se_10244030.out',
        'color': '#e377c2',   # pink/magenta — matches paper
        'ls':    ':',
        'lw':    1.8,
    },
]

SMOOTH_WINDOW = 15       # rolling-mean window (number of log-points, each 500 iters)
SKIP_ITERS   = 3         # drop first N log-points (iters 0/500/1000 — not yet converged)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_log(filepath: str):
    """Return (iterations, test_loss) arrays from a training stdout log."""
    pattern = re.compile(r'test loss at iteration (\d+):\s*([0-9.eE+\-]+)')
    iters, losses = [], []
    with open(filepath) as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(iters, dtype=float), np.array(losses, dtype=float)


def parse_log_new(filepath: str):
    """Return (iterations, test_loss, residual) from '[iter N] test_loss: X residual: X' logs."""
    pattern = re.compile(
        r'\[iter\s+(\d+)\]\s+test_loss:\s*([0-9.eE+\-]+)\s+residual:\s*([0-9.eE+\-]+)'
    )
    iters, losses, residuals = [], [], []
    with open(filepath) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                residuals.append(float(m.group(3)))
    return np.array(iters, dtype=float), np.array(losses, dtype=float), np.array(residuals, dtype=float)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean; edges filled with cumulative mean to avoid phase lag."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(wspace=0.38)

    # ── (a) Residual RMAE — PIDM-ME and PIDM-SE only ────────────────────
    for m in RESIDUAL_MODELS:
        path = os.path.join(LOG_DIR, m['file'])
        iters, _, residuals = parse_log_new(path)

        iters     = iters[SKIP_ITERS:]
        residuals = residuals[SKIP_ITERS:]

        residuals_sm = rolling_mean(residuals, SMOOTH_WINDOW)
        x = iters / 1e3

        axes[0].semilogy(x, residuals_sm,
                         color=m['color'], linestyle=m['ls'], linewidth=m['lw'],
                         label=m['name'])

    # ── (b) Test data loss — all 5 models ───────────────────────────────
    for m in MODELS:
        path = os.path.join(LOG_DIR, m['file'])
        iters, losses = parse_log(path)

        iters  = iters[SKIP_ITERS:]
        losses = losses[SKIP_ITERS:]

        losses_sm = rolling_mean(losses, SMOOTH_WINDOW)
        x = iters / 1e3

        axes[1].semilogy(x, losses_sm,
                         color=m['color'], linestyle=m['ls'], linewidth=m['lw'],
                         label=m['name'])

    # ── Axis formatting ──────────────────────────────────────────────────

    for ax, letter, ylabel in [
        (axes[0], 'a', 'Residual Error RMAE'),
        (axes[1], 'b', 'Test Data Loss'),
    ]:
        ax.set_xlabel('Training Iterations (×10³)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'({letter})', loc='left', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs='auto'))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

    axes[0].legend(fontsize=9, loc='upper right', framealpha=0.85)
    axes[1].legend(fontsize=9, loc='upper right', framealpha=0.85)

    plt.tight_layout()

    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2_darcy.pdf')
    out_png = out_pdf.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'Saved {out_pdf}')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
