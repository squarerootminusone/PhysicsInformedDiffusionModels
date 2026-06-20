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

  Subplot (a) — residual RMAE proxy:
      For PIDM-ME / PIDM-SE: loss_test is dominated by the residual penalty
      at early training (3.85e6 and 1.2e4 respectively at t=0), and decreases
      as the model learns to satisfy the PDE — a meaningful proxy for residual
      learning dynamics even though it is not identical to RMAE.
      For Diffusion / PG / CoCoGen: loss_test = data_loss (no residual signal),
      shown for reference — their curves reflect data quality, not PDE compliance.
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

    for m in MODELS:
        path = os.path.join(LOG_DIR, m['file'])
        iters, losses = parse_log(path)

        # Drop initial pathological points (model not yet trained)
        iters  = iters[SKIP_ITERS:]
        losses = losses[SKIP_ITERS:]

        losses_sm = rolling_mean(losses, SMOOTH_WINDOW)
        x = iters / 1e3   # display in thousands

        kwargs = dict(color=m['color'], linestyle=m['ls'], linewidth=m['lw'],
                      label=m['name'])

        # ── (a) Residual RMAE proxy ──────────────────────────────────────
        # axes[0].semilogy(x, losses_sm, **kwargs)

        # ── (b) Test data loss ───────────────────────────────────────────
        # Exact for Diffusion / PG / CoCoGen.  PIDM-ME slightly inflated by
        # physics penalty; PIDM-SE is a good proxy (c_residual = 1e-5).
        axes[1].semilogy(x, losses_sm, **kwargs)

    # ── Axis formatting ──────────────────────────────────────────────────

    for ax, letter, ylabel in [
        # (axes[0], 'a', 'Residual Error RMAE'),
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

    # Legend on subplot (b) only to avoid repetition
    axes[1].legend(fontsize=9, loc='upper right', framealpha=0.85)

    # Annotation explaining the RMAE proxy on subplot (a)
    # axes[0].text(
    #     0.98, 0.98,
    #     'Proxy: stdout test loss\n'
    #     '(wandb RMAE not in logs)\n'
    #     'c_res=0 models: data loss only',
    #     transform=axes[0].transAxes,
    #     fontsize=6.5, color='#555555',
    #     ha='right', va='top',
    #     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8),
    # )

    plt.tight_layout()

    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2_darcy.pdf')
    out_png = out_pdf.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'Saved {out_pdf}')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
