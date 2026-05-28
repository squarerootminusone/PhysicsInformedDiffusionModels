"""
Reproduce Fig 8 style from Bastek et al. (ICLR 2025): Physics-Informed Diffusion Models.

Layout: 4 rows (a-d) × 3 cols (Permeability K | Pressure p | PDE Residual |R|)
  (a) PIDM-ME sample_0
  (b) PIDM-ME sample_1
  (c) PIDM-SE sample_0
  (d) PIDM-SE sample_1

File convention (from sample.py / residuals_darcy.py):
  sample_0.csv = channel 0 = p (pressure)
  sample_1.csv = channel 1 = K (permeability)

Residual: R = -div(K * grad(p)) - f_s  where f_s is the stationary source field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatterSciNotation
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results", "reproduced", "darcy")

SAMPLES = [
    ("PIDM-ME", "sample_0", "(a)"),
    ("PIDM-ME", "sample_1", "(b)"),
    ("PIDM-SE", "sample_0", "(c)"),
    ("PIDM-SE", "sample_1", "(d)"),
]
STEP = "step_300000"

# ---------------------------------------------------------------------------
# Darcy PDE parameters (must match main.py / residuals_darcy.py)
# ---------------------------------------------------------------------------
PIXELS_PER_DIM = 64
PIXELS_AT_BOUNDARY = True   # True → grid from 0 to 1 inclusive, h = 1/(N-1)
DOMAIN_LENGTH = 1.0
W_SOURCE = 0.125
R_SOURCE = 10.0


def make_source_field(n=PIXELS_PER_DIM, domain=DOMAIN_LENGTH, w=W_SOURCE, r=R_SOURCE):
    """Stationary source f_s on a cell-centered grid (always 1/N spacing)."""
    pixel_size = domain / n
    coords = np.linspace(pixel_size / 2, domain - pixel_size / 2, n)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    f_s = np.zeros((n, n), dtype=np.float64)
    mask_lo = np.abs(X - 0.5 * w) <= 0.5 * w
    mask_hi = np.abs(X - 1.0 + 0.5 * w) <= 0.5 * w
    mask_y_lo = np.abs(Y - 0.5 * w) <= 0.5 * w
    mask_y_hi = np.abs(Y - 1.0 + 0.5 * w) <= 0.5 * w
    f_s[mask_lo & mask_y_lo] = r
    f_s[mask_hi & mask_y_hi] = -r
    return f_s


F_S = make_source_field()


def darcy_residual(K, p):
    """
    Per-pixel PDE residual R = -div(K * grad(p)) - f_s.

    Uses second-order central FD (np.gradient) with spacing h = 1/(N-1)
    for pixels_at_boundary=True (boundary-inclusive grid).
    Returns |R| as a 2-D array of shape (N, N).
    """
    h = DOMAIN_LENGTH / (PIXELS_PER_DIM - 1) if PIXELS_AT_BOUNDARY else DOMAIN_LENGTH / PIXELS_PER_DIM

    p_x  = np.gradient(p, h, axis=0)
    p_y  = np.gradient(p, h, axis=1)
    p_xx = np.gradient(p_x, h, axis=0)
    p_yy = np.gradient(p_y, h, axis=1)
    K_x  = np.gradient(K, h, axis=0)
    K_y  = np.gradient(K, h, axis=1)

    R = -(K * p_xx + K_x * p_x) - (K * p_yy + K_y * p_y) - F_S
    return np.abs(R)


def load_sample(variant, sample_dir):
    """Return (K, p) arrays for a given variant / sample directory."""
    base = os.path.join(RESULTS, variant, "validation", STEP, sample_dir)
    # channel 0 → p (pressure), channel 1 → K (permeability)
    p = np.loadtxt(os.path.join(base, "sample_0.csv"), delimiter=",")
    K = np.loadtxt(os.path.join(base, "sample_1.csv"), delimiter=",")
    return K, p


# ---------------------------------------------------------------------------
# Precompute all data and determine shared colour limits per column
# ---------------------------------------------------------------------------
data = []
for variant, sample_dir, row_label in SAMPLES:
    K, p = load_sample(variant, sample_dir)
    R = darcy_residual(K, p)
    data.append({"K": K, "p": p, "R": R, "label": row_label, "variant": variant})

K_all = np.concatenate([d["K"].ravel() for d in data])
p_all = np.concatenate([d["p"].ravel() for d in data])
R_all = np.concatenate([d["R"].ravel() for d in data])

K_vmin, K_vmax = K_all.min(), K_all.max()
p_vmin, p_vmax = p_all.min(), p_all.max()
R_vmin = max(R_all[R_all > 0].min(), 1e-6)
R_vmax = R_all.max()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COL_TITLES = ["Permeability $K$", "Pressure $p$", r"$|R(K,p)|$"]
CMAPS = ["viridis", "inferno", "viridis"]
NROWS, NCOLS = 4, 3
FIG_W, FIG_H = 10, 13

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))

for row_idx, d in enumerate(data):
    fields = [d["K"], d["p"], d["R"]]
    vnorms = [
        mcolors.Normalize(vmin=K_vmin, vmax=K_vmax),
        mcolors.Normalize(vmin=p_vmin, vmax=p_vmax),
        mcolors.LogNorm(vmin=R_vmin, vmax=R_vmax),
    ]

    for col_idx in range(NCOLS):
        ax = axes[row_idx, col_idx]
        field = fields[col_idx]
        cmap = CMAPS[col_idx]
        norm = vnorms[col_idx]

        im = ax.imshow(
            field.T,           # transpose so x→horizontal, y→vertical
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=[0, 1, 0, 1],
        )
        ax.set_xticks([])
        ax.set_yticks([])

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        if col_idx == 2:
            cb.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

        # column titles on first row
        if row_idx == 0:
            ax.set_title(COL_TITLES[col_idx], fontsize=11)

        # row labels on left
        if col_idx == 0:
            ax.set_ylabel(
                f"{d['label']} {d['variant']}",
                fontsize=10,
                rotation=0,
                labelpad=60,
                va="center",
            )

plt.suptitle(
    "Fig 8 — Darcy flow: PIDM-ME and PIDM-SE generated fields and PDE residuals",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()

out_pdf = os.path.join(BASE, "fig8_darcy.pdf")
out_png = os.path.join(BASE, "fig8_darcy.png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close()
