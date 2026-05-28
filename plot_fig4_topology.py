"""
Topology optimization visualization matching paper Fig 4 style.

Loads saved sample data from results/reproduced/topology/<model>/test_level_2/,
binarizes the density field at rho=0.5, and plots panels with paper-matching
colormaps, axis labels, and metric annotations.

Comparison layout (5 cols × n_rows):
  PIDM ρ | PIDM Residual | SIMP ρ | Diffusion ρ | Diffusion Residual

Note on residuals: residuals.csv stores one scalar R_MAE per sample row.
The spatial residual panel recomputes R = KU − F on-the-fly using the same
StiffnessMatrix assembly as the training code (src/residuals_mechanics_K.py).
The scalar R_MAE from residuals.csv is shown as a subtitle on each residual panel.

Usage:
    conda run -n pidm python plot_fig4_topology.py
    conda run -n pidm python plot_fig4_topology.py --compare --output fig4.pdf
    conda run -n pidm python plot_fig4_topology.py --results-dir results/reproduced/topology/PIDM/test_level_2 --n-samples 4 --output fig4_PIDM.pdf
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# ── channel layout from sample.py ──────────────────────────────────────────
# output channels (sample_N.csv):  0=u_x, 1=u_y, 2=rho
# cond channels (cond_channel_N.csv):
#   0=vf_arr, 1=strain_energy_fem, 2=von_mises_fem
#   3=disp_x_fem, 4=disp_y_fem, 5=E_field (ref design)
#   6=BC_node_x, 7=BC_node_y, 8=load_x, 9=load_y

BINARIZE_THRESHOLD = 0.5

# Path to SolidsPy mesh files (no BCs — BCs are applied at runtime from cond channels)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_NO_BC_FOLDER = os.path.join(_PROJECT_ROOT, 'data', 'mechanics', 'solidspy_k_no_BC') + os.sep

# Lazily initialised; shared across all samples in one run
_STIFFNESS = None


def _get_stiffness():
    """Return a cached StiffnessMatrix built from the SolidsPy mesh files."""
    global _STIFFNESS
    if _STIFFNESS is None:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        import torch
        from src.residuals_mechanics_K import StiffnessMatrix
        _STIFFNESS = StiffnessMatrix(no_BC_folder=_NO_BC_FOLDER, device='cpu')
    return _STIFFNESS

# Paper Fig 4 colormaps
# Density: black (void, ρ=0) → cream/wheat (solid, ρ=1)
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    'density_warm', ['#000000', '#C8A850', '#F5DEB3'], N=256
)
# Residual: dark purple (low) → yellow (high), log scale
RESIDUAL_CMAP = 'viridis'

ROW_LABELS = list('abcdefghijklmnop')


def load_sample(sample_dir: str):
    def csv(name):
        return np.loadtxt(os.path.join(sample_dir, name), delimiter=',')
    vf_raw = csv('cond_channel_0.csv')
    return {
        'u_x':     csv('sample_0.csv'),
        'u_y':     csv('sample_1.csv'),
        'rho':     csv('sample_2.csv'),
        'vf':      float(np.atleast_1d(vf_raw).flat[0]),
        'sed_fem': csv('cond_channel_1.csv'),
        'vm_fem':  csv('cond_channel_2.csv'),
        'ux_fem':  csv('cond_channel_3.csv'),
        'uy_fem':  csv('cond_channel_4.csv'),
        'E_field': csv('cond_channel_5.csv'),
        'bc_x':    csv('cond_channel_6.csv'),
        'bc_y':    csv('cond_channel_7.csv'),
        'load_x':  csv('cond_channel_8.csv'),
        'load_y':  csv('cond_channel_9.csv'),
    }


def load_metrics(results_dir: str, n_samples: int):
    def col(fname):
        path = os.path.join(results_dir, fname)
        return np.loadtxt(path, delimiter=',')[:n_samples]
    return {
        'residual': col('residuals.csv'),
        'ce':       col('rel_CE_error.csv'),
        'vf':       col('rel_vf_error.csv'),
        'fm':       col('fm_error.csv'),
    }


def binarize(rho, threshold=BINARIZE_THRESHOLD):
    return (rho > threshold).astype(float)


def mechanics_residual_map(s):
    """
    True mechanical residual R = KU - F per element, returned as a 64×64 map.

    Follows the same assembly as src/residuals_mechanics_K.py:
      - K is assembled element-by-element scaled by density rho (SIMP)
      - BCs enforced via row-identity replacement (same as compute_residual)
      - R = K·u − F in global DOF space [65×65×2 = 8450 DOFs]
      - Node-wise residual magnitude averaged over 4 element corners → [64, 64]
    """
    import torch
    import einops as ein
    from einops import rearrange

    stiffs = _get_stiffness()

    # rho is saved as 65×65 with a zero-padded last row/col; strip it to get 64×64 elements
    rho_64   = np.array(s['rho'][:-1, :-1], dtype=np.float32)
    rho_t    = torch.tensor(rho_64).unsqueeze(0)          # [1, 64, 64]
    rho_flat = rho_t.reshape(1, -1)                       # [1, 4096]

    # Displacements: saved as 65×65 bilinear-upscaled FEM node values
    u_x = torch.tensor(np.array(s['u_x'], dtype=np.float32)).unsqueeze(0)   # [1, 65, 65]
    u_y = torch.tensor(np.array(s['u_y'], dtype=np.float32)).unsqueeze(0)   # [1, 65, 65]

    # BC and load fields: 65×65 node-level arrays from cond_channel_{6-9}.csv
    bc_x   = torch.tensor(np.array(s['bc_x'],   dtype=np.float32)).unsqueeze(0)
    bc_y   = torch.tensor(np.array(s['bc_y'],   dtype=np.float32)).unsqueeze(0)
    load_x = torch.tensor(np.array(s['load_x'], dtype=np.float32)).unsqueeze(0)
    load_y = torch.tensor(np.array(s['load_y'], dtype=np.float32)).unsqueeze(0)

    # Global DOF vectors [1, 8450]
    u_stiff = (stiffs.image_to_stiffness_coord(u_x, 0) +
               stiffs.image_to_stiffness_coord(u_y, 1))
    f_stiff = (stiffs.image_to_stiffness_coord(load_x, 0) +
               stiffs.image_to_stiffness_coord(load_y, 1))

    # Assemble global stiffness K [1, 8450, 8450] — same index_put_ pattern as compute_residual
    batch_size = 1
    glob_idcs  = stiffs.glob_assembler_idcs.unsqueeze(0).expand(batch_size, -1, -1, -1)
    global_b_idcs = torch.arange(batch_size).repeat_interleave(stiffs.nels * stiffs.ndof ** 2)

    k_glob = torch.zeros((batch_size, stiffs.neq, stiffs.neq))
    scaled_kloc     = stiffs.tot_local_stiffness.unsqueeze(0) * rho_flat[:, :, None, None]
    scaled_kloc_val = scaled_kloc[:, :, stiffs.indices_ext[:, 0], stiffs.indices_ext[:, 1]]
    k_glob = k_glob.index_put_(
        (global_b_idcs,
         glob_idcs[:, :, :, 0].flatten(),
         glob_idcs[:, :, :, 1].flatten()),
        scaled_kloc_val.flatten(), accumulate=True)

    # Apply BCs: zero out BC rows in K, replace diagonal with 1, zero BC entries in F
    bc_stiff = (stiffs.image_to_stiffness_coord(bc_x, 0) +
                stiffs.image_to_stiffness_coord(bc_y, 1))
    bc_mask  = bc_stiff != 0                              # [1, 8450]
    mask_ext = bc_mask.unsqueeze(-1).expand_as(k_glob)
    k_glob[mask_ext] = 0
    k_glob += torch.eye(stiffs.neq).expand(batch_size, -1, -1) * mask_ext
    f_stiff[bc_mask] = 0

    # R = K·u − F  →  [1, 8450]
    residual = ein.einsum(k_glob, u_stiff, 'b i j, b j -> b i') - f_stiff

    # Reshape [8450] → [65, 65, 2], compute per-node magnitude [65, 65]
    r = rearrange(residual[0].detach(), '(x y d) -> x y d', x=65, y=65, d=2).numpy()
    r_mag = np.sqrt(r[:, :, 0] ** 2 + r[:, :, 1] ** 2)

    # Average 4 corner nodes per element → [64, 64]
    r_elem = (r_mag[:-1, :-1] + r_mag[:-1, 1:] +
              r_mag[1:, :-1]  + r_mag[1:, 1:]) / 4.0
    return r_elem


def simp_compliance(s):
    """Compliance from FEM strain energy: C = 2 * sum(SE_element)."""
    return 2.0 * s['sed_fem'].sum()


def _style_ax(ax):
    """Hide tick marks/numbers; show ξ₁ and ξ₂ axis labels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('ξ₁', fontsize=7, labelpad=1)
    ax.set_ylabel('ξ₂', fontsize=7, labelpad=1, rotation=0, va='center')
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_visible(True)


def _design_ax(ax, field, title=''):
    """Plot a density field with the warm black→cream colormap."""
    ax.imshow(field, cmap=DENSITY_CMAP, vmin=0, vmax=1,
              origin='upper', aspect='equal')
    if title:
        ax.set_title(title, fontsize=7, pad=3)
    _style_ax(ax)


def _residual_ax(ax, residual_map, norm=None):
    """Plot displacement residual with viridis log-scale colormap."""
    eps = 1e-9
    r_pos = residual_map[residual_map > 0]
    vmin = r_pos.min() if r_pos.size > 0 else eps
    vmax = np.percentile(residual_map, 98) if residual_map.max() > 0 else eps * 10
    if norm is None:
        norm = LogNorm(vmin=max(vmin, eps), vmax=max(vmax, vmin * 10))
    im = ax.imshow(np.clip(residual_map, norm.vmin, None),
                   cmap=RESIDUAL_CMAP, norm=norm,
                   origin='upper', aspect='equal')
    _style_ax(ax)
    return im, norm


def plot_loadcase(ax, s):
    """Reference design silhouette with BC hatching and load arrows."""
    ref = (s['E_field'] > 0.5).astype(float)
    ax.imshow(ref, cmap='Greys', vmin=0, vmax=1, origin='upper', aspect='equal')

    bc_mask = (s['bc_x'] > 0) | (s['bc_y'] > 0)
    rows, cols = np.where(bc_mask)
    if len(cols):
        ax.fill_betweenx(np.arange(ref.shape[0]),
                         cols.min() - 0.5, cols.min() + 0.5,
                         color='steelblue', alpha=0.35, linewidth=0)
        ax.axvline(cols.min() + 0.5, color='steelblue', linewidth=1.5)

    load_pts = np.argwhere((np.abs(s['load_x']) + np.abs(s['load_y'])) > 0)
    scale = ref.shape[0] * 0.15
    for (r, c) in load_pts:
        dx = s['load_x'][r, c]
        dy = s['load_y'][r, c]
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            ax.annotate('', xy=(c + dx / mag * scale, r - dy / mag * scale),
                        xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color='crimson',
                                        lw=2.5, mutation_scale=18))
    _style_ax(ax)


def make_figure(results_dir, n_samples=4, title='PIDM'):
    """Single-model figure: Load case | SIMP ρ | Generated ρ | Residual."""
    sample_dirs = [os.path.join(results_dir, f'sample_{i}') for i in range(n_samples)]
    samples = [load_sample(d) for d in sample_dirs]
    metrics = load_metrics(results_dir, n_samples)

    res_maps = [mechanics_residual_map(s) for s in samples]
    all_vals = np.concatenate([r.ravel() for r in res_maps])
    eps = 1e-9
    shared_norm = LogNorm(
        vmin=max(all_vals[all_vals > 0].min() if (all_vals > 0).any() else eps, eps),
        vmax=max(np.percentile(all_vals, 98), eps * 10),
    )

    ncols, nrows = 4, n_samples
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.3, nrows * 2.1),
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.45})
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Load case', 'SIMP design ρ', f'Generated ρ', 'Residual']
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8.5, fontweight='bold', pad=5)

    for i, (s, res_map) in enumerate(zip(samples, res_maps)):
        rho_bin = binarize(s['rho'])
        ref_bin = (s['E_field'] > 0.5).astype(float)
        rho_bar = rho_bin.mean()
        ce_pct  = metrics['ce'][i] * 100
        C       = simp_compliance(s)
        V_max   = s['vf']

        # row label
        ax0 = axes[i, 0]
        ax0.annotate(f'({ROW_LABELS[i]})', xy=(-0.18, 0.5),
                     xycoords='axes fraction', fontsize=10,
                     fontweight='bold', va='center', ha='right')

        plot_loadcase(ax0, s)

        _design_ax(axes[i, 1], ref_bin,
                   title=f'C = {C:.2f}, V$_{{max}}$ = {V_max:.2f}')

        _design_ax(axes[i, 2], rho_bin,
                   title=f'CE = {ce_pct:+.2f}%, ρ̄ = {rho_bar:.2f}')

        im, _ = _residual_ax(axes[i, 3], res_map, norm=shared_norm)

    # residual colorbar
    cb_ax = fig.add_axes([0.92, 0.12, 0.015, 0.25])
    cb = fig.colorbar(ScalarMappable(cmap=RESIDUAL_CMAP, norm=shared_norm), cax=cb_ax)
    cb.set_label('|KU − F| per element (log)', fontsize=6, labelpad=4)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    return fig


def make_comparison_figure(pidm_dir, diff_dir, n_samples=4):
    """
    Fig 4-style comparison: PIDM vs Diffusion, 5 columns per row.

    Layout: PIDM ρ | PIDM Residual | SIMP ρ | Diffusion ρ | Diffusion Residual
    """
    samples_p = [load_sample(os.path.join(pidm_dir,  f'sample_{i}')) for i in range(n_samples)]
    samples_d = [load_sample(os.path.join(diff_dir,  f'sample_{i}')) for i in range(n_samples)]
    metrics_p = load_metrics(pidm_dir,  n_samples)
    metrics_d = load_metrics(diff_dir, n_samples)

    # shared log norm for both residual columns
    all_res_maps = [mechanics_residual_map(s) for s in samples_p + samples_d]
    all_vals = np.concatenate([r.ravel() for r in all_res_maps])
    eps = 1e-9
    pos = all_vals[all_vals > 0]
    shared_norm = LogNorm(
        vmin=max(pos.min() if pos.size > 0 else eps, eps),
        vmax=max(np.percentile(all_vals, 98), eps * 10),
    )
    res_maps_p = all_res_maps[:n_samples]
    res_maps_d = all_res_maps[n_samples:]

    ncols, nrows = 5, n_samples
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.2, nrows * 2.1),
        gridspec_kw={'wspace': 0.08, 'hspace': 0.50},
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        'Design ρ',
        'Residual R_MAE(ρ, u1, u2)',
        'SIMP Design ρ',
        'Design ρ',
        'Residual R_MAE(ρ, u1, u2)',
    ]

    for i in range(n_samples):
        sp, sd = samples_p[i], samples_d[i]
        res_p, res_d = res_maps_p[i], res_maps_d[i]

        rho_p    = binarize(sp['rho'])
        rho_d    = binarize(sd['rho'])
        ref_bin  = (sp['E_field'] > 0.5).astype(float)  # same ref for both

        rho_bar_p = rho_p.mean()
        rho_bar_d = rho_d.mean()
        ce_p      = metrics_p['ce'][i] * 100
        ce_d      = metrics_d['ce'][i] * 100
        C         = simp_compliance(sp)
        V_max     = sp['vf']

        # (a), (b), (c), (d) row label on the left of col 0
        axes[i, 0].annotate(
            f'({ROW_LABELS[i]})', xy=(-0.20, 0.5),
            xycoords='axes fraction', fontsize=10,
            fontweight='bold', va='center', ha='right',
        )

        # col 0 — PIDM generated design
        _design_ax(axes[i, 0], rho_p,
                   title=f'CE = {ce_p:+.2f}%, ρ̄ = {rho_bar_p:.2f}')

        # col 1 — PIDM spatial residual |KU-F| per element; scalar R_MAE as subtitle
        _residual_ax(axes[i, 1], res_p, norm=shared_norm)
        axes[i, 1].set_title(f'R_MAE = {metrics_p["residual"][i]:.4f}', fontsize=6, pad=2)

        # col 2 — SIMP reference design
        _design_ax(axes[i, 2], ref_bin,
                   title=f'C = {C:.2f}, V$_{{max}}$ = {V_max:.2f}')

        # col 3 — Diffusion generated design
        _design_ax(axes[i, 3], rho_d,
                   title=f'CE = {ce_d:+.2f}%, ρ̄ = {rho_bar_d:.2f}')

        # col 4 — Diffusion spatial residual |KU-F| per element; scalar R_MAE as subtitle
        _residual_ax(axes[i, 4], res_d, norm=shared_norm)
        axes[i, 4].set_title(f'R_MAE = {metrics_d["residual"][i]:.4f}', fontsize=6, pad=2)

    # Bold column headers set after the loop so they are not overwritten by per-sample subtitles
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8, fontweight='bold', pad=5)

    # density colorbar (right of SIMP / design columns)
    cb_den_ax = fig.add_axes([0.93, 0.55, 0.012, 0.30])
    sm_den = ScalarMappable(cmap=DENSITY_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm_den.set_array([])
    cb_den = fig.colorbar(sm_den, cax=cb_den_ax)
    cb_den.set_label('ρ', fontsize=7, labelpad=4)
    cb_den.ax.tick_params(labelsize=6)

    # residual colorbar (right of residual columns)
    cb_res_ax = fig.add_axes([0.93, 0.12, 0.012, 0.30])
    sm_res = ScalarMappable(cmap=RESIDUAL_CMAP, norm=shared_norm)
    sm_res.set_array([])
    cb_res = fig.colorbar(sm_res, cax=cb_res_ax)
    cb_res.set_label('|KU − F| per element (log)', fontsize=7, labelpad=4)
    cb_res.ax.tick_params(labelsize=6)

    fig.suptitle('Topology optimization — PIDM vs. Diffusion (test level 2)',
                 fontsize=10, fontweight='bold', y=1.01)
    return fig


def parse_args():
    p = argparse.ArgumentParser(description='Plot Fig 4 style topology optimization results')
    p.add_argument('--results-dir', default='results/reproduced/topology/PIDM/test_level_2',
                   help='Path to model test_level_2 results directory')
    p.add_argument('--n-samples', type=int, default=4,
                   help='Number of samples to plot (default 4 for (a)-(d) rows)')
    p.add_argument('--output', default='fig4_topology.pdf',
                   help='Output file (pdf or png)')
    p.add_argument('--title', default='PIDM — Topology Optimization (test level 2)')
    p.add_argument('--compare', action='store_true',
                   help='Side-by-side comparison of PIDM vs standard_diffusion')
    p.add_argument('--pidm-dir',   default='results/reproduced/topology/PIDM/test_level_2')
    p.add_argument('--diff-dir',   default='results/reproduced/topology/standard_diffusion/test_level_2')
    return p.parse_args()


def main():
    args = parse_args()
    n = min(args.n_samples, 5)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'figure.dpi': 150,
    })

    if args.compare:
        fig = make_comparison_figure(args.pidm_dir, args.diff_dir, n_samples=n)
        out = args.output if args.output != 'fig4_topology.pdf' else 'fig4_topology_comparison.pdf'
    else:
        fig = make_figure(args.results_dir, n_samples=n, title=args.title)
        out = args.output

    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')

    if out.endswith('.pdf'):
        png_out = out.replace('.pdf', '.png')
        fig.savefig(png_out, dpi=200, bbox_inches='tight')
        print(f'Saved: {png_out}')

    plt.close(fig)


if __name__ == '__main__':
    main()
