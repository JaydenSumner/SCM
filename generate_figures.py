import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

WORKDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(WORKDIR, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# matplotlib style
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'axes.labelsize'   : 11,
    'axes.titlesize'   : 11,
    'legend.fontsize'  : 9,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
    'xtick.top'        : True,
    'ytick.right'      : True,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 200,
    'savefig.bbox'     : 'tight',
    'lines.linewidth'  : 1.6,
})

COLORS = {
    'step6' : '#555555',
    'step7' : '#2166ac',
    'step8' : '#f4a40d',
    'step9' : '#d6604d',
    'step10': '#1a9641',
    'desi'  : '#8e24aa',
    'lcdm'  : '#000000',
}

STEP_LABELS = {
    'step6' : 'Step 6  $(\\varepsilon,z_c)$',
    'step7' : 'Step 7  $A_{\\rm fp}\\in[0,5]$',
    'step8' : 'Step 8  $A_{\\rm fp}\\in[0,15]$',
    'step9' : 'Step 9  $Z_{c,{\\rm max}}=5.0$',
    'step10': 'Step 10  $Z_{c,{\\rm max}}=3.0$ (rejection)',
}

# R-1 trajectory data
R1 = {}

R1['step6'] = {
    'steps': [1000,  5000, 10000, 15000, 20000, 30000, 40000, 50000],
    'r1'   : [2.10,  0.45,  0.25,  0.20,  0.18,  0.15,  0.14,  0.13],
}

R1['step7'] = {
    'steps': [320,   960,  1920,  4096,  4864,  5120,  6144,  8192,
              12800, 16000, 18176, 22528, 23040, 24064, 24320,
              30464, 31488, 38080, 39680, 49920],
    'r1'   : [19.99, 10.59, 2.29,  0.70,  0.41,  0.31,  0.14,  0.097,
               0.095, 0.095, 0.078, 0.085, 0.070, 0.044, 0.042,
               0.036, 0.041, 0.036, 0.044, 0.090],
}

R1['step8'] = {
    'steps': [256,   1024,  1792,  3328,  4608,  5632,  6144,  8192,
              10240, 11520, 13568, 16000, 17920, 18176, 18944,
              22720, 26624, 30720, 33024, 38144, 41280, 43200, 49920],
    'r1'   : [6.631, 1.934, 0.690, 0.645, 0.488, 0.391, 0.309, 0.159,
               0.101, 0.146, 0.080, 0.102, 0.097, 0.040, 0.063,
               0.040, 0.035, 0.092, 0.048, 0.095, 0.048, 0.045, 0.050],
}

R1['step9'] = {
    'steps': [1000,  5000, 10000, 15000, 18000, 20000, 21000, 22000,
              23000, 24000, 25000, 27000, 30000, 35000, 40000, 45000, 50000],
    'r1'   : [37.0,  20.0,  12.0,   8.0,   5.5,   5.1,   1.97,  1.12,
               0.798, 0.92,  1.11,   1.35,  2.38,  5.37,  7.17,  8.88,  8.5],
}

R1['step10'] = {
    'steps': [256,   6400,  7936,  8448,  8704, 14336, 15360, 16128,
              16640, 17408, 18688, 19200, 20992, 21248, 22272, 23040,
              23296, 26368, 26624, 27648, 28160, 28416, 28672, 29184,
              29440, 29696, 29952, 30208, 30464, 30720, 30976, 31232],
    'r1'   : [19.5,  1.94,  0.790, 0.605, 0.601, 0.734, 0.743, 0.643,
               0.597, 0.404, 0.362, 0.294, 0.235, 0.199, 0.139, 0.111,
               0.095, 0.059, 0.063, 0.088, 0.090, 0.066, 0.048, 0.046,
               0.046, 0.041, 0.036, 0.033, 0.034, 0.033, 0.037, 0.027],
}

# chain loading utilities
def load_chain(filename, burnin_frac=0.30):
    """Load a Cobaya chain file; returns (weights, param_dict)."""
    with open(filename, 'r') as f:
        header = f.readline().strip().lstrip('#').split()
    data = np.loadtxt(filename)
    n = len(data)
    cut = int(n * burnin_frac)
    data = data[cut:]
    weights = data[:, 0]
    result = {}
    for i, name in enumerate(header):
        result[name] = data[:, i]
    return weights, result

def weighted_percentile(x, w, pct):
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx]
    cdf = np.cumsum(ws) / np.sum(ws)
    return np.interp(pct / 100.0, cdf, xs)

def kde_1d(x, w, xmin, xmax, n=500):
    xi = np.linspace(xmin, xmax, n)
    bw = 1.06 * np.std(x) * len(x)**(-0.2)
    kde = gaussian_kde(x, weights=w, bw_method=bw / np.std(x))
    return xi, kde(xi)

def kde_2d(x, y, w, xlim, ylim, n=80):
    pts = np.vstack([x, y])
    kde = gaussian_kde(pts, weights=w, bw_method=0.12)
    xi = np.linspace(*xlim, n)
    yi = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(n, n)
    return xi, yi, Z

# figure: R-1 all steps
def fig_r1_all():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes = axes.ravel()

    entries = [
        ('step6',  axes[0], 'Step 6: $(\\varepsilon, z_c)$ — plateau'),
        ('step7',  axes[1], 'Step 7: $A_{\\rm fp}\\in[0,5]$'),
        ('step8',  axes[2], 'Step 8: $A_{\\rm fp}\\in[0,15]$'),
        ('step9',  axes[3], 'Step 9: $z_{c,{\\rm max}}=5$ — bimodal'),
        ('step10', axes[4], 'Step 10: rejection fix — converged'),
    ]

    for key, ax, title in entries:
        d = R1[key]
        c = COLORS[key]
        ax.semilogy(d['steps'], d['r1'], color=c, lw=1.8, marker='o',
                    ms=3.5, mec='none')
        ax.axhline(0.1,  color='gray',   lw=0.8, ls='--', alpha=0.6)
        ax.axhline(0.01, color='gray',   lw=0.8, ls=':',  alpha=0.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Accepted steps', fontsize=9)
        ax.set_ylabel('$R-1$', fontsize=9)
        ax.tick_params(labelsize=8)
        # Annotate key milestones
        rmin = min(d['r1'])
        ax.text(0.98, 0.95, f'min $R-1={rmin:.3f}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color=c,
                bbox=dict(fc='white', ec='none', alpha=0.7))

    # Add legend in unused 6th panel
    ax6 = axes[5]
    ax6.axis('off')
    ax6.text(0.5, 0.72, 'Reference lines:', ha='center', va='center',
             transform=ax6.transAxes, fontsize=9)
    ax6.plot([0.1, 0.9], [0.60, 0.60], color='gray', lw=0.8, ls='--',
             transform=ax6.transAxes, alpha=0.8)
    ax6.text(0.5, 0.55, '$R-1 = 0.1$  (dashed)', ha='center', transform=ax6.transAxes,
             fontsize=8.5, color='gray')
    ax6.plot([0.1, 0.9], [0.44, 0.44], color='gray', lw=0.8, ls=':',
             transform=ax6.transAxes, alpha=0.8)
    ax6.text(0.5, 0.39, '$R-1 = 0.01$ (dotted)', ha='center', transform=ax6.transAxes,
             fontsize=8.5, color='gray')
    ax6.text(0.5, 0.20, 'Step 10 achieves\n$R-1 = 0.027$\n(first converged result)',
             ha='center', va='center', transform=ax6.transAxes, fontsize=9,
             color=COLORS['step10'],
             bbox=dict(fc='#e8f5e9', ec=COLORS['step10'], alpha=0.9, lw=1.2, boxstyle='round,pad=0.4'))

    fig.suptitle('Gelman--Rubin $R-1$ convergence trajectories: Steps 6--10',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_r1_all_steps.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: R-1 step 10 detail
def fig_r1_step10_detail():
    d = R1['step10']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(d['steps'], d['r1'], color=COLORS['step10'], lw=2, marker='o', ms=4, mec='none')

    milestones = [
        (7936,  0.790, 'Sub-1 crossing'),
        (21248, 0.199, '$R-1 < 0.2$'),
        (23296, 0.095, '$R-1 < 0.1$'),
        (31232, 0.027, r'$\mathbf{R-1 = 0.027}$'),
    ]
    for xs, ys, label in milestones:
        ax.scatter([xs], [ys], s=60, zorder=5, color=COLORS['step10'], edgecolors='white', lw=0.8)
        offset = (1200, 1.4) if ys < 0.1 else (1200, 1.4)
        ax.annotate(label, (xs, ys), xytext=(xs + 600, ys * 1.7),
                    fontsize=8.5, ha='left', color='#333333',
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))

    ax.axhline(0.1,  color='gray', lw=0.9, ls='--', alpha=0.7, label='$R-1=0.1$')
    ax.axhline(0.01, color='gray', lw=0.9, ls=':',  alpha=0.7, label='$R-1=0.01$ (target)')
    ax.set_xlabel('Accepted steps')
    ax.set_ylabel('$R-1$ (log scale)')
    ax.set_title('Step 10: $R-1$ convergence trajectory\n($\\varepsilon, A_{\\rm fp}$) parameterisation, $z_{c,{\\rm max}}=3.0$ with prior rejection')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(0, 33000)

    # Shade burn-in region
    ax.axvspan(0, 9450, alpha=0.07, color='gray', label='burn-in')
    ax.text(4700, 12, 'burn-in', ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_r1_step10_detail.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: eps progression
def fig_eps_progression():
    steps  = ['Step 6', 'Step 7', 'Step 8', 'Step 9\n(not conv.)', 'Step 10']
    ul_95  = [0.0034,    0.0028,   0.0029,   0.0029,               0.0032]
    colors = [COLORS[f'step{i}'] for i in [6,7,8,9,10]]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(steps, ul_95, color=colors, edgecolor='white', linewidth=0.6, alpha=0.85)

    # Annotate bars
    for bar, val in zip(bars, ul_95):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.00005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(0.003, color='black', lw=1.0, ls='--', alpha=0.5)
    ax.text(4.45, 0.003 + 0.00004, '$0.003$', ha='right', fontsize=8.5, color='#444444')

    ax.set_ylabel('$\\varepsilon$ 95th percentile upper limit')
    ax.set_title('$\\varepsilon$ constraint robustness across the Steps 6--10 campaign')
    ax.set_ylim(0, 0.0042)
    ax.tick_params(axis='x', length=0)

    # Mark Step 9 as not converged
    ax.text(3, ul_95[3] + 0.00025, '(not\nconverged)', ha='center', fontsize=7.5,
            color=COLORS['step9'], style='italic')

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_eps_progression.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 8 zc ceiling
def fig_step8_zc_ceiling():
    fname = os.path.join(WORKDIR, 'scm_step8_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    zc = d.get('zc_scm', d.get('zc', None))
    if zc is None:
        print('  SKIP: zc not found in Step 8 chain')
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    xi, yi = kde_1d(zc, w, 0.3, 3.15)
    ax.fill_between(xi, yi, alpha=0.35, color=COLORS['step8'])
    ax.plot(xi, yi, color=COLORS['step8'], lw=2)

    # Mark ceiling and pileup
    ax.axvline(3.0, color='red', lw=1.5, ls='--', label='$Z_{c,{\\rm max}} = 3.0$ (code ceiling)')
    pileup_frac = np.sum(w[zc > 2.99]) / np.sum(w)
    ax.fill_betweenx([0, max(yi)*1.05], 2.99, 3.15, alpha=0.18, color='red',
                     label=f'Pileup region: {pileup_frac*100:.1f}\\% of samples')

    ax.set_xlabel('$z_c$ (derived)')
    ax.set_ylabel('Marginalised posterior density')
    ax.set_title('Step 8: $z_c$ posterior — 34.5\\% of samples pile up at the brentq ceiling')
    ax.set_xlim(0.3, 3.15)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step8_zc_ceiling.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 9 bimodal A_fp vs zc
def fig_step9_bimodal():
    fname = os.path.join(WORKDIR, 'scm_step9_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    afp = d.get('A_fp', None)
    zc  = d.get('zc_scm', d.get('zc', None))
    if afp is None or zc is None:
        print('  SKIP: required columns not found in Step 9 chain')
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: zc marginal showing bimodality
    ax = axes[0]
    xi, yi = kde_1d(zc, w, 0.3, 5.0)
    ax.fill_between(xi, yi, alpha=0.3, color=COLORS['step9'])
    ax.plot(xi, yi, color=COLORS['step9'], lw=2)
    ax.axvline(3.0, color='black', lw=1.2, ls='--', alpha=0.6, label='$z_c = 3$')
    modeA_frac = np.sum(w[zc < 3.0]) / np.sum(w)
    modeB_frac = np.sum(w[zc >= 3.0]) / np.sum(w)
    ax.text(1.8, max(yi)*0.75, f'Mode A\n{modeA_frac*100:.0f}\\%\n$z_c < 3$',
            ha='center', fontsize=9, color='#1565c0',
            bbox=dict(fc='#e3f2fd', ec='#1565c0', alpha=0.8, boxstyle='round,pad=0.3'))
    ax.text(3.8, max(yi)*0.40, f'Mode B\n{modeB_frac*100:.0f}\\%\n$z_c > 3$',
            ha='center', fontsize=9, color='#c62828',
            bbox=dict(fc='#ffebee', ec='#c62828', alpha=0.8, boxstyle='round,pad=0.3'))
    ax.set_xlabel('$z_c$ (derived)')
    ax.set_ylabel('Posterior density')
    ax.set_title('Step 9: Bimodal $z_c$ posterior')
    ax.set_xlim(0.3, 5.0)
    ax.legend(fontsize=8.5)

    # Right: 2D A_fp vs zc scatter (thinned)
    ax2 = axes[1]
    # Thin to manageable number
    idx = np.random.choice(len(zc), min(4000, len(zc)), replace=False,
                           p=w / w.sum())
    maskA = zc[idx] < 3.0
    maskB = zc[idx] >= 3.0
    ax2.scatter(zc[idx][maskA], afp[idx][maskA], s=2, alpha=0.3,
                color='#1565c0', label=f'Mode A ({modeA_frac*100:.0f}\\%)')
    ax2.scatter(zc[idx][maskB], afp[idx][maskB], s=2, alpha=0.3,
                color='#c62828', label=f'Mode B ({modeB_frac*100:.0f}\\%)')
    ax2.axvline(3.0, color='black', lw=1.2, ls='--', alpha=0.6)
    ax2.set_xlabel('$z_c$ (derived)')
    ax2.set_ylabel('$A_{\\rm fp}$')
    ax2.set_title('Step 9: $A_{\\rm fp}$ vs $z_c$ — two distinct modes')
    ax2.legend(fontsize=8.5, markerscale=4)
    ax2.set_xlim(0.3, 5.0)
    ax2.set_ylim(0, 45)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step9_bimodal.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 10 eps and zc marginals
def fig_step10_marginals():
    fname = os.path.join(WORKDIR, 'scm_step10_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    eps = d['eps']
    zc  = d['zc_scm']
    w0  = d['w0_scm']
    wa  = d['wa_scm']

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))

    panels = [
        (eps, 0.0, 0.008, '$\\varepsilon$', '95th pct UL = 0.0032', 0.00318),
        (zc,  1.0, 3.1,   '$z_c$',          '$z_c = 2.43 \\pm 0.38$', None),
        (w0, -2.5, 0.0,   '$w_0$',          '$w_0 = -0.570 \\pm 0.357$', None),
        (wa, -4.5, 3.0,   '$w_a$',          '$w_a = -1.883 \\pm 1.218$', None),
    ]

    for ax, (x, xlo, xhi, xlabel, note, ul) in zip(axes, panels):
        xi, yi = kde_1d(x, w, xlo, xhi)
        ax.fill_between(xi, yi, alpha=0.35, color=COLORS['step10'])
        ax.plot(xi, yi, color=COLORS['step10'], lw=2)

        # 68% and 95% intervals
        p16 = weighted_percentile(x, w, 16)
        p84 = weighted_percentile(x, w, 84)
        p025 = weighted_percentile(x, w, 2.5)
        p975 = weighted_percentile(x, w, 97.5)
        xi2, yi2 = kde_1d(x, w, xlo, xhi)
        mask68  = (xi2 >= p16) & (xi2 <= p84)
        mask95  = (xi2 >= p025) & (xi2 <= p975)
        ax.fill_between(xi2, yi2, where=mask95, alpha=0.25, color=COLORS['step10'])
        ax.fill_between(xi2, yi2, where=mask68, alpha=0.45, color=COLORS['step10'])

        if ul is not None:
            ax.axvline(ul, color='red', lw=1.2, ls='--', label=f'95th pct = {ul:.4f}')
            ax.legend(fontsize=7.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Posterior density' if ax is axes[0] else '')
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.text(0.97, 0.96, note, transform=ax.transAxes,
                ha='right', va='top', fontsize=7.8,
                bbox=dict(fc='white', ec='none', alpha=0.7))

    fig.suptitle('Step 10: Marginalised posteriors for key parameters (33\\,185 post-burn samples)',
                 fontsize=10.5, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step10_marginals.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 10 eps vs zc 2D
def fig_step10_eps_zc():
    fname = os.path.join(WORKDIR, 'scm_step10_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    eps = d['eps']
    zc  = d['zc_scm']

    fig, ax = plt.subplots(figsize=(6, 5))

    # 2D KDE
    xi, yi, Z = kde_2d(zc, eps, w, xlim=(1.0, 3.05), ylim=(0.0, 0.005))
    levels_frac = [0.393, 0.865]  # 1-sigma and 2-sigma contour levels for 2D
    Zmax = Z.max()
    levels = [Zmax * (1 - f) for f in levels_frac][::-1]

    ax.contourf(xi, yi, Z, levels=[0, levels[0]], colors=[COLORS['step10']], alpha=0.12)
    ax.contourf(xi, yi, Z, levels=[levels[0], levels[1]], colors=[COLORS['step10']], alpha=0.28)
    ax.contour(xi, yi, Z, levels=levels, colors=[COLORS['step10']], linewidths=[1.0, 1.6])

    # Best-fit scatter
    idx_thin = np.random.choice(len(zc), min(1500, len(zc)), replace=False, p=w/w.sum())
    ax.scatter(zc[idx_thin], eps[idx_thin], s=1, alpha=0.15, color=COLORS['step10'])

    ax.axvline(3.0, color='red', lw=1.2, ls='--', alpha=0.6, label='$Z_{c,{\\rm max}}=3.0$')
    ax.set_xlabel('$z_c$')
    ax.set_ylabel('$\\varepsilon$')
    ax.set_title('Step 10: $\\varepsilon$ vs $z_c$ posterior\n(1$\\sigma$ and 2$\\sigma$ contours)')
    ax.legend(fontsize=9)
    ax.set_xlim(1.0, 3.1)
    ax.set_ylim(0, 0.005)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step10_eps_zc.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: w(z) evolution
def fig_wz_evolution():
    fname = os.path.join(WORKDIR, 'scm_step10_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    w0 = d['w0_scm']
    wa = d['wa_scm']

    z = np.linspace(0, 3, 300)
    # CPL: w(z) = w0 + wa * z/(1+z)
    wz_all = w0[:, None] + wa[:, None] * z[None, :] / (1 + z[None, :])

    # Weighted percentiles at each z
    wz_16 = np.array([weighted_percentile(wz_all[:, i], w, 16) for i in range(len(z))])
    wz_84 = np.array([weighted_percentile(wz_all[:, i], w, 84) for i in range(len(z))])
    wz_50 = np.array([weighted_percentile(wz_all[:, i], w, 50) for i in range(len(z))])

    # Mean
    w_norm = w / w.sum()
    wz_mean = np.average(wz_all, axis=0, weights=w)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(z, wz_16, wz_84, alpha=0.25, color=COLORS['step10'], label='68\\% CI')
    ax.plot(z, wz_50, color=COLORS['step10'], lw=2, label='Median')
    ax.plot(z, wz_mean, color=COLORS['step10'], lw=1.5, ls='--', alpha=0.7, label='Mean')

    ax.axhline(-1, color='black', lw=0.9, ls=':', alpha=0.6, label='$w=-1$ (phantom boundary)')
    ax.axhline(-0.827, color=COLORS['desi'], lw=1.2, ls='--', alpha=0.7, label='DESI $w_0=-0.827$')

    # Mark zc region
    ax.axvspan(1.8, 3.0, alpha=0.07, color='orange')
    ax.text(2.4, -3.5, '$z_c$ range\n$[1.8, 3.0]$', ha='center', fontsize=8.5,
            color='darkorange', style='italic')

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$w(z) = w_0 + w_a \\frac{z}{1+z}$')
    ax.set_title('Step 10: Dark energy equation of state $w(z)$\n(CPL approximation, 68\\% posterior band)')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, 3)
    ax.set_ylim(-4.5, 0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_wz_evolution.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: w0-wa plane all steps
def fig_w0wa_comparison():
    fig, ax = plt.subplots(figsize=(8, 6))

    step_data = [
        ('step6',  'scm_mcmc.1.txt',         'w0_scm', 'wa_scm'),
        ('step7',  'scm_repar_mcmc.1.txt',   'w0_scm', 'wa_scm'),
        ('step8',  'scm_step8_mcmc.1.txt',   'w0_scm', 'wa_scm'),
        ('step10', 'scm_step10_mcmc.1.txt',  'w0_scm', 'wa_scm'),
    ]

    for key, fname, col_w0, col_wa in step_data:
        fpath = os.path.join(WORKDIR, fname)
        if not os.path.exists(fpath):
            continue
        w, d = load_chain(fpath)
        w0 = d.get(col_w0)
        wa = d.get(col_wa)
        if w0 is None or wa is None:
            continue
        c = COLORS[key]
        label = STEP_LABELS[key]
        try:
            xi, yi, Z = kde_2d(w0, wa, w, xlim=(-2.5, 0.2), ylim=(-4.5, 3.5))
            Zmax = Z.max()
            levels = sorted([Zmax * 0.607, Zmax * 0.135])
            ax.contour(xi, yi, Z, levels=levels, colors=[c], linewidths=[1.0, 1.8],
                       linestyles=['--', '-'])
        except Exception:
            pass
        ax.plot([], [], color=c, lw=2, label=label)

    # DESI Pantheon+ centre
    ax.scatter(-0.827, -0.750, s=120, marker='*', color=COLORS['desi'], zorder=10,
               label='DESI 2024 Pantheon$+$ centre')
    ax.scatter(-0.727, -1.050, s=80, marker='s', color=COLORS['desi'], zorder=10,
               facecolors='none', linewidths=1.5, label='DESI 2024 DESY5 centre')

    ax.axvline(-1, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_a$')
    ax.set_title('$w_0$--$w_a$ plane: SCM Steps 6, 7, 8, 10 vs DESI 2024\n(contours: 1$\\sigma$ and 2$\\sigma$; Step 9 excluded — not converged)')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(-2.5, 0.2)
    ax.set_ylim(-4.5, 3.5)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_w0wa_comparison.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: BEC mass constraint
def fig_bec_mass():
    fname = os.path.join(WORKDIR, 'scm_step10_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    zc = d['zc_scm']

    # BEC mass: m ~ m0 * ((1+zc)/(1+zc0))^(3/5), m0=22 meV at zc0=2.20
    m0, zc0 = 22.0, 2.20
    m_bec = m0 * ((1 + zc) / (1 + zc0))**0.6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: m_BEC marginal
    ax = axes[0]
    xi, yi = kde_1d(m_bec, w, 14, 35)
    ax.fill_between(xi, yi, alpha=0.35, color=COLORS['step10'])
    ax.plot(xi, yi, color=COLORS['step10'], lw=2)

    p16 = weighted_percentile(m_bec, w, 16)
    p84 = weighted_percentile(m_bec, w, 84)
    p50 = weighted_percentile(m_bec, w, 50)
    mask = (xi >= p16) & (xi <= p84)
    ax.fill_between(xi, yi, where=mask, alpha=0.45, color=COLORS['step10'],
                    label=f'68\\% CI: [{p16:.1f}, {p84:.1f}] meV/$c^2$')

    ax.axvline(p50, color=COLORS['step10'], lw=1.5, ls='--', label=f'Median: {p50:.1f} meV/$c^2$')
    ax.set_xlabel('$m_{\\rm BEC}$ (meV/$c^2$)')
    ax.set_ylabel('Posterior density')
    ax.set_title(f'BEC particle mass: $m_{{\\rm BEC}} = {p50:.1f} \\pm 1.6$\\,meV/$c^2$')
    ax.legend(fontsize=8.5)
    ax.set_xlim(14, 35)
    ax.set_ylim(bottom=0)

    # Right: m_BEC vs zc showing the scaling
    ax2 = axes[1]
    zc_range = np.linspace(0.3, 3.0, 200)
    m_range  = m0 * ((1 + zc_range) / (1 + zc0))**0.6
    ax2.plot(zc_range, m_range, color='#666666', lw=1.5, ls='--',
             label='$m \\propto (1+z_c)^{3/5}$')

    idx_thin = np.random.choice(len(zc), min(2000, len(zc)), replace=False, p=w/w.sum())
    ax2.scatter(zc[idx_thin], m_bec[idx_thin], s=3, alpha=0.2, color=COLORS['step10'])

    ax2.axvline(2.43, color=COLORS['step10'], lw=1.3, ls=':', label='Posterior mean $z_c=2.43$')
    ax2.axhline(22.9, color=COLORS['step10'], lw=1.3, ls=':', alpha=0.6)

    ax2.set_xlabel('$z_c$')
    ax2.set_ylabel('$m_{\\rm BEC}$ (meV/$c^2$)')
    ax2.set_title('BEC mass vs condensation redshift\n(Step 10 posterior samples)')
    ax2.legend(fontsize=8.5)
    ax2.set_xlim(1.0, 3.1)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_bec_mass.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: zc comparison across steps
def fig_zc_comparison():
    step_files = [
        ('step6',  'scm_mcmc.1.txt',         'zc'),
        ('step7',  'scm_repar_mcmc.1.txt',   'zc_scm'),
        ('step8',  'scm_step8_mcmc.1.txt',   'zc_scm'),
        ('step10', 'scm_step10_mcmc.1.txt',  'zc_scm'),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key, fname, col in step_files:
        fpath = os.path.join(WORKDIR, fname)
        if not os.path.exists(fpath):
            continue
        w, d = load_chain(fpath)
        zc = d.get(col)
        if zc is None:
            continue
        xi, yi = kde_1d(zc, w, 0.3, 3.2)
        c = COLORS[key]
        ax.plot(xi, yi, color=c, lw=2, label=STEP_LABELS[key])
        ax.fill_between(xi, yi, alpha=0.08, color=c)

    ax.axvline(3.0, color='red', lw=1.2, ls='--', alpha=0.6, label='$Z_{c,{\\rm max}}=3.0$')
    ax.set_xlabel('$z_c$')
    ax.set_ylabel('Posterior density')
    ax.set_title('$z_c$ posterior: Steps 6, 7, 8, 10\n(Step 9 excluded — not converged, bimodal)')
    ax.legend(fontsize=8.5)
    ax.set_xlim(0.3, 3.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_zc_comparison.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 7 eps and A_fp marginals
def fig_step7_prior_wall():
    fname = os.path.join(WORKDIR, 'scm_repar_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    afp = d.get('A_fp')
    eps = d.get('eps')
    if afp is None:
        print('  SKIP: A_fp not found in Step 7 chain')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # A_fp marginal
    ax = axes[0]
    xi, yi = kde_1d(afp, w, 0, 5.5)
    ax.fill_between(xi, yi, alpha=0.35, color=COLORS['step7'])
    ax.plot(xi, yi, color=COLORS['step7'], lw=2)
    ax.axvline(5.0, color='red', lw=1.5, ls='--', label='Prior wall: $A_{\\rm fp}=5$')
    pctls = np.array([weighted_percentile(afp, w, p) for p in [16, 50, 84, 95]])
    ax.text(0.97, 0.96,
            f'Median = {pctls[1]:.2f}\n95th pct = {pctls[3]:.2f} (wall!)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(fc='white', ec='none', alpha=0.7))
    ax.set_xlabel('$A_{\\rm fp}$')
    ax.set_ylabel('Posterior density')
    ax.set_title('Step 7: $A_{\\rm fp}$ posterior truncated at prior wall ($A_{\\rm fp,max}=5$)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(bottom=0)

    # eps marginal
    ax2 = axes[1]
    xi2, yi2 = kde_1d(eps, w, 0.0, 0.006)
    ax2.fill_between(xi2, yi2, alpha=0.35, color=COLORS['step7'])
    ax2.plot(xi2, yi2, color=COLORS['step7'], lw=2)
    ul95 = weighted_percentile(eps, w, 95)
    ax2.axvline(ul95, color='red', lw=1.2, ls='--', label=f'95th pct = {ul95:.4f}')
    ax2.set_xlabel('$\\varepsilon$')
    ax2.set_ylabel('Posterior density')
    ax2.set_title(f'Step 7: $\\varepsilon$ posterior ($\\varepsilon < {ul95:.4f}$ at 95\\%)')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 0.006)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step7_prior_wall.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


# figure: step 10 no boundary pileup
def fig_step10_no_pileup():
    fname = os.path.join(WORKDIR, 'scm_step10_mcmc.1.txt')
    if not os.path.exists(fname):
        print(f'  SKIP: {fname} not found')
        return
    w, d = load_chain(fname)
    zc  = d['zc_scm']
    afp = d['A_fp']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # zc marginal — no pileup
    ax = axes[0]
    xi, yi = kde_1d(zc, w, 1.0, 3.1)
    ax.fill_between(xi, yi, alpha=0.35, color=COLORS['step10'])
    ax.plot(xi, yi, color=COLORS['step10'], lw=2)
    ax.axvline(3.0, color='red', lw=1.5, ls='--', label='$Z_{c,{\\rm max}}=3.0$ (rejection)')
    pileup = np.sum(w[zc > 2.99]) / np.sum(w)
    ax.text(0.97, 0.96, f'Samples at ceiling: {pileup*100:.1f}\\%\n(cf. Step 8: 34.5\\%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            color=COLORS['step10'],
            bbox=dict(fc='white', ec='none', alpha=0.7))
    ax.set_xlabel('$z_c$')
    ax.set_ylabel('Posterior density')
    ax.set_title('Step 10: $z_c$ — no pileup at $Z_{c,{\\rm max}}$\n(prior rejection correctly applied)')
    ax.legend(fontsize=9)
    ax.set_xlim(1.0, 3.1)
    ax.set_ylim(bottom=0)

    # A_fp marginal
    ax2 = axes[1]
    xi2, yi2 = kde_1d(afp, w, 0, 20)
    ax2.fill_between(xi2, yi2, alpha=0.35, color=COLORS['step10'])
    ax2.plot(xi2, yi2, color=COLORS['step10'], lw=2)
    ax2.axvline(20.0, color='orange', lw=1.2, ls='--', alpha=0.7, label='Prior boundary $A_{\\rm fp,max}=20$')
    pileup2 = np.sum(w[afp > 18]) / np.sum(w)
    ax2.text(0.97, 0.96, f'Samples at $A_{{\\rm fp}}>18$: {pileup2*100:.1f}\\%',
             transform=ax2.transAxes, ha='right', va='top', fontsize=8.5,
             bbox=dict(fc='white', ec='none', alpha=0.7))
    ax2.set_xlabel('$A_{\\rm fp}$')
    ax2.set_ylabel('Posterior density')
    ax2.set_title('Step 10: $A_{\\rm fp}$ — prior not constraining\n(2.5\\% near upper boundary)')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 22)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fig_step10_no_pileup.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    print('Generating figures...')
    np.random.seed(42)

    fig_r1_all()
    fig_r1_step10_detail()
    fig_eps_progression()
    fig_step7_prior_wall()
    fig_step8_zc_ceiling()
    fig_step9_bimodal()
    fig_step10_marginals()
    fig_step10_eps_zc()
    fig_wz_evolution()
    fig_w0wa_comparison()
    fig_bec_mass()
    fig_zc_comparison()
    fig_step10_no_pileup()

    print('\nAll figures saved to:', FIGDIR)
    figs = [f for f in os.listdir(FIGDIR) if f.endswith('.pdf')]
    for f in sorted(figs):
        print(f'  {f}')
