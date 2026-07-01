import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SCM_perturbations import (
    compute_fsigma8, load_rsd_data,
    ZC_MEAN, EPS_MEAN, SIGMA8_SCM, OM_M,
)

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'legend.fontsize': 9,
    'figure.dpi': 120,
})

CHAIN_FILE  = 'scm_step11_mcmc.1.txt'
BURNIN      = 0.30
OUT_DIR     = 'figures'

# column mapping
COL = {
    'weight':    0,
    'logpost':   1,   # -2*logpost
    'H0':        2,
    'ombh2':     3,
    'omch2':     4,
    'tau':       5,
    'ns':        6,
    'logA':      7,
    'A_planck':  8,
    'eps':       9,
    'A_fp':      10,
    'As':        11,
    'sigma8':    12,
    'w0_scm':    13,
    'wa_scm':    14,
    'Mahal_DESI':15,
    'zc_scm':    16,
    'fs8_z015':  17,
    'fs8_z038':  18,
    'fs8_z051':  19,
    'fs8_z061':  20,
    'fs8_z070':  21,
    'fs8_z085':  22,
    'fs8_z148':  23,
    'chi2_BAO':  24,
    'chi2_CMB':  25,
    'chi2_SN':   26,
}

# eBOSS / BOSS reference data
EBOSS = {
    'z':    np.array([0.38,  0.51,  0.61,  0.698, 1.48 ]),
    'fs8':  np.array([0.497, 0.459, 0.436, 0.473, 0.462]),
    'err':  np.array([0.045, 0.038, 0.034, 0.044, 0.045]),
    'label':['BOSS DR12','BOSS DR12','BOSS DR12','eBOSS LRG','eBOSS QSO'],
}

# Step 10 posterior means (for comparison)
STEP10 = {
    'eps':    (0.00159, 0.00091),
    'zc_scm': (2.427,   0.382),
    'sigma8': (0.8349,  0.012),
    'H0':     (69.20,   0.42),
}


def load_chain(burnin=BURNIN):
    """Load chain, apply burnin, return weighted array and weights."""
    chain = np.loadtxt(CHAIN_FILE)
    n_burnin = int(len(chain) * burnin)
    chain = chain[n_burnin:]
    weights = chain[:, COL['weight']]
    print(f"  Loaded chain: {len(chain)} post-burnin samples "
          f"(burnin={burnin*100:.0f}%, {n_burnin} samples discarded)")
    print(f"  Effective samples: {(weights.sum()**2 / (weights**2).sum()):.0f}")
    return chain, weights


def wstats(x, weights):
    """Weighted mean and std."""
    w = weights / weights.sum()
    mean = np.dot(w, x)
    var  = np.dot(w, (x - mean)**2)
    return mean, np.sqrt(var)


def wquantile(x, weights, q):
    """Weighted quantile."""
    idx  = np.argsort(x)
    xs   = x[idx]
    ws   = weights[idx] / weights.sum()
    cdf  = np.cumsum(ws)
    return np.interp(q, cdf, xs)


def credible_interval(x, weights, frac=0.68):
    """Shortest credible interval containing frac of probability."""
    lo = wquantile(x, weights, 0.5*(1-frac))
    hi = wquantile(x, weights, 0.5*(1+frac))
    return lo, hi


def print_summary(chain, weights):
    print()
    print("=" * 70)
    print("  SCM Step 11 Posterior Summary")
    print("  Chain: scm_step11_mcmc.1.txt  |  Burnin: 30%")
    print("=" * 70)

    params_display = [
        ('H0',        r'H0 [km/s/Mpc]',  None),
        ('ombh2',     r'Omega_b h^2',     None),
        ('omch2',     r'Omega_c h^2',     None),
        ('tau',       r'tau',             None),
        ('ns',        r'n_s',             None),
        ('eps',       r'eps',             STEP10['eps']),
        ('A_fp',      r'A_fp',            None),
        ('sigma8',    r'sigma8',          STEP10['sigma8']),
        ('zc_scm',    r'z_c',             STEP10['zc_scm']),
        ('w0_scm',    r'w0_SCM',          None),
        ('wa_scm',    r'wa_SCM',          None),
        ('Mahal_DESI',r'Mahal_DESI',      None),
    ]

    print(f"\n  {'Parameter':<18} {'Mean':>10} {'Std':>10} "
          f"{'68% lo':>10} {'68% hi':>10}  Step10_mean")
    print("  " + "-"*68)

    for key, label, step10 in params_display:
        col = COL[key]
        x   = chain[:, col]
        mean, std = wstats(x, weights)
        lo, hi    = credible_interval(x, weights)
        s10 = f"{step10[0]:.5g}" if step10 else "---"
        print(f"  {label:<18} {mean:>10.5g} {std:>10.4g} "
              f"{lo:>10.5g} {hi:>10.5g}  {s10}")

    print()
    print("  f*sigma8 predictions (Step 11 posterior mean):")
    fs8_cols = [('fs8_z015',17,0.15,None,None),
                ('fs8_z038',18,0.38,0.497,0.045),
                ('fs8_z051',19,0.51,0.459,0.038),
                ('fs8_z061',20,0.61,0.436,0.034),
                ('fs8_z070',21,0.698,0.473,0.044),
                ('fs8_z085',22,0.85,None,None),
                ('fs8_z148',23,1.48,0.462,0.045)]
    print(f"  {'z':>6} {'fs8_SCM':>10} {'68% CI':>18}  {'data':>8} {'pull':>6}")
    print("  " + "-"*52)
    for name, col, z, data, err in fs8_cols:
        x = chain[:, col]
        mean, std = wstats(x, weights)
        lo, hi    = credible_interval(x, weights)
        if data:
            pull = (mean - data) / err
            print(f"  {z:>6.3f} {mean:>10.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"{data:>8.3f} {pull:>+6.2f}s")
        else:
            print(f"  {z:>6.3f} {mean:>10.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"{'---':>8} {'---':>6}")

    print()
    print("  Chi-squared contributions (posterior mean):")
    for key, col in [('chi2_CMB',25),('chi2_BAO',24),('chi2_SN',26)]:
        mean, _ = wstats(chain[:, col], weights)
        print(f"    {key}: {mean:.1f}")
    print("=" * 70)


def fig_step10_vs_step11(chain11, weights11):
    """Triangle-style comparison: eps, zc, sigma8, H0."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle('Step 10 vs Step 11 Posterior Comparison', fontsize=13)

    # Load Step 10 chain
    step10_chain = None
    if os.path.isfile('scm_step10_mcmc.1.txt'):
        raw = np.loadtxt('scm_step10_mcmc.1.txt')
        nb = int(len(raw) * 0.30)
        step10_chain = raw[nb:]

    pairs = [
        ('eps',    COL['eps'],    9,  r'$\varepsilon$'),
        ('zc_scm', COL['zc_scm'],16, r'$z_c$'),
        ('sigma8', COL['sigma8'],12, r'$\sigma_8$'),
        ('H0',     COL['H0'],    2,  r'$H_0$'),
    ]

    # Step 10 col mapping (different file, same structure)
    # For step10, we need to find the right columns
    # Step 10 uses SCMCAMBRepar which has fewer derived params
    # eps=9, A_fp=10, sigma8=12, zc_scm=16 (same positions since logA is dropped in both)
    STEP10_COLS = {'eps': 9, 'zc_scm': 16, 'sigma8': 12, 'H0': 2}

    colors = {'step10': '#2196F3', 'step11': '#FF5722'}

    for ax, (name, col11, col10, label) in zip(axes.flat, pairs):
        x11 = chain11[:, col11]
        w11 = weights11

        xlo = wquantile(x11, w11, 0.005)
        xhi = wquantile(x11, w11, 0.995)
        bins = np.linspace(xlo, xhi, 60)

        # Step 11
        h11, _ = np.histogram(x11, bins=bins, weights=w11, density=True)
        xc = 0.5*(bins[:-1]+bins[1:])
        ax.fill_between(xc, h11, alpha=0.4, color=colors['step11'], label='Step 11')
        ax.plot(xc, h11, color=colors['step11'], lw=1.5)

        # Step 10
        if step10_chain is not None:
            col10_idx = STEP10_COLS.get(name, col11)
            if col10_idx < step10_chain.shape[1]:
                x10 = step10_chain[:, col10_idx]
                w10 = step10_chain[:, 0]
                h10, _ = np.histogram(x10, bins=bins, weights=w10, density=True)
                ax.fill_between(xc, h10, alpha=0.3, color=colors['step10'], label='Step 10')
                ax.plot(xc, h10, color=colors['step10'], lw=1.5, ls='--')

        mean11, std11 = wstats(x11, w11)
        ax.axvline(mean11, color=colors['step11'], lw=1.5, ls=':')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Posterior density', fontsize=10)
        ax.set_xlim(xlo, xhi)

    axes[0,0].legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_step11_vs_step10.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def fig_fsigma8_step11(chain, weights):
    """f*sigma8 posterior predictive band from Step 11, vs eBOSS/BOSS."""
    z_plot = np.linspace(0.05, 2.0, 200)
    rsd    = load_rsd_data()

    # Posterior predictive band
    n_samp = 600
    idx    = np.random.choice(len(chain), size=n_samp,
                               replace=False,
                               p=weights/weights.sum())
    fs8_mat = []
    for i in idx:
        row  = chain[i]
        H0   = row[COL['H0']]
        ob   = row[COL['ombh2']]
        oc   = row[COL['omch2']]
        h    = H0/100.
        om0  = (ob+oc)/h**2
        s8   = row[COL['sigma8']]
        zc   = row[COL['zc_scm']]
        eps  = row[COL['eps']]
        try:
            r = compute_fsigma8(zc, eps, z_plot, sigma8_0=s8, om0=om0)
            fs8_mat.append(r['fs8_scm'])
        except Exception:
            pass
    fs8_mat = np.array(fs8_mat)
    med  = np.percentile(fs8_mat, 50,  axis=0)
    lo68 = np.percentile(fs8_mat, 16,  axis=0)
    hi68 = np.percentile(fs8_mat, 84,  axis=0)
    lo95 = np.percentile(fs8_mat, 2.5, axis=0)
    hi95 = np.percentile(fs8_mat, 97.5,axis=0)

    # LCDM reference
    from SCM_perturbations import E_lcdm, OM_M_LCDM, SIGMA8_LCDM
    from SCM_perturbations import solve_growth_ODE
    from scipy.interpolate import PchipInterpolator as PCHI
    def E_lcdm_a(a):
        z = 1./a - 1.
        return E_lcdm(z)
    a_l, D_l, f_l = solve_growth_ODE(E_lcdm_a, OM_M_LCDM)
    z_l = 1./a_l - 1.
    idx_s = np.argsort(z_l)
    z_l,D_l,f_l = z_l[idx_s],D_l[idx_s],f_l[idx_s]
    fs8_lcdm = PCHI(z_l,f_l)(z_plot)*PCHI(z_l,D_l)(z_plot)*SIGMA8_LCDM

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(z_plot, lo95, hi95, color='#FF5722', alpha=0.15, label='Step 11 95%')
    ax.fill_between(z_plot, lo68, hi68, color='#FF5722', alpha=0.35, label='Step 11 68%')
    ax.plot(z_plot, med, color='#FF5722', lw=2.0, label='Step 11 median')
    ax.plot(z_plot, fs8_lcdm, 'k--', lw=1.5, label=r'$\Lambda$CDM')

    # Data points
    markers = ['o','s','^','D','v','p','*']
    clrs    = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
    if rsd is not None:
        for i, row in enumerate(rsd):
            ax.errorbar(row['z'], row['fs8'], yerr=row['err'],
                        fmt=markers[i%7], color=clrs[i%7], ms=7, capsize=3,
                        label=row.get('label',''))

    ax.set_xlabel(r'Redshift $z$', fontsize=13)
    ax.set_ylabel(r'$f(z)\sigma_8(z)$', fontsize=13)
    ax.set_title('SCM Step 11: $f\\sigma_8$ Posterior Predictive vs RSD Data', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0.25, 0.65)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_fsigma8_step11.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def fig_eps_zc_2d(chain, weights):
    """2D posterior in (eps, zc) space, Step 10 vs Step 11."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    eps11 = chain[:, COL['eps']]
    zc11  = chain[:, COL['zc_scm']]

    for ax, (label, ec, zc, ww, col) in zip(axes, [
        ('Step 11', eps11, zc11, weights, '#FF5722'),
    ]):
        xlo,xhi = wquantile(ec,ww,0.01), wquantile(ec,ww,0.99)
        ylo,yhi = wquantile(zc,ww,0.01), wquantile(zc,ww,0.99)
        H, xe, ye = np.histogram2d(ec, zc, bins=60,
                                    range=[[xlo,xhi],[ylo,yhi]],
                                    weights=ww, density=True)
        H /= H.max()
        xc = 0.5*(xe[:-1]+xe[1:])
        yc = 0.5*(ye[:-1]+ye[1:])
        ax.contourf(xc, yc, H.T, levels=[0.05,0.32,1.01],
                    colors=[col,col], alpha=[0.2,0.4])
        ax.contour(xc, yc, H.T, levels=[0.05,0.32],
                   colors=[col,col], linewidths=[1.5,2.0])
        ax.axvline(wstats(ec,ww)[0], color=col, ls=':', lw=1.5)
        ax.axhline(wstats(zc,ww)[0], color=col, ls=':', lw=1.5)
        ax.set_xlabel(r'$\varepsilon$', fontsize=13)
        ax.set_ylabel(r'$z_c$', fontsize=13)
        ax.set_title(label, fontsize=12)

    # Step 10 on second axis
    ax2 = axes[1]
    ax2.set_title('Step 10 vs Step 11')
    if os.path.isfile('scm_step10_mcmc.1.txt'):
        raw = np.loadtxt('scm_step10_mcmc.1.txt')
        nb  = int(len(raw)*0.30)
        raw = raw[nb:]
        e10, z10, w10 = raw[:,9], raw[:,16], raw[:,0]
        xlo2 = min(wquantile(e10,w10,0.01), wquantile(eps11,weights,0.01))
        xhi2 = max(wquantile(e10,w10,0.99), wquantile(eps11,weights,0.99))
        ylo2 = min(wquantile(z10,w10,0.01), wquantile(zc11,weights,0.01))
        yhi2 = max(wquantile(z10,w10,0.99), wquantile(zc11,weights,0.99))
        for (ec,zc,ww,col,lab) in [
            (e10,z10,w10,'#2196F3','Step 10'),
            (eps11,zc11,weights,'#FF5722','Step 11'),
        ]:
            H,xe,ye = np.histogram2d(ec,zc,bins=60,
                                      range=[[xlo2,xhi2],[ylo2,yhi2]],
                                      weights=ww,density=True)
            H /= H.max()
            xc = 0.5*(xe[:-1]+xe[1:])
            yc = 0.5*(ye[:-1]+ye[1:])
            ax2.contour(xc,yc,H.T, levels=[0.32], colors=[col],
                        linewidths=2.0, linestyles='-')
            ax2.contourf(xc,yc,H.T, levels=[0.32,1.01],
                         colors=[col], alpha=0.2)
        ax2.set_xlabel(r'$\varepsilon$', fontsize=13)
        ax2.set_ylabel(r'$z_c$', fontsize=13)
        from matplotlib.lines import Line2D
        ax2.legend(handles=[
            Line2D([0],[0],color='#2196F3',lw=2,label='Step 10'),
            Line2D([0],[0],color='#FF5722',lw=2,label='Step 11'),
        ], fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_step11_eps_zc.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def fig_convergence():
    """R-1 convergence history from progress file."""
    data = np.loadtxt('scm_step11_mcmc.progress',
                      usecols=(0,4), comments='#')
    n_acc = data[:,0]
    r1    = data[:,1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(n_acc, r1, 'o-', ms=3, color='#FF5722', lw=1.5)
    ax.axhline(0.01, color='green', ls='--', lw=1.5, label=r'$R-1 = 0.01$ target')
    ax.axhline(0.05, color='orange', ls=':', lw=1.2, label=r'$R-1 = 0.05$')
    ax.set_xlabel('Accepted samples', fontsize=12)
    ax.set_ylabel(r'$R - 1$', fontsize=12)
    ax.set_title('Step 11 MCMC Convergence History', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, n_acc[-1]*1.05)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_step11_convergence.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("\n" + "="*70)
    print("  SCM Step 11 Posterior Analysis")
    print("="*70)

    chain, weights = load_chain()
    print_summary(chain, weights)

    print("\n  Generating figures...")
    fig_step10_vs_step11(chain, weights)
    fig_fsigma8_step11(chain, weights)
    fig_eps_zc_2d(chain, weights)
    fig_convergence()

    print("\n  Analysis complete. Figures saved to figures/")
