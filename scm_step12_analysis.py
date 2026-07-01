import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_CHAIN12 = os.path.join(_HERE, 'scm_step12_mcmc.1.txt')
_CHAIN11 = os.path.join(_HERE, 'scm_step11_mcmc.1.txt')
_FIG_DIR  = os.path.join(_HERE, 'figures')
os.makedirs(_FIG_DIR, exist_ok=True)

BURNIN = 0.30   # fraction of chain to discard

# column indices
# Step 12
C12 = dict(
    weight=0, mlp=1, A_planck=2, H0=3, logA=4, ns=5,
    ombh2=6, omch2=7, tau=8, eps=9, A_fp=10,
    DES_DzS1=11, DES_DzS2=12, DES_DzS3=13, DES_DzS4=14,
    DES_m1=15, DES_m2=16, DES_m3=17, DES_m4=18,
    DES_AIA=19, DES_alphaIA=20,
    As=21, Mahal_DESI=22,
    fs8_015=23, fs8_038=24, fs8_051=25, fs8_061=26,
    fs8_070=27, fs8_085=28, fs8_148=29,
    s8=30, sigma8=31, sigma8_z03=32,
    w0_scm=33, wa_scm=34, zc_scm=35,
    chi2_BAO=36, chi2_CMB=37, chi2_SN=38,
    mlprior=39, mlprior0=40, chi2=41,
    chi2_desi=42, chi2_lrg=43, chi2_qso=44,
    chi2_des=45, chi2_plik=46, chi2_lens=47,
    chi2_lowEE=48, chi2_lowTT=49, chi2_pantheon=50,
)

# Step 11
C11 = dict(
    weight=0, mlp=1, H0=2, ombh2=3, omch2=4, tau=5, ns=6, logA=7,
    A_planck=8, eps=9, A_fp=10, As=11, sigma8=12,
    w0_scm=13, wa_scm=14, Mahal_DESI=15, zc_scm=16,
    fs8_015=17, fs8_038=18, fs8_051=19, fs8_061=20,
    fs8_070=21, fs8_085=22, fs8_148=23,
    chi2_BAO=24, chi2_CMB=25, chi2_SN=26,
    mlprior=27, mlprior0=28, chi2=29,
    chi2_plik=30, chi2_lowTT=31, chi2_lowEE=32,
    chi2_desi=33, chi2_pantheon=34, chi2_lrg=35, chi2_qso=36,
)


# helpers

def load_chain(path, burnin=BURNIN):
    """Load chain, discard burnin fraction, return array and weights."""
    data = np.loadtxt(path, comments='#')
    n = len(data)
    cut = int(burnin * n)
    data = data[cut:]
    weights = data[:, 0]
    return data, weights


def wstats(arr, w):
    """Weighted mean and std."""
    mean = np.average(arr, weights=w)
    var  = np.average((arr - mean)**2, weights=w)
    return mean, np.sqrt(var)


def percentile_w(arr, w, q):
    """Weighted percentile via CDF."""
    idx  = np.argsort(arr)
    cs   = np.cumsum(w[idx])
    cs  /= cs[-1]
    return np.interp(q/100.0, cs, arr[idx])


def fmt(mean, std):
    """Format mean ± std with appropriate precision."""
    if std == 0:
        return f'{mean:.4g}'
    mag = int(np.floor(np.log10(std)))
    dec = max(0, -mag + 1)
    return f'{mean:.{dec}f} ± {std:.{dec}f}'


def compute_s8_step11(d11, c=C11):
    """S8 = sigma8 * sqrt(Omega_m / 0.3) from Step 11 chain."""
    h = d11[:, c['H0']] / 100.0
    omega_m = (d11[:, c['ombh2']] + d11[:, c['omch2']]) / h**2
    return d11[:, c['sigma8']] * np.sqrt(omega_m / 0.3)


# load chains

print("Loading Step 12 chain ...")
d12, w12 = load_chain(_CHAIN12)

print(f"  {len(d12)} samples after {int(BURNIN*100)}% burnin")
print(f"  Effective N ~{w12.sum():.0f} total MCMC steps")

print("Loading Step 11 chain ...")
d11, w11 = load_chain(_CHAIN11)
s8_11 = compute_s8_step11(d11, C11)
print(f"  {len(d11)} samples")


# summary table

print()
print("=" * 70)
print("  STEP 12 POSTERIOR SUMMARY")
print("=" * 70)

params_to_print = [
    ('H0',         'H_0 [km/s/Mpc]', C12['H0']),
    ('ombh2',      'Omega_b h^2',     C12['ombh2']),
    ('omch2',      'Omega_c h^2',     C12['omch2']),
    ('tau',        'tau',             C12['tau']),
    ('ns',         'n_s',             C12['ns']),
    ('logA',       'ln(10^10 A_s)',   C12['logA']),
    ('eps',        'epsilon',         C12['eps']),
    ('zc_scm',     'z_c',             C12['zc_scm']),
    ('A_fp',       'A_fp',            C12['A_fp']),
    ('sigma8',     'sigma_8',         C12['sigma8']),
    ('s8',         'S_8',             C12['s8']),
    ('sigma8_z03', 'sigma_8(z=0.3)',  C12['sigma8_z03']),
    ('DES_AIA',    'A_IA',            C12['DES_AIA']),
    ('DES_alphaIA','alpha_IA',        C12['DES_alphaIA']),
]

print(f"  {'Parameter':<22}  {'Step 12':>20}  {'Step 11':>20}")
print(f"  {'-'*22}  {'-'*20}  {'-'*20}")

# Step 11 reference values (posterior mean ± std)
s11_refs = {
    'H0':    wstats(d11[:, C11['H0']],    w11),
    'ombh2': wstats(d11[:, C11['ombh2']], w11),
    'omch2': wstats(d11[:, C11['omch2']], w11),
    'tau':   wstats(d11[:, C11['tau']],   w11),
    'ns':    wstats(d11[:, C11['ns']],    w11),
    'logA':  wstats(d11[:, C11['logA']],  w11),
    'eps':   wstats(d11[:, C11['eps']],   w11),
    'zc_scm':wstats(d11[:, C11['zc_scm']],w11),
    'A_fp':  wstats(d11[:, C11['A_fp']],  w11),
    'sigma8':wstats(d11[:, C11['sigma8']],w11),
    's8':    wstats(s8_11, w11),
    'sigma8_z03': (None, None),
    'DES_AIA':    (None, None),
    'DES_alphaIA':(None, None),
}

for key, label, col in params_to_print:
    m12, s12 = wstats(d12[:, col], w12)
    ref = s11_refs.get(key, (None, None))
    str12 = fmt(m12, s12)
    str11 = fmt(*ref) if ref[0] is not None else '  —'
    print(f"  {label:<22}  {str12:>20}  {str11:>20}")

# S8 tension with DES Y1 shear
m_s8_12, s_s8_12 = wstats(d12[:, C12['s8']], w12)
des_s8_central = 0.782;  des_s8_err = 0.027
tension = (m_s8_12 - des_s8_central) / np.sqrt(s_s8_12**2 + des_s8_err**2)
print()
print(f"  SCM Step 12 S8 = {m_s8_12:.3f} ± {s_s8_12:.3f}")
print(f"  DES Y1 shear   = {des_s8_central:.3f} ± {des_s8_err:.3f}")
print(f"  Tension = {tension:.2f} sigma")

print()
print("  CHI2 BREAKDOWN (posterior mean)")
chi2_terms = [
    ('Planck CMB (plik-lite)',     C12['chi2_plik']),
    ('Planck CMB (lowl-TT)',       C12['chi2_lowTT']),
    ('Planck CMB (lowl-EE)',       C12['chi2_lowEE']),
    ('Planck lensing',             C12['chi2_lens']),
    ('DES Y1 shear',               C12['chi2_des']),
    ('DESI 2024 BAO',              C12['chi2_desi']),
    ('eBOSS LRG BAO+FS',          C12['chi2_lrg']),
    ('eBOSS QSO BAO+FS',          C12['chi2_qso']),
    ('Pantheon+',                  C12['chi2_pantheon']),
    ('Total chi2',                 C12['chi2']),
]
for name, col in chi2_terms:
    m, _ = wstats(d12[:, col], w12)
    print(f"    {name:<32}  chi2 = {m:.1f}")

print("=" * 70)


# matplotlib style
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
})
SCM_ORANGE = '#e87722'
LCDM_BLUE  = '#3366cc'
DES_RED    = '#c62026'
PLANCK_GRN = '#117733'


# figure: S8 comparison
print("\n[1/4] fig_s8_step12.pdf ...")

fig, ax = plt.subplots(figsize=(7, 4.5))

surveys = [
    # (central, -err, +err, label, color, marker)
    (0.832, 0.013, 0.013,
     r'Planck 2018 (CMB)', PLANCK_GRN, 's'),
    (m_s8_12, s_s8_12, s_s8_12,
     r'SCM Step~12 (this work)', SCM_ORANGE, 'D'),
    (wstats(s8_11, w11)[0], wstats(s8_11, w11)[1], wstats(s8_11, w11)[1],
     r'SCM Step~11 (this work)', '#ff9944', 'd'),
    (0.776, 0.017, 0.017,
     r'DES Y3 $3\times2$pt', DES_RED, 'o'),
    (0.782, 0.027, 0.027,
     r'DES Y1 shear (in Step~12)', '#aa0000', 'v'),
    (0.759, 0.024, 0.021,
     r'KiDS-1000', '#6600cc', '^'),
    (0.780, 0.030, 0.033,
     r'HSC Y1', '#007755', 'p'),
]

y_pos = list(range(len(surveys)))[::-1]   # top → bottom
for i, (cen, em, ep, label, color, mk) in enumerate(surveys):
    y = y_pos[i]
    ax.errorbar(cen, y, xerr=[[em], [ep]],
                fmt=mk, color=color, ms=7, capsize=4, lw=1.5,
                zorder=3 + (1 if 'Step~12' in label else 0))
    ax.text(cen, y + 0.32, label, ha='center', va='bottom',
            fontsize=8, color=color)

# Shade SCM Step 12 band
ax.axvspan(m_s8_12 - s_s8_12, m_s8_12 + s_s8_12,
           color=SCM_ORANGE, alpha=0.12, label='SCM Step 12 1-$\\sigma$')
ax.axvline(m_s8_12, color=SCM_ORANGE, lw=1.2, ls='--', alpha=0.7)

ax.set_xlabel(r'$S_8 = \sigma_8\,\sqrt{\Omega_{\rm m}/0.3}$')
ax.set_xlim(0.70, 0.89)
ax.set_ylim(-0.8, len(surveys) + 0.5)
ax.set_yticks([])
ax.yaxis.set_visible(False)

# Tension annotation
ax.annotate(
    rf'SCM vs DES: ${tension:.1f}\,\sigma$ tension',
    xy=(des_s8_central, y_pos[-3]), xytext=(0.74, y_pos[-3] - 0.6),
    fontsize=8.5, color=DES_RED,
    arrowprops=dict(arrowstyle='->', color=DES_RED, lw=1),
)

ax.set_title(r'$S_8$ comparison: SCM Step~12 and weak-lensing surveys', fontsize=11)
ax.grid(axis='x', alpha=0.3, ls=':')
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_s8_step12.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: parameter shifts
print("[2/4] fig_param_shift.pdf ...")

shift_params = [
    ('H0',    r'$H_0$',           C12['H0'],    C11['H0'],    d11),
    ('ombh2', r'$\Omega_b h^2$',  C12['ombh2'], C11['ombh2'], d11),
    ('omch2', r'$\Omega_c h^2$',  C12['omch2'], C11['omch2'], d11),
    ('ns',    r'$n_s$',           C12['ns'],    C11['ns'],    d11),
    ('tau',   r'$\tau$',          C12['tau'],   C11['tau'],   d11),
    ('logA',  r'$\ln(10^{10}A_s)$',C12['logA'], C11['logA'],  d11),
    ('eps',   r'$\varepsilon$',   C12['eps'],   C11['eps'],   d11),
    ('zc',    r'$z_c$',           C12['zc_scm'],C11['zc_scm'],d11),
    ('sigma8',r'$\sigma_8$',      C12['sigma8'],C11['sigma8'],d11),
    ('s8',    r'$S_8$',           C12['s8'],    None,         None),
]

labels = []
shifts = []
yerrs  = []

for key, label, col12, col11, dn in shift_params:
    m12, s12 = wstats(d12[:, col12], w12)
    if col11 is not None and dn is not None:
        m11, s11 = wstats(dn[:, col11], w11 if dn is d11 else w12)
        # normalise by combined uncertainty
        sigma_comb = np.sqrt(s12**2 + s11**2)
        delta = (m12 - m11) / sigma_comb if sigma_comb > 0 else 0.0
        err   = 1.0  # ±1 sigma bars already in units of sigma_comb
    elif key == 's8':
        m11_s8, s11_s8 = wstats(s8_11, w11)
        sigma_comb = np.sqrt(s12**2 + s11_s8**2)
        delta = (m12 - m11_s8) / sigma_comb if sigma_comb > 0 else 0.0
        err   = 1.0
    else:
        continue
    labels.append(label)
    shifts.append(delta)
    yerrs.append(err)

fig, ax = plt.subplots(figsize=(6.5, 4))

x = np.arange(len(labels))
colors = [SCM_ORANGE if abs(s) > 0.5 else '#888888' for s in shifts]
bars = ax.bar(x, shifts, color=colors, alpha=0.75, width=0.6, zorder=2)

ax.axhline(0, color='k', lw=0.8)
ax.axhline( 1, color='gray', lw=0.6, ls=':', alpha=0.6)
ax.axhline(-1, color='gray', lw=0.6, ls=':', alpha=0.6)
ax.axhline( 2, color='gray', lw=0.6, ls='--', alpha=0.4)
ax.axhline(-2, color='gray', lw=0.6, ls='--', alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(r'$(p_{12} - p_{11})\,/\,\sigma_{\rm comb}$')
ax.set_title(r'Parameter shifts: Step~12 vs Step~11 (normalised)', fontsize=11)
ax.set_ylim(-3.5, 3.5)
ax.grid(axis='y', alpha=0.3, ls=':')

# Annotate numerical shifts
for xi, delta in zip(x, shifts):
    ax.text(xi, delta + (0.15 if delta >= 0 else -0.25),
            f'{delta:+.2f}$\\sigma$', ha='center', fontsize=7)

fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_param_shift.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: DES nuisance posteriors
print("[3/4] fig_des_nuisance.pdf ...")

des_params = [
    (C12['DES_DzS1'], r'$\Delta z_{S,1}$', 0.0,   0.016),
    (C12['DES_DzS2'], r'$\Delta z_{S,2}$', 0.0,   0.013),
    (C12['DES_DzS3'], r'$\Delta z_{S,3}$', 0.0,   0.011),
    (C12['DES_DzS4'], r'$\Delta z_{S,4}$', 0.0,   0.022),
    (C12['DES_m1'],   r'$m_1$',            0.012, 0.023),
    (C12['DES_m2'],   r'$m_2$',            0.012, 0.023),
    (C12['DES_m3'],   r'$m_3$',            0.012, 0.023),
    (C12['DES_m4'],   r'$m_4$',            0.012, 0.023),
    (C12['DES_AIA'],  r'$A_{\rm IA}$',     0.5,   1.5),
    (C12['DES_alphaIA'], r'$\alpha_{\rm IA}$', -5.0, 5.0),
]

fig, axes = plt.subplots(2, 5, figsize=(13, 5))
axes = axes.flatten()

n_kde = 200
for i, (col, label, prior_mean, prior_std) in enumerate(des_params):
    ax = axes[i]
    vals = d12[:, col]
    mn, sd = wstats(vals, w12)

    # KDE using weighted samples
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, weights=w12/w12.sum(), bw_method='scott')
        lo = mn - 4*sd; hi = mn + 4*sd
        xg = np.linspace(lo, hi, n_kde)
        ax.fill_between(xg, kde(xg), alpha=0.4, color=SCM_ORANGE)
        ax.plot(xg, kde(xg), color=SCM_ORANGE, lw=1.5)
    except Exception:
        ax.hist(vals, weights=w12, bins=30, color=SCM_ORANGE, alpha=0.5,
                density=True)

    # Gaussian prior overlay
    xp = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 200)
    yp = np.exp(-0.5*((xp - prior_mean)/prior_std)**2) / (prior_std*np.sqrt(2*np.pi))
    ax.plot(xp, yp, 'k--', lw=0.9, alpha=0.6, label='prior')

    ax.axvline(mn, color=SCM_ORANGE, lw=1.0, ls='-')
    ax.axvline(mn - sd, color=SCM_ORANGE, lw=0.7, ls=':')
    ax.axvline(mn + sd, color=SCM_ORANGE, lw=0.7, ls=':')

    ax.set_xlabel(label, fontsize=9)
    ax.set_yticks([])
    ax.set_title(f'${mn:.3f} \\pm {sd:.3f}$', fontsize=8)
    ax.grid(alpha=0.2)

axes[0].legend(fontsize=7, loc='upper left')
fig.suptitle(r'DES Y1 shear nuisance parameters: Step~12 posterior (orange) vs prior (dashed)',
             fontsize=10)
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_des_nuisance.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: f*sigma8 posteriors
print("[4/4] fig_fs8_step12.pdf ...")

# RSD/BAO+FS data points (z_eff, fs8, sigma_fs8)
rsd_data = [
    (0.150, 0.490, 0.145, 'BOSS DR7 MGS',    '#999999'),
    (0.380, 0.497, 0.045, 'BOSS LOWZ',        '#4477aa'),
    (0.510, 0.458, 0.038, 'BOSS CMASS',       '#4477aa'),
    (0.610, 0.436, 0.034, 'BOSS CMASS',       '#4477aa'),
    (0.700, 0.448, 0.043, 'eBOSS ELG',        '#bb5511'),
    (0.850, 0.315, 0.095, 'VIPERS PDR2',      '#228833'),
    (1.480, 0.462, 0.045, 'eBOSS QSO',        '#ccbb44'),
]

fs8_z_vals  = [0.15, 0.38, 0.51, 0.61, 0.70, 0.85, 1.48]
fs8_cols_12 = [C12['fs8_015'], C12['fs8_038'], C12['fs8_051'],
               C12['fs8_061'], C12['fs8_070'], C12['fs8_085'], C12['fs8_148']]
fs8_cols_11 = [C11['fs8_015'], C11['fs8_038'], C11['fs8_051'],
               C11['fs8_061'], C11['fs8_070'], C11['fs8_085'], C11['fs8_148']]

fig, ax = plt.subplots(figsize=(7, 4.5))

# Step 12 posterior band
m12_fs8 = [wstats(d12[:, c], w12)[0] for c in fs8_cols_12]
s12_fs8 = [wstats(d12[:, c], w12)[1] for c in fs8_cols_12]
ax.fill_between(fs8_z_vals,
                [m - s for m, s in zip(m12_fs8, s12_fs8)],
                [m + s for m, s in zip(m12_fs8, s12_fs8)],
                color=SCM_ORANGE, alpha=0.25, label='SCM Step 12 $1\\sigma$')
ax.plot(fs8_z_vals, m12_fs8, '-', color=SCM_ORANGE, lw=2.0,
        label='SCM Step 12 mean')

# Step 11 posterior band (lighter)
m11_fs8 = [wstats(d11[:, c], w11)[0] for c in fs8_cols_11]
s11_fs8 = [wstats(d11[:, c], w11)[1] for c in fs8_cols_11]
ax.fill_between(fs8_z_vals,
                [m - s for m, s in zip(m11_fs8, s11_fs8)],
                [m + s for m, s in zip(m11_fs8, s11_fs8)],
                color='#ff9944', alpha=0.15, label='SCM Step 11 $1\\sigma$')
ax.plot(fs8_z_vals, m11_fs8, '--', color='#ff9944', lw=1.2,
        label='SCM Step 11 mean')

# Data points
for z, fs8, err, label, color in rsd_data:
    ax.errorbar(z, fs8, yerr=err, fmt='o', color=color, ms=5,
                capsize=3, lw=1.2, zorder=4)

# Custom legend entries for data
legend_handles = [
    mpatches.Patch(color=SCM_ORANGE, alpha=0.4, label='SCM Step 12 $1\\sigma$'),
    plt.Line2D([0], [0], color=SCM_ORANGE, lw=2, label='SCM Step 12 mean'),
    mpatches.Patch(color='#ff9944', alpha=0.3, label='SCM Step 11 $1\\sigma$'),
    plt.Line2D([0], [0], color='#ff9944', lw=1.2, ls='--', label='SCM Step 11 mean'),
    plt.Line2D([0], [0], marker='o', color='#4477aa', ls='None', ms=5, label='BOSS/eBOSS'),
    plt.Line2D([0], [0], marker='o', color='#ccbb44', ls='None', ms=5, label='eBOSS QSO'),
    plt.Line2D([0], [0], marker='o', color='#228833', ls='None', ms=5, label='VIPERS'),
]
ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, fontsize=7.5)

ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$f\sigma_8(z)$')
ax.set_title(r'Growth rate $f\sigma_8$: SCM Step~12 posterior vs RSD data', fontsize=11)
ax.set_xlim(0.0, 1.7)
ax.set_ylim(0.2, 0.70)
ax.grid(alpha=0.3, ls=':')
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_fs8_step12.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# numbers for paper
print()
print("=" * 70)
print("  NUMBERS FOR SCM V PAPER")
print("=" * 70)

h_12, _ = wstats(d12[:, C12['H0']], w12)
ombh2_12, _ = wstats(d12[:, C12['ombh2']], w12)
omch2_12, _ = wstats(d12[:, C12['omch2']], w12)
h_12_h = h_12 / 100.0
omega_m_12 = (ombh2_12 + omch2_12) / h_12_h**2

print(f"  H0      = {fmt(*wstats(d12[:, C12['H0']],    w12))} km/s/Mpc")
print(f"  ombh2   = {fmt(*wstats(d12[:, C12['ombh2']], w12))}")
print(f"  omch2   = {fmt(*wstats(d12[:, C12['omch2']], w12))}")
print(f"  Omega_m = {omega_m_12:.4f}")
print(f"  sigma8  = {fmt(*wstats(d12[:, C12['sigma8']],w12))}")
print(f"  S8      = {fmt(*wstats(d12[:, C12['s8']],    w12))}")
print(f"  eps     = {fmt(*wstats(d12[:, C12['eps']],   w12))}")
print(f"  z_c     = {fmt(*wstats(d12[:, C12['zc_scm']],w12))}")
print()
print(f"  DES chi2 = {wstats(d12[:, C12['chi2_des']],   w12)[0]:.1f}")
print(f"  Lens chi2= {wstats(d12[:, C12['chi2_lens']],  w12)[0]:.1f}")
print(f"  S8 tension vs DES Y1: {tension:.2f} sigma")
print("=" * 70)

print("\n  All figures saved to:", _FIG_DIR)
