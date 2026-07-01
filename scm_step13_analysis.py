import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# paths
_HERE    = os.path.dirname(os.path.abspath(__file__))
_CHAIN13 = os.path.join(_HERE, 'scm_step13v4_mcmc.1.txt')
_CHAIN12 = os.path.join(_HERE, 'scm_step12_mcmc.1.txt')
_FIG_DIR = os.path.join(_HERE, 'figures')
os.makedirs(_FIG_DIR, exist_ok=True)

BURNIN = 0.30

# matplotlib style
plt.rcParams.update({
    'font.family':    'serif',
    'text.usetex':    True,
    'font.size':      10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi':     150,
})
SCM_ORANGE = '#e87722'
LCDM_BLUE  = '#3366cc'
DES_RED    = '#c62026'
PLANCK_GRN = '#117733'
PSZ_BLUE   = '#1144aa'

# external constraints
# S8 = sigma8 * sqrt(Omega_m / 0.3)
DES_Y1_S8    = (0.782, 0.027)   # DES Y1 shear (Troxel+ 2018)
DES_Y3_S8    = (0.776, 0.017)   # DES Y3 3x2pt (Abbott+ 2022)
KIDS_S8      = (0.759, 0.024)   # KiDS-1000 (Asgari+ 2021)
PLANCK_CMB_S8 = (0.832, 0.013)  # Planck 2018 TT+TE+EE+lowE

# S8_cluster = sigma8 * (Omega_m / 0.3)^0.3
# Planck CMB prediction: sigma8=0.832, Omega_m=0.315 → 0.832*(0.315/0.3)^0.3
_om_planck = 0.315
PLANCK_CMB_S8CL = (0.832 * (_om_planck / 0.3)**0.3, 0.014)  # ~0.847

# Planck 2015 XXIV PSZ2 cluster constraint (with (1-b)=0.8 prior)
# Reported: sigma8*(Omega_m/0.3)^0.3 = 0.762 ± 0.025
PSZ2_S8CL    = (0.762, 0.025)   # Planck 2015 XXIV Table 7

# chain loading

def load_chain_with_header(path, burnin=BURNIN):
    """
    Load a Cobaya chain file; parse '#' header line for column names.
    Returns (data, weights, col_dict) where col_dict maps name → int index.
    """
    header_line = None
    with open(path, 'r') as f:
        for line in f:
            ls = line.strip()
            if ls.startswith('#'):
                header_line = ls
            elif ls:
                break

    if header_line is None:
        raise ValueError(f'No header found in {path}')

    col_names = header_line.lstrip('#').split()
    col_dict  = {name: i for i, name in enumerate(col_names)}

    data    = np.loadtxt(path, comments='#')
    cut     = int(burnin * len(data))
    data    = data[cut:]
    weights = data[:, col_dict.get('weight', 0)]
    return data, weights, col_dict


def wstats(arr, w):
    """Weighted mean and std."""
    mean = np.average(arr, weights=w)
    var  = np.average((arr - mean)**2, weights=w)
    return mean, np.sqrt(var)


def percentile_w(arr, w, q):
    """Weighted percentile (q in percent 0–100)."""
    idx = np.argsort(arr)
    cs  = np.cumsum(w[idx])
    cs /= cs[-1]
    return float(np.interp(q / 100.0, cs, arr[idx]))


def fmt(mean, std):
    """Format mean ± std at appropriate precision."""
    if not np.isfinite(std) or std == 0:
        return f'{mean:.4g}'
    mag = int(np.floor(np.log10(std)))
    dec = max(0, -mag + 1)
    return f'{mean:.{dec}f} \\pm {std:.{dec}f}'


def get_col(data, col_dict, name):
    """Return column array by name, or None if not present."""
    idx = col_dict.get(name)
    return data[:, idx] if idx is not None else None


def col_wstats(data, w, col_dict, name):
    """wstats for a named column; returns (nan, 0) if column missing."""
    arr = get_col(data, col_dict, name)
    if arr is None:
        return float('nan'), 0.0
    return wstats(arr, w)


# load chains

print("Loading Step 13 chain ...")
d13, w13, C13 = load_chain_with_header(_CHAIN13)
print(f"  {len(d13)} rows after {int(BURNIN*100)}% burnin  "
      f"(eff N ~ {w13.sum():.0f} steps)")
print(f"  Free params detected: "
      f"{[k for k in C13 if k not in ('weight','-logposterior') and not k.startswith('chi2')][:8]} ...")

print("Loading Step 12 chain ...")
d12, w12, C12 = load_chain_with_header(_CHAIN12)
print(f"  {len(d12)} rows after {int(BURNIN*100)}% burnin")

# derived quantities

def omega_m(data, cd):
    h     = get_col(data, cd, 'H0') / 100.0
    om    = (get_col(data, cd, 'ombh2') + get_col(data, cd, 'omch2')) / h**2
    return om


def compute_s8(data, cd):
    """S8 = sigma8 * sqrt(Omega_m / 0.3); use stored column if available."""
    if 's8' in cd:
        return get_col(data, cd, 's8')
    return get_col(data, cd, 'sigma8') * np.sqrt(omega_m(data, cd) / 0.3)


def compute_s8_cluster(data, cd):
    """S8_cluster = sigma8 * (Omega_m/0.3)^0.3; always compute from sigma8/omega_m
    because the stored s8_cluster column is zero when eps is fixed to 0."""
    return get_col(data, cd, 'sigma8') * (omega_m(data, cd) / 0.3)**0.3


s8_13   = compute_s8(d13, C13)
s8cl_13 = compute_s8_cluster(d13, C13)
s8_12   = compute_s8(d12, C12)
s8cl_12 = compute_s8_cluster(d12, C12)

m_s8_13,   s_s8_13   = wstats(s8_13,   w13)
m_s8cl_13, s_s8cl_13 = wstats(s8cl_13, w13)
m_s8_12,   s_s8_12   = wstats(s8_12,   w12)
m_s8cl_12, s_s8cl_12 = wstats(s8cl_12, w12)

# summary table

print()
print("=" * 76)
print("  STEP 13 POSTERIOR SUMMARY")
print("=" * 76)

PRINT_PARAMS = [
    ('H0',          r'H_0  [km/s/Mpc]'),
    ('ombh2',       r'Omega_b h^2'),
    ('omch2',       r'Omega_c h^2'),
    ('tau',         r'tau'),
    ('ns',          r'n_s'),
    ('logA',        r'ln(10^10 A_s)'),
    ('eps',         r'epsilon (SCM)'),
    ('zc_scm',      r'z_c (SCM)'),
    ('A_fp',        r'A_fp (SCM)'),
    ('sigma8',      r'sigma_8'),
    ('s8',          r'S_8 (WL)'),
    ('s8_cluster',  r'S8_cluster (SZ)'),
    ('sigma8_z01',  r'sigma_8(z=0.1)'),
    ('sigma8_z05',  r'sigma_8(z=0.5)'),
    ('sigma8_z08',  r'sigma_8(z=0.8)'),
    ('mass_bias_b', r'mass_bias b'),
    ('log_Y_star',  r'log10(Y_star)'),
    ('DES_AIA',     r'A_IA  (DES)'),
    ('DES_alphaIA', r'alpha_IA  (DES)'),
]

print(f"  {'Parameter':<24}  {'Step 13':>22}  {'Step 12':>22}")
print(f"  {'-'*24}  {'-'*22}  {'-'*22}")

for key, label in PRINT_PARAMS:
    m13p, s13p = col_wstats(d13, w13, C13, key)
    m12p, s12p = col_wstats(d12, w12, C12, key)
    # Fall back to derived S8/S8cl arrays for these two
    if key == 's8':
        m13p, s13p = m_s8_13,   s_s8_13
        m12p, s12p = m_s8_12,   s_s8_12
    if key == 's8_cluster':
        m13p, s13p = m_s8cl_13, s_s8cl_13
        m12p, s12p = m_s8cl_12, s_s8cl_12
    str13 = fmt(m13p, s13p) if np.isfinite(m13p) else '  —'
    str12 = fmt(m12p, s12p) if np.isfinite(m12p) else '  —'
    print(f"  {label:<24}  {str13:>22}  {str12:>22}")

# tension analysis

def tension(a, sa, b, sb):
    denom = np.sqrt(sa**2 + sb**2)
    return (a - b) / denom if denom > 0 else float('nan')

t_des_s8    = tension(m_s8_13,   s_s8_13,   *DES_Y1_S8)
t_psz_s8cl  = tension(m_s8cl_13, s_s8cl_13, *PSZ2_S8CL)
t_cmb_s8cl  = tension(m_s8cl_13, s_s8cl_13, *PLANCK_CMB_S8CL)
t_wl_sz     = tension(m_s8_13,   s_s8_13,   m_s8cl_13, s_s8cl_13)

print()
print("  TENSION ANALYSIS")
print(f"  SCM Step 13  S8 (WL)      = {m_s8_13:.3f} ± {s_s8_13:.3f}")
print(f"  SCM Step 13  S8_cluster   = {m_s8cl_13:.3f} ± {s_s8cl_13:.3f}")
print(f"  DES Y1 shear S8           = {DES_Y1_S8[0]:.3f} ± {DES_Y1_S8[1]:.3f}")
print(f"  Planck SZ    S8_cluster   = {PSZ2_S8CL[0]:.3f} ± {PSZ2_S8CL[1]:.3f}")
print(f"  Planck CMB   S8_cluster   = {PLANCK_CMB_S8CL[0]:.3f} ± {PLANCK_CMB_S8CL[1]:.3f}")
print()
print(f"  SCM S8 (WL) vs DES Y1:           {t_des_s8:+.2f} sigma")
print(f"  SCM S8_cluster vs Planck SZ:      {t_psz_s8cl:+.2f} sigma")
print(f"  SCM S8_cluster vs Planck CMB:     {t_cmb_s8cl:+.2f} sigma")
print(f"  SCM S8 (WL) vs S8_cluster (int):  {t_wl_sz:+.2f} sigma")

# chi2 breakdown

print()
print("  CHI2 BREAKDOWN (posterior mean)")
chi2_cols = {k: v for k, v in C13.items() if 'chi2' in k.lower()}
for name in sorted(chi2_cols):
    m, _ = wstats(d13[:, chi2_cols[name]], w13)
    print(f"    {name:<50}  {m:.1f}")

print("=" * 76)


# figure helpers

def kde_1d(vals, w, n=300):
    """Weighted 1D KDE → (x, y)."""
    mn, sd = wstats(vals, w)
    lo = mn - 4.5 * sd;  hi = mn + 4.5 * sd
    kde = gaussian_kde(vals, weights=w / w.sum(), bw_method='scott')
    x = np.linspace(lo, hi, n)
    return x, kde(x)


def contour_2d(x, y, w, ax, color, levels=(0.68, 0.95), lw=1.5, alpha_fill=0.12):
    """Draw 2D weighted KDE contours at probability levels. Returns patch for legend."""
    wn  = w / w.sum()
    kde = gaussian_kde(np.vstack([x, y]), weights=wn, bw_method='scott')

    xg = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 120)
    yg = np.linspace(np.percentile(y, 0.5), np.percentile(y, 99.5), 120)
    xx, yy = np.meshgrid(xg, yg)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Convert level fractions → density thresholds
    z_flat = np.sort(zz.ravel())[::-1]
    z_cdf  = np.cumsum(z_flat) / z_flat.sum()
    thresholds = [float(z_flat[np.searchsorted(z_cdf, lv)]) for lv in levels]

    ax.contourf(xx, yy, zz, levels=sorted(thresholds) + [zz.max() * 10],
                colors=[color], alpha=alpha_fill)
    ax.contour(xx, yy, zz, levels=sorted(thresholds), colors=[color], linewidths=lw)

    return mpatches.Patch(color=color, alpha=max(alpha_fill + 0.2, 0.35))


# figure: S8 forest
print("\n[1/4] fig_s8_cluster_step13.pdf ...")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

# --- left panel: S8 (weak lensing) ---
ax = axes[0]
surveys_wl = [
    (PLANCK_CMB_S8[0], PLANCK_CMB_S8[1], r'Planck~2018 (CMB)',        PLANCK_GRN, 's'),
    (m_s8_13, s_s8_13,                   r'SCM Step~13 (this work)',   SCM_ORANGE, 'D'),
    (m_s8_12, s_s8_12,                   r'SCM Step~12 (prior step)',  '#ff9944',  'd'),
    (DES_Y3_S8[0], DES_Y3_S8[1],         r'DES~Y3 $3\times2$pt',      DES_RED,    'o'),
    (DES_Y1_S8[0], DES_Y1_S8[1],         r'DES~Y1 shear',             '#aa0000',  'v'),
    (KIDS_S8[0], KIDS_S8[1],             r'KiDS-1000',                '#6600cc',  '^'),
]
y_pos = list(range(len(surveys_wl)))[::-1]
for i, (cen, err, label, color, mk) in enumerate(surveys_wl):
    y = y_pos[i]
    ax.errorbar(cen, y, xerr=err, fmt=mk, color=color, ms=7,
                capsize=4, lw=1.5, zorder=3)
    ax.text(cen, y + 0.36, label, ha='center', va='bottom',
            fontsize=7.5, color=color)
ax.axvspan(m_s8_13 - s_s8_13, m_s8_13 + s_s8_13,
           color=SCM_ORANGE, alpha=0.13)
ax.axvline(m_s8_13, color=SCM_ORANGE, lw=1.2, ls='--', alpha=0.7)
ax.annotate(
    rf'vs DES~Y1: ${t_des_s8:+.1f}\,\sigma$',
    xy=(DES_Y1_S8[0], y_pos[-2]), xytext=(0.74, y_pos[-2] - 0.65),
    fontsize=8, color='#aa0000',
    arrowprops=dict(arrowstyle='->', color='#aa0000', lw=0.9),
)
ax.set_xlabel(r'$S_8 = \sigma_8\,\sqrt{\Omega_{\rm m}/0.3}$')
ax.set_xlim(0.70, 0.90)
ax.set_ylim(-0.9, len(surveys_wl) + 0.9)
ax.set_yticks([])
ax.set_title(r'$S_8$: weak-lensing combination', fontsize=10)
ax.grid(axis='x', alpha=0.28, ls=':')

# --- right panel: S8_cluster (cluster counts) ---
ax = axes[1]
surveys_sz = [
    (PLANCK_CMB_S8CL[0], PLANCK_CMB_S8CL[1], r'Planck~2018 CMB (predicted)', PLANCK_GRN, 's'),
    (m_s8cl_13, s_s8cl_13,                    r'SCM Step~13 (this work)',     SCM_ORANGE, 'D'),
    (m_s8cl_12, s_s8cl_12,                    r'SCM Step~12 (prior step)',    '#ff9944',  'd'),
    (PSZ2_S8CL[0], PSZ2_S8CL[1],              r'Planck~SZ PSZ2 (XXIV 2016)', PSZ_BLUE,   'o'),
]
y_pos = list(range(len(surveys_sz)))[::-1]
for i, (cen, err, label, color, mk) in enumerate(surveys_sz):
    y = y_pos[i]
    ax.errorbar(cen, y, xerr=err, fmt=mk, color=color, ms=7,
                capsize=4, lw=1.5, zorder=3)
    ax.text(cen, y + 0.36, label, ha='center', va='bottom',
            fontsize=7.5, color=color)
ax.axvspan(m_s8cl_13 - s_s8cl_13, m_s8cl_13 + s_s8cl_13,
           color=SCM_ORANGE, alpha=0.13)
ax.axvline(m_s8cl_13, color=SCM_ORANGE, lw=1.2, ls='--', alpha=0.7)
ax.annotate(
    rf'vs PSZ2: ${t_psz_s8cl:+.1f}\,\sigma$',
    xy=(PSZ2_S8CL[0], y_pos[-1]), xytext=(0.77, y_pos[-1] - 0.65),
    fontsize=8, color=PSZ_BLUE,
    arrowprops=dict(arrowstyle='->', color=PSZ_BLUE, lw=0.9),
)
ax.annotate(
    rf'vs Planck CMB: ${t_cmb_s8cl:+.1f}\,\sigma$',
    xy=(PLANCK_CMB_S8CL[0], y_pos[0]), xytext=(0.82, y_pos[0] - 0.65),
    fontsize=8, color=PLANCK_GRN,
    arrowprops=dict(arrowstyle='->', color=PLANCK_GRN, lw=0.9),
)
ax.set_xlabel(r'$S_{8,{\rm cl}} = \sigma_8\,(\Omega_{\rm m}/0.3)^{0.3}$')
ax.set_xlim(0.71, 0.92)
ax.set_ylim(-0.9, len(surveys_sz) + 0.9)
ax.set_yticks([])
ax.set_title(r'$S_{8,{\rm cl}}$: cluster-count combination', fontsize=10)
ax.grid(axis='x', alpha=0.28, ls=':')

fig.suptitle(
    r'SCM Step~13: $S_8$ (WL) and $S_{8,{\rm cl}}$ (SZ) constraints',
    fontsize=11.5,
)
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_s8_cluster_step13.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: mass bias posteriors
print("[2/4] fig_mass_bias.pdf ...")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# --- Panel 1: mass_bias_b ---
ax = axes[0]
b_arr = get_col(d13, C13, 'mass_bias_b')
if b_arr is not None:
    xb, yb = kde_1d(b_arr, w13)
    ax.fill_between(xb, yb, alpha=0.35, color=SCM_ORANGE)
    ax.plot(xb, yb, color=SCM_ORANGE, lw=2.0, label='Step~13 posterior')
    m_b, s_b = wstats(b_arr, w13)
    ax.axvline(m_b, color=SCM_ORANGE, lw=1.3, ls='--')
    ax.axvline(m_b - s_b, color=SCM_ORANGE, lw=0.8, ls=':')
    ax.axvline(m_b + s_b, color=SCM_ORANGE, lw=0.8, ls=':')
    ax.set_title(rf'$b = {m_b:.3f} \pm {s_b:.3f}$', fontsize=9.5)
else:
    m_b, s_b = 0.22, 0.06
    ax.text(0.5, 0.5, r'\texttt{mass\_bias\_b} not found in chain',
            transform=ax.transAxes, ha='center', fontsize=9)

# Gaussian prior
xp = np.linspace(0.00, 0.60, 400)
yp = np.exp(-0.5 * ((xp - 0.22) / 0.06)**2) / (0.06 * np.sqrt(2 * np.pi))
ax.plot(xp, yp, 'k--', lw=1.1, alpha=0.65,
        label=r'Prior $\mathcal{N}(0.22,\,0.06^2)$')

# External lensing calibrations
ax.axvline(0.31, color='#6600cc', lw=1.1, ls='-.', alpha=0.80,
           label=r'WtG (von der Linden+14)')   # (1-b)=0.69 → b=0.31
ax.axvline(0.20, color='#228833', lw=1.1, ls='-.', alpha=0.80,
           label=r'X-ray HSE (Planck XV 2013)')  # (1-b)=0.80 → b=0.20

ax.set_xlabel(r'Hydrostatic mass bias $b$  [$M_{\rm true} = (1-b)\,M_{\rm SZ}$]')
ax.set_ylabel(r'Probability density')
ax.set_xlim(0.00, 0.60)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax.grid(alpha=0.22, ls=':')

# --- Panel 2: log_Y_star ---
ax = axes[1]
ly_arr = get_col(d13, C13, 'log_Y_star')
if ly_arr is not None:
    xl, yl = kde_1d(ly_arr, w13)
    ax.fill_between(xl, yl, alpha=0.35, color=SCM_ORANGE)
    ax.plot(xl, yl, color=SCM_ORANGE, lw=2.0, label='Step~13 posterior')
    m_ly, s_ly = wstats(ly_arr, w13)
    ax.axvline(m_ly, color=SCM_ORANGE, lw=1.3, ls='--')
    ax.axvline(m_ly - s_ly, color=SCM_ORANGE, lw=0.8, ls=':')
    ax.axvline(m_ly + s_ly, color=SCM_ORANGE, lw=0.8, ls=':')
    ax.set_title(
        rf'$\log_{{10}}Y_\star = {m_ly:.3f} \pm {s_ly:.3f}$', fontsize=9.5
    )
else:
    m_ly, s_ly = -0.19, 0.02
    ax.text(0.5, 0.5, r'\texttt{log\_Y\_star} not found in chain',
            transform=ax.transAxes, ha='center', fontsize=9)

# X-ray prior (Arnaud+ 2010)
xq = np.linspace(-0.30, -0.08, 400)
yq = np.exp(-0.5 * ((xq + 0.19) / 0.02)**2) / (0.02 * np.sqrt(2 * np.pi))
ax.plot(xq, yq, 'k--', lw=1.1, alpha=0.65,
        label=r'X-ray prior $\mathcal{N}(-0.19,\,0.02^2)$')

ax.set_xlabel(r'$\log_{10} Y_\star$')
ax.set_ylabel(r'Probability density')
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax.grid(alpha=0.22, ls=':')

fig.suptitle(
    r'Planck SZ nuisance parameters: Step~13 posterior vs prior',
    fontsize=11.0,
)
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_mass_bias.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: S8 vs S8_cluster contours
print("[3/4] fig_s8_s8cluster.pdf ...")

fig, ax = plt.subplots(figsize=(6.2, 5.8))

legend_handles = []

# Step 13 contours
try:
    patch13 = contour_2d(s8_13, s8cl_13, w13, ax, SCM_ORANGE,
                          levels=(0.68, 0.95), lw=1.8, alpha_fill=0.22)
    patch13.set_label(r'SCM Step~13 (68\%/95\% CI)')
    legend_handles.append(patch13)
except Exception as e:
    print(f"  Warning: Step 13 contour failed ({e}); using scatter")
    ax.scatter(s8_13[::5], s8cl_13[::5], s=1, color=SCM_ORANGE, alpha=0.2)

# Step 12 contours (lighter, for comparison)
try:
    patch12 = contour_2d(s8_12, s8cl_12, w12, ax, '#ff9944',
                          levels=(0.68, 0.95), lw=1.1, alpha_fill=0.10)
    patch12.set_label(r'SCM Step~12 (68\%/95\% CI)')
    legend_handles.append(patch12)
except Exception as e:
    pass

# Planck SZ constraint: horizontal band on S8_cluster axis
ax.axhspan(PSZ2_S8CL[0] - PSZ2_S8CL[1],
           PSZ2_S8CL[0] + PSZ2_S8CL[1],
           color=PSZ_BLUE, alpha=0.13,
           label=r'Planck SZ PSZ2 $1\sigma$')
ax.axhline(PSZ2_S8CL[0], color=PSZ_BLUE, lw=1.0, ls='--', alpha=0.7)
legend_handles.append(
    mpatches.Patch(color=PSZ_BLUE, alpha=0.30,
                   label=r'Planck~SZ PSZ2 $1\sigma$ (XXIV 2016)')
)

# DES Y1 vertical band on S8 axis
ax.axvspan(DES_Y1_S8[0] - DES_Y1_S8[1],
           DES_Y1_S8[0] + DES_Y1_S8[1],
           color=DES_RED, alpha=0.10,
           label=r'DES~Y1 $1\sigma$')
ax.axvline(DES_Y1_S8[0], color=DES_RED, lw=1.0, ls='--', alpha=0.7)
legend_handles.append(
    mpatches.Patch(color=DES_RED, alpha=0.25,
                   label=r'DES~Y1 shear $1\sigma$')
)

# Diagonal guide: if S8 ≈ S8_cluster the point lies near y=x
# (exact only at Omega_m=0.3; otherwise y/x = (Omega_m/0.3)^{-0.2} ~ constant)
x_diag = np.linspace(0.74, 0.90, 100)
ax.plot(x_diag, x_diag, 'k:', lw=0.7, alpha=0.45,
        label=r'$S_8 = S_{8,{\rm cl}}$ (at $\Omega_m=0.3$)')
legend_handles.append(
    plt.Line2D([0], [0], color='k', ls=':', lw=0.7, alpha=0.6,
               label=r'$S_8 = S_{8,{\rm cl}}$')
)

# Consistency annotation
ax.text(0.04, 0.97,
        rf'Internal: $S_8 - S_{{8,{{\rm cl}}}} = {m_s8_13 - m_s8cl_13:+.3f}$'
        rf' ($\approx{abs(t_wl_sz):.1f}\,\sigma$)',
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

ax.set_xlabel(r'$S_8 = \sigma_8\,(\Omega_{\rm m}/0.3)^{1/2}$  (WL)')
ax.set_ylabel(r'$S_{8,{\rm cl}} = \sigma_8\,(\Omega_{\rm m}/0.3)^{0.3}$  (SZ)')
ax.set_title(r'WL $S_8$ vs cluster $S_{8,{\rm cl}}$: internal consistency', fontsize=10.5)
ax.set_xlim(0.74, 0.90)
ax.set_ylim(0.76, 0.92)
ax.legend(handles=legend_handles, loc='lower right', fontsize=7.5, framealpha=0.9)
ax.grid(alpha=0.22, ls=':')
fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_s8_s8cluster.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# figure: parameter shifts
print("[4/4] fig_param_shift_step13.pdf ...")

SHIFT_PARAMS = [
    ('H0',         r'$H_0$'),
    ('ombh2',      r'$\Omega_b h^2$'),
    ('omch2',      r'$\Omega_c h^2$'),
    ('ns',         r'$n_s$'),
    ('tau',        r'$\tau$'),
    ('logA',       r'$\ln(10^{10}A_s)$'),
    ('eps',        r'$\varepsilon$'),
    ('zc_scm',     r'$z_c$'),
    ('sigma8',     r'$\sigma_8$'),
    ('s8',         r'$S_8$ (WL)'),
    ('s8_cluster', r'$S_{8,{\rm cl}}$'),
]

labels = []
shifts = []

for key, label in SHIFT_PARAMS:
    m13p, s13p = col_wstats(d13, w13, C13, key)
    m12p, s12p = col_wstats(d12, w12, C12, key)

    if key == 's8':
        m13p, s13p = m_s8_13,   s_s8_13
        m12p, s12p = m_s8_12,   s_s8_12
    elif key == 's8_cluster':
        m13p, s13p = m_s8cl_13, s_s8cl_13
        m12p, s12p = m_s8cl_12, s_s8cl_12

    if not (np.isfinite(m13p) and np.isfinite(m12p)):
        continue
    if s13p == 0 and s12p == 0:
        continue

    sigma_comb = np.sqrt(s13p**2 + s12p**2)
    delta = (m13p - m12p) / sigma_comb if sigma_comb > 0 else 0.0
    labels.append(label)
    shifts.append(delta)

fig, ax = plt.subplots(figsize=(8.0, 4.2))

x      = np.arange(len(labels))
colors = [SCM_ORANGE if abs(s) > 0.5 else '#888888' for s in shifts]
ax.bar(x, shifts, color=colors, alpha=0.75, width=0.62, zorder=2)

ax.axhline(0, color='k', lw=0.9)
for level, ls, alpha in [(1, ':', 0.65), (2, '--', 0.40)]:
    ax.axhline(+level, color='gray', lw=0.7, ls=ls, alpha=alpha)
    ax.axhline(-level, color='gray', lw=0.7, ls=ls, alpha=alpha)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(r'$(p_{13} - p_{12})\,/\,\sigma_{\rm comb}$')
ax.set_title(
    r'Parameter shifts Step~13 vs Step~12: effect of adding Planck~SZ clusters',
    fontsize=10.5,
)
ax.set_ylim(-3.2, 3.2)
ax.grid(axis='y', alpha=0.28, ls=':')

for xi, delta in zip(x, shifts):
    ax.text(xi, delta + (0.13 if delta >= 0 else -0.25),
            f'{delta:+.2f}$\\sigma$', ha='center', fontsize=7.2)

fig.tight_layout()
out = os.path.join(_FIG_DIR, 'fig_param_shift_step13.pdf')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# numbers for paper
print()
print("=" * 76)
print("  NUMBERS FOR SCM VI PAPER")
print("=" * 76)

h_val  = col_wstats(d13, w13, C13, 'H0')[0]
om_b   = col_wstats(d13, w13, C13, 'ombh2')[0]
om_c   = col_wstats(d13, w13, C13, 'omch2')[0]
om_m_val = (om_b + om_c) / (h_val / 100.0)**2

print(f"  H0          = {fmt(*col_wstats(d13, w13, C13, 'H0'))} km/s/Mpc")
print(f"  ombh2       = {fmt(*col_wstats(d13, w13, C13, 'ombh2'))}")
print(f"  omch2       = {fmt(*col_wstats(d13, w13, C13, 'omch2'))}")
print(f"  Omega_m     = {om_m_val:.4f}")
print(f"  sigma8      = {fmt(*col_wstats(d13, w13, C13, 'sigma8'))}")
print(f"  S8  (WL)    = {fmt(m_s8_13, s_s8_13)}")
print(f"  S8_cl (SZ)  = {fmt(m_s8cl_13, s_s8cl_13)}")
print(f"  mass_bias b = {fmt(*col_wstats(d13, w13, C13, 'mass_bias_b'))}")
print(f"  log10 Y*    = {fmt(*col_wstats(d13, w13, C13, 'log_Y_star'))}")
print(f"  eps (SCM)   = {fmt(*col_wstats(d13, w13, C13, 'eps'))}")
print(f"  z_c (SCM)   = {fmt(*col_wstats(d13, w13, C13, 'zc_scm'))}")

print()
s8cl_arr = s8cl_13
lo68 = percentile_w(s8cl_arr, w13, 16)
hi68 = percentile_w(s8cl_arr, w13, 84)
lo95 = percentile_w(s8cl_arr, w13,  2.5)
hi95 = percentile_w(s8cl_arr, w13, 97.5)
print(f"  S8_cluster 68% CI : [{lo68:.3f}, {hi68:.3f}]")
print(f"  S8_cluster 95% CI : [{lo95:.3f}, {hi95:.3f}]")

b_arr = get_col(d13, C13, 'mass_bias_b')
if b_arr is not None:
    lo68b = percentile_w(b_arr, w13, 16)
    hi68b = percentile_w(b_arr, w13, 84)
    print(f"  mass_bias_b 68% CI: [{lo68b:.3f}, {hi68b:.3f}]")
    print(f"  1 - b = {1 - col_wstats(d13, w13, C13, 'mass_bias_b')[0]:.3f}  "
          f"(hydrostatic bias factor)")

print()
print(f"  TENSIONS:")
print(f"    S8 (WL) vs DES Y1:           {t_des_s8:+.2f} sigma")
print(f"    S8_cluster vs Planck SZ PSZ2: {t_psz_s8cl:+.2f} sigma")
print(f"    S8_cluster vs Planck CMB:     {t_cmb_s8cl:+.2f} sigma")
print(f"    S8 (WL) vs S8_cluster (int):  {t_wl_sz:+.2f} sigma")
print("=" * 76)

print("\n  All figures saved to:", _FIG_DIR)
