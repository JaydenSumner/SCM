import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import warnings
import sys

# CAMB import
try:
    import camb
    from camb import model as camb_model
except ImportError:
    sys.exit("ERROR: CAMB not found.  Install with:  pip install camb")

matplotlib.rcParams.update({
    'font.family':  'serif',
    'font.size':    11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi':   120,
})

# cosmological parameters
H0_val    = 67.32           # km/s/Mpc
h         = H0_val / 100.0  # dimensionless Hubble
ombh2     = 0.02237         # Omega_b h^2
omch2     = 0.1200          # Omega_DM h^2
ns        = 0.9649          # scalar spectral index
As        = 2.1e-9          # primordial amplitude (approx)

Omega_b   = ombh2  / h**2
Omega_DM  = omch2  / h**2
Omega_m   = Omega_b + Omega_DM
Omega_L   = 1.0 - Omega_m   # flat universe

rho_DM0   = 2.2416e-27      # kg/m^3  (physical DM density today)
rho_L0    = 5.9233e-27      # kg/m^3  (dark energy density today)

# Sheth-Tormen parameters
delta_c   = 1.686
A_ST      = 0.3222
a_ST      = 0.707
p_ST      = 0.3
M_min     = 1e8             # M_sun (minimum halo mass, fiducial)

print("=" * 65)
print("  SCM II Numerical Computation")
print("=" * 65)
print(f"  Planck 2018:  H0={H0_val}, Omega_m={Omega_m:.4f}, Omega_L={Omega_L:.4f}")
print(f"  M_min = {M_min:.0e} M_sun")

# Hubble function
def E(z):
    """H(z)/H0 for flat LCDM (matter + Lambda, ignoring radiation)."""
    return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

# linear growth factor
def _growth_integrand(zp):
    return (1.0 + zp) / E(zp)**3

_D0_integral, _ = integrate.quad(_growth_integrand, 0.0, 1000.0, limit=500)
_D0 = E(0.0) * _D0_integral  # = 1 * integral since E(0) = 1

def growth_factor(z_arr):
    """Compute D(z) normalised to D(0) = 1 for an array of redshifts."""
    z_arr = np.atleast_1d(z_arr)
    D = np.zeros(len(z_arr))
    for i, z in enumerate(z_arr):
        I, _ = integrate.quad(_growth_integrand, float(z), 1000.0, limit=500)
        D[i] = E(float(z)) * I
    return D / _D0

print("\n  Growth factor check:")
_D_refs = {0: 1.0, 1: 0.6052, 2: 0.4119, 5: 0.2027}
for zc, Dc in zip([0, 1, 2, 5], growth_factor([0, 1, 2, 5])):
    print(f"    D({zc}) = {Dc:.4f}  (SCM-I ref: {_D_refs[zc]})")

# sigma(M_min) from CAMB
def compute_sigma_Mmin():
    """
    Use CAMB to get P(k) at z=0, then compute sigma(R(M_min)).
    Returns sigma(M_min, z=0).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    pars.NonLinear = camb_model.NonLinear_none
    results = camb.get_results(pars)

    # P(k): k in h/Mpc, Pk in (Mpc/h)^3
    kh, _, Pk2d = results.get_matter_power_spectrum(
        minkh=1e-5, maxkh=200.0, npoints=3000)
    Pk = Pk2d[0]  # z=0

    # Convert to physical units: k in 1/Mpc, P in Mpc^3
    k_phys = kh * h
    P_phys = Pk / h**3

    # Comoving mean matter density in M_sun/Mpc^3
    rho_m_comoving = Omega_m * 2.775e11 * h**2  # M_sun/Mpc^3

    # Scale radius R for M_min
    R_Mpc = (3.0 * M_min / (4.0 * np.pi * rho_m_comoving))**(1.0 / 3.0)

    # Top-hat window function W(x) = 3(sin x - x cos x)/x^3
    x = k_phys * R_Mpc
    # Avoid numerical issues at small x
    Wx = np.where(
        np.abs(x) > 1e-4,
        3.0 * (np.sin(x) - x * np.cos(x)) / x**3,
        1.0 - x**2 / 10.0
    )

    integrand = k_phys**2 * P_phys * Wx**2
    sigma2 = np.trapezoid(integrand, k_phys) / (2.0 * np.pi**2)
    return float(np.sqrt(sigma2)), R_Mpc

print("\n  Running CAMB...")
sigma_Mmin_z0, R_Mmin_Mpc = compute_sigma_Mmin()
print(f"  sigma(M_min={M_min:.0e} Msun, z=0) = {sigma_Mmin_z0:.4f}")
print(f"  R(M_min) = {R_Mmin_Mpc:.4f} Mpc")

# Sheth-Tormen f_seq(z)
def f_ST(nu):
    """Sheth-Tormen multiplicity function f_ST(nu)."""
    x = a_ST * nu**2
    return A_ST * np.sqrt(2.0 * a_ST / np.pi) * (1.0 + x**(-p_ST)) * nu * np.exp(-x / 2.0)

def _fseq_from_numin(nu_min):
    """f_seq for a given nu_min (scalar)."""
    if nu_min > 30.0:
        return 0.0
    val, _ = integrate.quad(lambda nu: f_ST(nu) / nu, nu_min, 60.0,
                             limit=300, epsabs=1e-9, epsrel=1e-7)
    return float(val)

# f_seq(z) interpolator
print("\n  Computing f_seq(z) grid...")

z_grid = np.concatenate([
    np.linspace(0.0, 1.0,  30),
    np.linspace(1.0, 4.0,  30),
    np.linspace(4.0, 10.0, 20),
    np.linspace(10.0, 20.0, 10),
])
z_grid = np.unique(z_grid)

D_grid = growth_factor(z_grid)

fseq_grid = np.zeros(len(z_grid))
for i, (z, D) in enumerate(zip(z_grid, D_grid)):
    sigma_z = sigma_Mmin_z0 * D
    if sigma_z < 1e-10:
        fseq_grid[i] = 0.0
        continue
    nu_min = delta_c / sigma_z
    fseq_grid[i] = _fseq_from_numin(nu_min)

# Smooth PCHIP interpolator + analytic derivative
fseq_func  = PchipInterpolator(z_grid, fseq_grid)
dfseq_func = fseq_func.derivative()

fseq0 = float(fseq_func(0.0))
print(f"  f_seq(0)  = {fseq0:.4f}  (SCM-I reference: 0.5764)")

# SCM I
def rho_RE_I(z):
    """SCM I RE: normalised rho_Lambda / rho_Lambda,0."""
    fs  = float(fseq_func(z))
    return fs * (2.0 - fs) / (fseq0 * (2.0 - fseq0))

def rho_FP_I(z):
    """SCM I FP: normalised rho_Lambda / rho_Lambda,0."""
    fs  = float(fseq_func(z))
    return (1.0 - fs)**2 / (1.0 - fseq0)**2

# SCM II corrected field density

# --- FP formulation ---
def rho_FP_II(z):
    """
    SCM II FP: (1+z)^6 * [(1-fseq(z))/(1-fseq0)]^2
    WARNING: diverges rapidly — physical only below a condensation threshold z_c.
    """
    fs = float(fseq_func(z))
    return (1.0 + z)**6 * ((1.0 - fs) / (1.0 - fseq0))**2

# --- RE integral kernels ---
# Case A: energy does NOT dilute after release (vacuum injection)
# Case B: energy dilutes as matter (a^-3) after release
#
# Integrand: (1+z')^power * (1-fseq(z')) * |dfseq/dz'|
# where |dfseq/dz'| = -dfseq/dz' > 0 (fseq decreases with z)
#
# Cumulative from z to infinity:
#   I(z) = integral_z^inf integrand dz'
# Normalised:
#   rho(z) / rho0 = I(z) / I(0)

print("\n  Computing RE integrals (Cases A and B)...")

Z_MAX_INT  = 20.0
N_INT      = 4000
z_int_grid = np.linspace(0.0, Z_MAX_INT, N_INT)

# Evaluate integrand on grid
def _integrand(z_arr, power):
    fs   = fseq_func(z_arr)
    dfs  = dfseq_func(z_arr)        # negative
    return (1.0 + z_arr)**power * (1.0 - fs) * (-dfs)  # positive

kern_A = _integrand(z_int_grid, 6)   # (1+z)^6 weighting
kern_B = _integrand(z_int_grid, 3)   # (1+z)^3 weighting

# Cumulative integral from 0 to z (trapezoidal)
from scipy.integrate import cumulative_trapezoid
cum_A = cumulative_trapezoid(kern_A, z_int_grid, initial=0.0)
cum_B = cumulative_trapezoid(kern_B, z_int_grid, initial=0.0)

total_A = cum_A[-1]
total_B = cum_B[-1]

# I(z) = total - cumulative(z)  →  decreasing, I(0) = total
IA_norm = (total_A - cum_A) / total_A
IB_norm = (total_B - cum_B) / total_B

# Interpolators (PCHIP for smooth derivatives)
IA_func = PchipInterpolator(z_int_grid, IA_norm)
IB_func = PchipInterpolator(z_int_grid, IB_norm)

def rho_RE_A(z):
    """SCM II RE Case A: normalised rho_Lambda/rho_Lambda0."""
    return float(np.clip(IA_func(z), 0.0, None))

def rho_RE_B(z):
    """SCM II RE Case B: normalised rho_Lambda/rho_Lambda0."""
    return float(np.clip(IB_func(z), 0.0, None))

print(f"  Integral totals: I_A = {total_A:.4f},  I_B = {total_B:.4f}")
print(f"  Peak of (1+z)^6 integrand at z ~ {z_int_grid[np.argmax(kern_A)]:.1f}")
print(f"  Peak of (1+z)^3 integrand at z ~ {z_int_grid[np.argmax(kern_B)]:.1f}")

# equation of state
def w_from_rho(rho_func, z_arr, dz=1e-3):
    """
    Compute w(z) using centred finite differences.
    w = -1 - (1/3) * (1+z)/rho * drho/dz
    """
    z_arr = np.asarray(z_arr)
    w = np.full(z_arr.shape, np.nan)
    for i, z in enumerate(z_arr.flat):
        zp = min(z + dz, Z_MAX_INT - dz)
        zm = max(z - dz, 1e-4)
        rho_c = rho_func(z)
        if rho_c < 1e-12:
            continue
        drho_dz = (rho_func(zp) - rho_func(zm)) / (zp - zm)
        w.flat[i] = -1.0 - (1.0 / 3.0) * (1.0 + z) * drho_dz / rho_c
    return w

# CPL fit
def cpl_fit(z_arr, w_arr):
    """Least-squares CPL fit.  Returns (w0, wa)."""
    mask = np.isfinite(w_arr) & (z_arr <= 2.0)
    z_f  = z_arr[mask]
    w_f  = w_arr[mask]
    x    = z_f / (1.0 + z_f)
    A    = np.column_stack([np.ones_like(x), x])
    (w0, wa), _, _, _ = np.linalg.lstsq(A, w_f, rcond=None)
    return float(w0), float(wa)

# DESI constraints
DESI_W0    = -0.827
DESI_WA    = -0.750
DESI_S_W0  =  0.059
DESI_S_WA  =  0.290
DESI_RHO   = -0.93

DESI_COV = np.array([
    [DESI_S_W0**2,                   DESI_RHO * DESI_S_W0 * DESI_S_WA],
    [DESI_RHO * DESI_S_W0 * DESI_S_WA,  DESI_S_WA**2                 ],
])
DESI_COV_INV = np.linalg.inv(DESI_COV)

def mahalanobis(w0, wa):
    d = np.array([w0 - DESI_W0, wa - DESI_WA])
    return float(np.sqrt(d @ DESI_COV_INV @ d))

# compute w(z) and CPL
print("\n  Computing w(z) for all models...")

z_eval  = np.linspace(0.02, 2.5, 500)
z_table = np.array([0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00])

models = {
    "SCM-I RE":   rho_RE_I,
    "SCM-I FP":   rho_FP_I,
    "SCM-II RE-A": rho_RE_A,
    "SCM-II RE-B": rho_RE_B,
}

w_curves = {}
cpl_params = {}
maha_dists = {}

for name, rfunc in models.items():
    w_arr = w_from_rho(rfunc, z_eval)
    w_curves[name] = w_arr
    w0, wa = cpl_fit(z_eval, w_arr)
    cpl_params[name] = (w0, wa)
    maha_dists[name] = mahalanobis(w0, wa)

# LCDM reference
cpl_params["LCDM"]       = (-1.000, 0.000)
maha_dists["LCDM"]       = mahalanobis(-1.0, 0.0)
cpl_params["DESI Pan+"]  = (DESI_W0, DESI_WA)
maha_dists["DESI Pan+"]  = 0.0

# print tables
print("\n" + "=" * 80)
print("  SCM II — Evolution Table")
print("=" * 80)
header = f"{'z':>5}  {'f_seq':>7}  {'rho_RE,I':>10}  {'rho_RE,A':>10}  " \
         f"{'rho_RE,B':>10}  {'w_RE,A':>8}  {'w_RE,B':>8}"
print(header)
print("-" * 80)
for z in z_table:
    fs  = float(fseq_func(z))
    r_I = float(rho_RE_I(z))
    r_A = float(rho_RE_A(z))
    r_B = float(rho_RE_B(z))
    wA  = float(w_from_rho(rho_RE_A, np.array([max(z, 0.02)]))[0])
    wB  = float(w_from_rho(rho_RE_B, np.array([max(z, 0.02)]))[0])
    print(f"{z:>5.2f}  {fs:>7.4f}  {r_I:>10.4f}  {r_A:>10.4f}  "
          f"{r_B:>10.4f}  {wA:>8.4f}  {wB:>8.4f}")
print("=" * 80)

# FP II catastrophe table
print("\n  FP II early dark energy (shows divergence):")
print(f"  {'z':>4}  {'rho_FP,I':>10}  {'rho_FP,II':>12}  {'ratio':>8}")
for z in [0, 0.5, 1, 2, 3, 4]:
    r1 = float(rho_FP_I(z))
    r2 = float(rho_FP_II(z))
    print(f"  {z:>4.1f}  {r1:>10.4f}  {r2:>12.1f}  {r2/r1:>8.1f}x")

print("\n" + "=" * 80)
print("  CPL Fits and DESI 2024 Comparison")
print("=" * 80)

def quadrant(w0, wa):
    if   w0 > -1 and wa < 0: return "Quintessence fading  [target]"
    elif w0 > -1 and wa > 0: return "Quintessence growing [wa wrong]"
    elif w0 < -1 and wa < 0: return "Phantom fading       [w0 wrong]"
    elif w0 < -1 and wa > 0: return "Phantom growing      [both wrong]"
    else:                     return "Boundary"

print(f"  {'Model':<18}  {'w0':>8}  {'wa':>8}  {'dist(sig)':>9}  Regime")
print("  " + "-" * 72)
for name in ["SCM-I RE", "SCM-I FP", "SCM-II RE-A", "SCM-II RE-B",
             "DESI Pan+", "LCDM"]:
    w0, wa = cpl_params[name]
    dist   = maha_dists[name]
    print(f"  {name:<18}  {w0:>8.3f}  {wa:>8.3f}  {dist:>9.2f}  {quadrant(w0, wa)}")
print("=" * 80)

# LaTeX-ready table rows
print("\n  LaTeX rows for SCM-II paper:")
for name in ["SCM-I RE", "SCM-I FP", "SCM-II RE-A", "SCM-II RE-B"]:
    w0, wa = cpl_params[name]
    dist   = maha_dists[name]
    print(f"  {name:<18} & {w0:.3f} & {wa:.3f} & {dist:.1f}$\\sigma$ \\\\")

# figure: four-panel main
print("\n  Generating figures...")

COL = {
    "SCM-I RE":    "#1f77b4",
    "SCM-I FP":    "#ff7f0e",
    "SCM-II RE-A": "#2ca02c",
    "SCM-II RE-B": "#d62728",
    "DESI":        "#9467bd",
    "LCDM":        "black",
}

fig = plt.figure(figsize=(13, 9))
fig.suptitle(
    "SCM II: Concurrent Baryonic Sequestration + Cosmological Dilution",
    fontsize=13, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.33)

# ---- Panel 1: f_seq(z) ----
ax1 = fig.add_subplot(gs[0, 0])
z_long = np.linspace(0, 6, 400)
ax1.plot(z_long, fseq_func(z_long), color=COL["SCM-I RE"], lw=2,
         label="Sheth–Tormen HMF")
ax1.axhline(fseq0, color="gray", ls="--", lw=1,
            label=f"$f_{{\\rm seq,0}} = {fseq0:.4f}$")
ax1.set_xlim(0, 6);  ax1.set_ylim(0, 1)
ax1.set_xlabel("Redshift $z$")
ax1.set_ylabel("$f_{\\rm seq}(z)$")
ax1.set_title("(a)  Sequestration Fraction")
ax1.legend(loc="upper right");  ax1.grid(alpha=0.25)

# ---- Panel 2: DE density evolution ----
ax2 = fig.add_subplot(gs[0, 1])
z_rho = np.linspace(0, 6, 400)
for name, rfunc, ls in [
    ("SCM-I RE",   rho_RE_I,  "--"),
    ("SCM-II RE-A", rho_RE_A, "-"),
    ("SCM-II RE-B", rho_RE_B, "-"),
]:
    vals = np.array([rfunc(z) for z in z_rho])
    ax2.plot(z_rho, vals, color=COL[name], lw=2, ls=ls, label=name)
ax2.axhline(1.0, color=COL["LCDM"], ls=":", lw=1.5, label="$\\Lambda$CDM")
ax2.set_xlim(0, 6);  ax2.set_ylim(-0.05, 1.2)
ax2.set_xlabel("Redshift $z$")
ax2.set_ylabel("$\\rho_{\\Lambda}(z)/\\rho_{\\Lambda,0}$")
ax2.set_title("(b)  Dark Energy Density Evolution")
ax2.legend(loc="lower right");  ax2.grid(alpha=0.25)

# ---- Panel 3: w(z) ----
ax3 = fig.add_subplot(gs[1, 0])
z_w = z_eval[z_eval <= 2.5]
for name, rfunc in [
    ("SCM-I RE",   rho_RE_I),
    ("SCM-I FP",   rho_FP_I),
    ("SCM-II RE-A", rho_RE_A),
    ("SCM-II RE-B", rho_RE_B),
]:
    ls = "--" if "SCM-I" in name else "-"
    w_plot = w_from_rho(rfunc, z_w)
    ax3.plot(z_w, w_plot, color=COL[name], lw=2, ls=ls, label=name)

ax3.axhline(-1.0, color=COL["LCDM"], ls=":", lw=1.5, label="$\\Lambda$CDM")
z_cpl = np.linspace(0, 2.5, 200)
ax3.plot(z_cpl, DESI_W0 + DESI_WA * z_cpl / (1 + z_cpl),
         color=COL["DESI"], lw=1.5, ls="-.", label="DESI Pan+ CPL")
ax3.set_xlim(0, 2.5);  ax3.set_ylim(-1.6, 0.3)
ax3.set_xlabel("Redshift $z$")
ax3.set_ylabel("$w_{\\rm eff}(z)$")
ax3.set_title("(c)  Dark Energy Equation of State")
ax3.legend(loc="lower right");  ax3.grid(alpha=0.25)

# ---- Panel 4: w0–wa plane ----
ax4 = fig.add_subplot(gs[1, 1])

def draw_ellipse(ax, mean, cov, n_std, color, alpha, label):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w_el, h_el = 2 * n_std * np.sqrt(vals)
    e = Ellipse(mean, w_el, h_el, angle=angle,
                facecolor=color, alpha=alpha, edgecolor=color, lw=0.8,
                label=label)
    ax.add_patch(e)

draw_ellipse(ax4, (DESI_W0, DESI_WA), DESI_COV, 1, COL["DESI"], 0.35, "DESI 1sig")
draw_ellipse(ax4, (DESI_W0, DESI_WA), DESI_COV, 2, COL["DESI"], 0.15, "DESI 2sig")

ax4.axhline(0.0,  color="gray", lw=0.7, ls="-")
ax4.axvline(-1.0, color="gray", lw=0.7, ls="-")

markers = {"SCM-I RE": "o", "SCM-I FP": "s",
           "SCM-II RE-A": "^", "SCM-II RE-B": "D", "LCDM": "*"}
for name in ["SCM-I RE", "SCM-I FP", "SCM-II RE-A", "SCM-II RE-B", "LCDM"]:
    w0, wa = cpl_params[name]
    ax4.scatter(w0, wa, color=COL.get(name, "gray"),
                marker=markers.get(name, "o"), s=90, zorder=6,
                label=f"{name} ({w0:.3f}, {wa:.3f})")

ax4.set_xlim(-1.55, -0.40);  ax4.set_ylim(-2.0, 0.8)
ax4.set_xlabel("$w_0$");  ax4.set_ylabel("$w_a$")
ax4.set_title("(d)  $w_0$–$w_a$ Plane")
ax4.legend(fontsize=7.5, loc="lower right");  ax4.grid(alpha=0.25)
ax4.text(-1.20,  0.55, "Quintessence\n(growing)",  fontsize=7, ha="center", color="gray")
ax4.text(-1.20, -1.70, "Phantom",                  fontsize=7, ha="center", color="gray")
ax4.text(-0.70, -1.70, "Quintessence\n(fading) target", fontsize=7, ha="center", color="darkgreen")

fig.savefig("SCM_II_results.pdf", bbox_inches="tight")
fig.savefig("SCM_II_results.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: SCM_II_results.pdf / .png")

# figure: FP II divergence
fig2, ax = plt.subplots(figsize=(7, 4.5))
z_fp = np.linspace(0, 4, 300)

ax.semilogy(z_fp, [rho_FP_I(z)   for z in z_fp],
            color=COL["SCM-I FP"], lw=2, ls="--", label="SCM I FP (comoving)")
ax.semilogy(z_fp, [rho_FP_II(z)  for z in z_fp],
            color=COL["SCM-I FP"], lw=2, ls="-",  label="SCM II FP (physical)")
ax.semilogy(z_fp, [rho_RE_A(z)   for z in z_fp],
            color=COL["SCM-II RE-A"], lw=2, label="SCM II RE Case A (reference)")
ax.axhline(1.0, color=COL["LCDM"], ls=":", lw=1.5, label="$\\Lambda$CDM")
ax.axhspan(2.0, 1e6, alpha=0.08, color="red",
           label="EDE-excluded ($\\rho > 2\\rho_{\\Lambda,0}$)")

ax.set_xlim(0, 4);  ax.set_ylim(0.3, 1e5)
ax.set_xlabel("Redshift $z$")
ax.set_ylabel("$\\rho_{\\Lambda}(z)/\\rho_{\\Lambda,0}$")
ax.set_title("SCM II Field-Pressure: Early Dark Energy Catastrophe")
ax.legend(loc="upper left");  ax.grid(alpha=0.25, which="both")

fig2.tight_layout()
fig2.savefig("SCM_II_FP_divergence.pdf", bbox_inches="tight")
fig2.savefig("SCM_II_FP_divergence.png", bbox_inches="tight", dpi=150)
plt.close(fig2)
print("  Saved: SCM_II_FP_divergence.pdf / .png")

# figure: integrand profiles
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4.5))

z_ki = z_int_grid[z_int_grid <= 12]
ax3a.plot(z_ki, kern_A[z_int_grid <= 12], color=COL["SCM-II RE-A"], lw=2,
          label="Case A: $(1+z)^6(1-f_{\\rm seq})|f^{\\prime}_{\\rm seq}|$")
ax3a.plot(z_ki, kern_B[z_int_grid <= 12], color=COL["SCM-II RE-B"], lw=2,
          label="Case B: $(1+z)^3(1-f_{\\rm seq})|f^{\\prime}_{\\rm seq}|$")
ax3a.set_xlabel("Redshift $z$")
ax3a.set_ylabel("Integrand (arb. units)")
ax3a.set_title("DE Production Rate per Unit Redshift")
ax3a.legend();  ax3a.grid(alpha=0.25);  ax3a.set_xlim(0, 12)

# Cumulative fraction of total integral
total_A_plot = np.trapezoid(kern_A[z_int_grid <= 12],
                        z_int_grid[z_int_grid <= 12])
total_B_plot = np.trapezoid(kern_B[z_int_grid <= 12],
                        z_int_grid[z_int_grid <= 12])
cum_A_plot = np.cumsum(kern_A[z_int_grid <= 12]) * (z_ki[1] - z_ki[0])
cum_B_plot = np.cumsum(kern_B[z_int_grid <= 12]) * (z_ki[1] - z_ki[0])
ax3b.plot(z_ki, cum_A_plot / cum_A_plot[-1], color=COL["SCM-II RE-A"], lw=2,
          label="Case A cumulative fraction")
ax3b.plot(z_ki, cum_B_plot / cum_B_plot[-1], color=COL["SCM-II RE-B"], lw=2,
          label="Case B cumulative fraction")
ax3b.axvline(2.0, color="gray", ls="--", lw=1, label="DESI survey limit $z=2$")
ax3b.set_xlabel("Redshift $z$")
ax3b.set_ylabel("Cumulative fraction of total $\\rho_{\\Lambda,0}$")
ax3b.set_title("Cumulative DE Budget by Redshift")
ax3b.legend();  ax3b.grid(alpha=0.25);  ax3b.set_xlim(0, 12);  ax3b.set_ylim(0, 1)

fig3.suptitle("Where Was the Dark Energy Produced?  (SCM II RE)", y=1.01)
fig3.tight_layout()
fig3.savefig("SCM_II_production.pdf", bbox_inches="tight")
fig3.savefig("SCM_II_production.png", bbox_inches="tight", dpi=150)
plt.close(fig3)
print("  Saved: SCM_II_production.pdf / .png")

# summary
print("\n" + "=" * 65)
print("  Summary")
print("=" * 65)
print(f"  f_seq,0 = {fseq0:.4f}")
for name in ["SCM-I RE", "SCM-I FP", "SCM-II RE-A", "SCM-II RE-B"]:
    w0, wa = cpl_params[name]
    d = maha_dists[name]
    print(f"  {name:<18}  w0={w0:+.3f}  wa={wa:+.3f}  "
          f"Mahal={d:.1f}sig  [{quadrant(w0, wa)[:22]}]")
print()
print("  Key finding (SCM II RE-A):")
w0_A, wa_A = cpl_params["SCM-II RE-A"]
print(f"    The (1+z)^6 weighting concentrates DE production at z ~ "
      f"{z_int_grid[np.argmax(kern_A)]:.1f}.")
pct = float(np.interp(2.0, z_ki, cum_A_plot)) / cum_A_plot[-1]
print(f"    Only {pct*100:.1f}% of total DE budget was produced by z=2")
print(f"    (the DESI observable window).  w(z) is nearly constant")
print(f"    at low z, giving (w0, wa) = ({w0_A:.3f}, {wa_A:.3f}).")
print()
print("  Key finding (SCM II RE-B):")
w0_B, wa_B = cpl_params["SCM-II RE-B"]
pct_B = float(np.interp(2.0, z_ki, cum_B_plot)) / cum_B_plot[-1]
print(f"    (1+z)^3 weighting: {pct_B*100:.1f}% of DE produced by z=2.")
print(f"    (w0, wa) = ({w0_B:.3f}, {wa_B:.3f}).")
print("=" * 65)
print("\nDone.  Output files saved to working directory.")
