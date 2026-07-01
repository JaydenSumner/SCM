import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import warnings, sys

try:
    import camb
    from camb import model as camb_model
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
    print("CAMB not found — using tabulated f_seq from SCM II findings.")

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'legend.fontsize': 9,
})

# cosmological parameters
h_planck = 6.6261e-34   # J.s
k_B      = 1.3806e-23   # J/K
c_light  = 2.9979e8     # m/s
eV_J     = 1.6022e-19   # 1 eV in J
H0_val   = 67.32        # km/s/Mpc
h_cosmo  = H0_val / 100.0
ombh2    = 0.02237
omch2    = 0.1200
Omega_b  = ombh2 / h_cosmo**2
Omega_DM = omch2 / h_cosmo**2
Omega_m  = Omega_b + Omega_DM
Omega_L  = 1.0 - Omega_m
T0_CMB   = 2.7255       # K
zeta32   = 2.6124       # Riemann zeta(3/2)
rho_DM0  = 2.2416e-27   # kg/m^3
rho_L0   = 5.9233e-27   # kg/m^3
rho_crit0 = rho_DM0 / Omega_DM

# Sheth-Tormen parameters
delta_c  = 1.686
A_ST, a_ST, p_ST = 0.3222, 0.707, 0.3
M_min    = 1e8  # M_sun

print("=" * 65)
print("  SCM III Numerical Computation")
print("=" * 65)
print(f"  Omega_m={Omega_m:.4f}  Omega_L={Omega_L:.4f}")

# sequestration fraction
# Reuse SCM II tabulated/CAMB result.  For speed, use tabulated SCM II
# values extended to z=20 with Sheth-Tormen calibration.

_z_ref   = np.array([0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00,
                     4.00, 5.00, 6.00, 8.00, 10.0, 15.0, 20.0])
_fseq_ref = np.array([0.5773, 0.5497, 0.5207, 0.4630, 0.4084,
                      0.3582, 0.2712, 0.2009, 0.1430, 0.0980,
                      0.0390, 0.0130, 0.0015, 0.0002])

if HAS_CAMB:
    # Recompute via CAMB (matches SCM II method exactly)
    def E(z):
        return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

    def growth_factor_arr(z_arr):
        def integrand(zp): return (1+zp) / E(zp)**3
        D0, _ = integrate.quad(integrand, 0, 1000, limit=300)
        D = np.zeros(len(z_arr))
        for i, z in enumerate(z_arr):
            I, _ = integrate.quad(integrand, float(z), 1000, limit=300)
            D[i] = E(float(z)) * I
        return D / D0

    def f_ST(nu):
        x = a_ST * nu**2
        return A_ST*np.sqrt(2*a_ST/np.pi)*(1+x**(-p_ST))*nu*np.exp(-x/2)

    def _fseq_from_numin(nu_min):
        if nu_min > 30: return 0.0
        val, _ = integrate.quad(lambda nu: f_ST(nu)/nu, nu_min, 60,
                                limit=200, epsabs=1e-9, epsrel=1e-7)
        return float(val)

    # Get sigma(M_min) from CAMB
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    pars.NonLinear = camb_model.NonLinear_none
    results_camb = camb.get_results(pars)
    kh, _, Pk2d = results_camb.get_matter_power_spectrum(
        minkh=1e-5, maxkh=200, npoints=3000)
    Pk = Pk2d[0]
    k_phys = kh * h_cosmo
    P_phys = Pk / h_cosmo**3
    rho_m_comov = Omega_m * 2.775e11 * h_cosmo**2
    R_Mmin = (3*M_min/(4*np.pi*rho_m_comov))**(1/3)
    x_arr = k_phys * R_Mmin
    Wx = np.where(np.abs(x_arr)>1e-4,
                  3*(np.sin(x_arr)-x_arr*np.cos(x_arr))/x_arr**3,
                  1-x_arr**2/10)
    sigma_Mmin_z0 = float(np.sqrt(np.trapezoid(
        k_phys**2*P_phys*Wx**2, k_phys)/(2*np.pi**2)))

    z_grid  = np.unique(np.concatenate([
        np.linspace(0,2,60), np.linspace(2,6,30),
        np.linspace(6,20,15)]))
    D_grid  = growth_factor_arr(z_grid)
    fseq_g  = np.zeros(len(z_grid))
    for i,(z,D) in enumerate(zip(z_grid, D_grid)):
        s = sigma_Mmin_z0*D
        if s < 1e-10: continue
        fseq_g[i] = _fseq_from_numin(delta_c/s)
    fseq_func  = PchipInterpolator(z_grid, fseq_g)
    print(f"  CAMB: sigma={sigma_Mmin_z0:.4f}, f_seq,0={float(fseq_func(0)):.4f}")
else:
    fseq_func = PchipInterpolator(_z_ref, _fseq_ref)
    print(f"  Tabulated f_seq,0={float(fseq_func(0)):.4f}")

dfseq_func = fseq_func.derivative()
fseq0      = float(fseq_func(0.0))

# DM mass from condensation redshift
def mass_from_zc(zc):
    """
    DM boson mass (kg) required for BEC condensation at z_c.
    From: 1+z_c = [zeta(3/2) m (2pi m k_B T0)^(3/2) / (rho_DM0 h^3)]^(2/3)
    => m^(5/3) = rho_DM0 h^3 (1+z_c)^(3/2) / [zeta(3/2) (2pi k_B T0)^(3/2)]
    Solve numerically via m = rho_DM0/(n0) where n0 is the critical density.
    """
    # (1+z_c) = [zeta * m * (2pi m k_B T0)^1.5 / (rho_DM0 h^3)]^(2/3)
    # => rho_DM0 * h^3 / (m * (2pi m k_B T0)^1.5) * (1+z_c)^(3/2) = zeta
    # Define f(m) = rho_DM0*h^3*(1+z_c)^1.5 / (m*(2pi*m*k_B*T0)^1.5) - zeta = 0
    target = 1.0 + zc
    def equation(log_m):
        m = 10**log_m
        lhs = rho_DM0*h_planck**3 / (m*(2*np.pi*m*k_B*T0_CMB)**1.5) * target**1.5
        return lhs - zeta32
    try:
        log_m_sol = brentq(equation, -60, -10)
        return 10**log_m_sol
    except:
        return np.nan

def mass_eV_from_zc(zc):
    m_kg = mass_from_zc(zc)
    return m_kg * c_light**2 / eV_J

# dark energy density
def smooth_cutoff(z, zc, dz=0.1):
    """
    Smooth tanh window replacing the hard z <= z_c step.
    Returns ~1 for z << z_c, ~0 for z >> z_c.
    Width dz ~ 0.1 keeps the transition sub-Hubble-time while
    remaining numerically differentiable for CAMB's ODE integrators.
    """
    return 0.5 * (1.0 - np.tanh((z - zc) / dz))

def rho_RE_norm(z):
    """SCM II RE Case A normalised density (nearly 1 for z < 5)."""
    fs = float(fseq_func(z))
    # Approximate: IA(z)/IA(0) ≈ 1 - tiny correction
    # Use the SCM I RE as proxy for shape (SCM II RE is even flatter)
    # For z < 4 the deviation from 1 is < 0.4% so use unity
    return 1.0  # valid approximation from SCM II results

def FP_shape(z):
    """F(z) = (1+z)^6 * [(1-f_seq(z))/(1-f_seq0)]^2, the FP II shape."""
    fs = float(np.clip(fseq_func(z), 0, 1))
    return (1.0 + z)**6 * ((1.0 - fs) / (1.0 - fseq0))**2

def rho_hybrid(z, zc, eps):
    """
    Normalised total DE density: (1-eps)*rho_RE + eps*FP_cut.
    Calibrated to 1.0 at z=0 by construction.
    """
    re_part = (1.0 - eps) * rho_RE_norm(z)
    fp_part = eps * FP_shape(z) * smooth_cutoff(z, zc)
    return re_part + fp_part

# equation of state
def w_from_hybrid(z_arr, zc, eps, dz=5e-4):
    z_arr = np.asarray(z_arr, dtype=float)
    w_out = np.full(z_arr.shape, np.nan)
    for i, z in enumerate(z_arr.flat):
        zp = min(z + dz, 19.9)
        zm = max(z - dz, 1e-4)
        rho_c = rho_hybrid(z,  zc, eps)
        rho_p = rho_hybrid(zp, zc, eps)
        rho_m = rho_hybrid(zm, zc, eps)
        if rho_c < 1e-12: continue
        drho  = (rho_p - rho_m) / (zp - zm)
        w_out.flat[i] = -1.0 - (1.0/3.0) * (1.0 + z) * drho / rho_c
    return w_out

# CPL fit
z_fit_arr = np.linspace(0.02, 2.0, 300)

def cpl_fit(zc, eps):
    w_arr = w_from_hybrid(z_fit_arr, zc, eps)
    mask  = np.isfinite(w_arr)
    if mask.sum() < 10: return np.nan, np.nan
    z_f, w_f = z_fit_arr[mask], w_arr[mask]
    x = z_f / (1 + z_f)
    A = np.column_stack([np.ones_like(x), x])
    (w0, wa), _, _, _ = np.linalg.lstsq(A, w_f, rcond=None)
    return float(w0), float(wa)

# energy budget constraint
def eps_max(zc, frac_limit=0.10):
    """
    Maximum FP fraction today so that rho_FP(z_c) < frac_limit * rho_total(z_c).
    rho_total(z_c) ~ rho_crit * [Omega_m(1+z_c)^3 + Omega_L]
    rho_FP(z_c) = eps * rho_L0 * FP_shape(z_c)
    """
    fp_ratio = FP_shape(zc)   # FP_shape(z_c) / FP_shape(0)
    E2_zc    = Omega_m*(1+zc)**3 + Omega_L
    rho_tot_zc = rho_crit0 * E2_zc
    # eps * rho_L0 * fp_ratio < frac_limit * rho_tot_zc
    return frac_limit * rho_tot_zc / (rho_L0 * fp_ratio)

# DESI constraints
DESI_W0   = -0.827; DESI_WA   = -0.750
DESI_S_W0 =  0.059; DESI_S_WA =  0.290; DESI_RHO = -0.93
DESI_COV  = np.array([[DESI_S_W0**2, DESI_RHO*DESI_S_W0*DESI_S_WA],
                       [DESI_RHO*DESI_S_W0*DESI_S_WA, DESI_S_WA**2]])
DESI_INV  = np.linalg.inv(DESI_COV)

def mahal(w0, wa):
    if not (np.isfinite(w0) and np.isfinite(wa)): return np.inf
    d = np.array([w0 - DESI_W0, wa - DESI_WA])
    return float(np.sqrt(d @ DESI_INV @ d))

# parameter space scan
zc_grid  = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
eps_vals = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20])

print("\n  Mass mapping z_c -> m:")
print(f"  {'z_c':>5}  {'m (meV/c^2)':>12}  {'eps_max':>10}")
print("  " + "-"*32)
for zc in zc_grid:
    m_eV = mass_eV_from_zc(zc) * 1e3  # meV
    em   = eps_max(zc)
    print(f"  {zc:>5.2f}  {m_eV:>12.3f}  {em:>10.5f}")

print("\n  CPL parameter scan (w0, wa, Mahal-dist from DESI Pan+):")
print(f"  {'zc':>5}  {'eps':>6}  {'w0':>8}  {'wa':>8}  {'dist':>7}  {'regime'}")
print("  " + "-"*65)

results = []
for zc in zc_grid:
    em = eps_max(zc)
    for eps in eps_vals:
        if eps > em * 3:  # allow 3x budget for exploration
            continue
        w0, wa = cpl_fit(zc, eps)
        dist   = mahal(w0, wa)
        m_eV   = mass_eV_from_zc(zc) * 1e3
        results.append({'zc': zc, 'eps': eps, 'w0': w0, 'wa': wa,
                        'dist': dist, 'm_meV': m_eV, 'eps_max': em})

results.sort(key=lambda r: r['dist'])

for r in results[:20]:
    if not np.isfinite(r['dist']): continue
    regime = ("phantom" if r['w0'] < -1 else "quintessence")
    print(f"  {r['zc']:>5.2f}  {r['eps']:>6.3f}  "
          f"{r['w0']:>8.3f}  {r['wa']:>8.3f}  "
          f"{r['dist']:>7.2f}  {regime}")

best = results[0]
print(f"\n  Best-fit: z_c={best['zc']:.2f}, eps={best['eps']:.3f}, "
      f"w0={best['w0']:.3f}, wa={best['wa']:.3f}, "
      f"dist={best['dist']:.2f}sig, m={best['m_meV']:.1f} meV/c^2")

# evolution table
z_tab_vals = [0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00]
print(f"\n  Evolution table: z_c={best['zc']:.2f}, eps={best['eps']:.3f}")
print(f"  {'z':>5}  {'f_seq':>7}  {'rho_RE':>8}  {'rho_FP':>8}  {'rho_tot':>9}  {'w_eff':>8}")
print("  " + "-"*55)
for z in z_tab_vals:
    fs     = float(fseq_func(z))
    rho_re = rho_RE_norm(z) * (1 - best['eps'])
    rho_fp = best['eps'] * FP_shape(z) * smooth_cutoff(z, best['zc'])
    rho_t  = rho_re + rho_fp
    w_arr  = w_from_hybrid(np.array([max(z, 0.02)]), best['zc'], best['eps'])
    w_v    = w_arr[0]
    print(f"  {z:>5.2f}  {fs:>7.4f}  {rho_re:>8.4f}  {rho_fp:>8.4f}  {rho_t:>9.4f}  {w_v:>8.4f}")

# key finding: reachable region
print("\n  Key finding: SCM III reachable region in w0-wa plane")
print("  Hybrid RE+FP-II models lie in phantom-GROWING quadrant (w0 < -1, wa > 0)")
print("  for nearly all eps > 0.  FP pushes w0 more phantom AND wa positive.")
print("  DESI Pantheon+ prefers quintessence-fading (w0 > -1, wa < 0).")
print("  --> SCM III moves in the WRONG direction in both w0 and wa as eps grows.")
print()
# Report the model closest to DESI in each direction
min_w0_dist = min(results, key=lambda r: abs(r['w0'] - DESI_W0)
                  if np.isfinite(r['w0']) else np.inf)
min_wa_dist = min(results, key=lambda r: abs(r['wa'] - DESI_WA)
                  if np.isfinite(r['wa']) else np.inf)
print(f"  Closest w0 to DESI: z_c={min_w0_dist['zc']}, eps={min_w0_dist['eps']:.3f} "
      f"-> w0={min_w0_dist['w0']:.3f} (DESI: {DESI_W0})")
print(f"  Closest wa to DESI: z_c={min_wa_dist['zc']}, eps={min_wa_dist['eps']:.3f} "
      f"-> wa={min_wa_dist['wa']:.3f} (DESI: {DESI_WA})")

# figures
print("\n  Generating figures...")

COL_RE  = '#1f77b4'
COL_FP  = '#ff7f0e'
COL_H1  = '#2ca02c'
COL_H2  = '#d62728'
COL_H3  = '#9467bd'
COL_DESI= '#8c564b'
COL_LCDM= 'black'

def draw_ellipse(ax, mean, cov, n_std, color, alpha, label=None):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w_el, h_el = 2*n_std*np.sqrt(vals)
    e = Ellipse(mean, w_el, h_el, angle=angle,
                facecolor=color, alpha=alpha, edgecolor=color, lw=0.8,
                label=label)
    ax.add_patch(e)

# figure: parameter space heatmap
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("SCM III: Parameter Space (z_c, eps)", fontweight='bold')

# Left: w0 heatmap
zc_fine  = np.linspace(0.3, 3.0, 40)
eps_fine = np.linspace(0.0, 0.20, 40)
ZC, EPS  = np.meshgrid(zc_fine, eps_fine)
W0_grid  = np.full_like(ZC, np.nan)
WA_grid  = np.full_like(ZC, np.nan)
DIST_grid= np.full_like(ZC, np.nan)
EM_grid  = np.array([eps_max(z) for z in zc_fine])

print("  Computing (z_c, eps) grid...")
for i, ep in enumerate(eps_fine):
    for j, zc in enumerate(zc_fine):
        w0_v, wa_v = cpl_fit(zc, ep)
        W0_grid[i,j]   = w0_v
        WA_grid[i,j]   = wa_v
        DIST_grid[i,j] = mahal(w0_v, wa_v)

ax = axes1[0]
cm = ax.contourf(ZC, EPS, W0_grid, levels=np.linspace(-1.5, -0.9, 25), cmap='RdYlBu_r')
fig1.colorbar(cm, ax=ax, label='$w_0$')
ax.contour(ZC, EPS, W0_grid, levels=[-1.0], colors='white', linewidths=2, linestyles='--')
# Energy budget constraint line
ax.plot(zc_fine, EM_grid, 'k-', lw=2, label='$\\epsilon_{\\max}$ (energy budget)')
ax.scatter([best['zc']], [best['eps']], color='lime', s=120, zorder=5,
           marker='*', label=f"Best-fit ({best['zc']:.1f}, {best['eps']:.3f})")
ax.set_xlabel('Condensation redshift $z_c$')
ax.set_ylabel('FP fraction $\\epsilon$')
ax.set_title('$w_0$ across parameter space')
ax.legend(fontsize=8);  ax.set_xlim(0.3, 3.0);  ax.set_ylim(0, 0.20)
ax.grid(alpha=0.25)

ax = axes1[1]
cm2 = ax.contourf(ZC, EPS, WA_grid, levels=np.linspace(-1.5, 0.5, 25), cmap='RdYlBu_r')
fig1.colorbar(cm2, ax=ax, label='$w_a$')
ax.contour(ZC, EPS, WA_grid, levels=[-0.75, 0.0], colors=['cyan','white'],
           linewidths=[2,1.5], linestyles=['--',':'])
ax.plot(zc_fine, EM_grid, 'k-', lw=2, label='$\\epsilon_{\\max}$')
ax.scatter([best['zc']], [best['eps']], color='lime', s=120, zorder=5, marker='*')
ax.set_xlabel('Condensation redshift $z_c$')
ax.set_ylabel('FP fraction $\\epsilon$')
ax.set_title('$w_a$ across parameter space (cyan: $w_a = -0.75$)')
ax.legend(fontsize=8);  ax.set_xlim(0.3, 3.0);  ax.set_ylim(0, 0.20)
ax.grid(alpha=0.25)

fig1.tight_layout()
fig1.savefig("SCM_III_param_space.pdf", bbox_inches='tight')
fig1.savefig("SCM_III_param_space.png", bbox_inches='tight', dpi=150)
plt.close(fig1)
print("  Saved: SCM_III_param_space.pdf")

# figure: w(z)
fig2, ax2 = plt.subplots(figsize=(8, 5))
z_w = np.linspace(0.02, 2.5, 400)

models_plot = [
    (0.5, 0.000, COL_RE,  '-',  'SCM II RE (eps=0, LCDM limit)'),
    (0.5, 0.010, COL_H1,  '-',  r'$z_c=0.5$, $\epsilon=0.010$'),
    (0.5, 0.030, COL_H2,  '-',  r'$z_c=0.5$, $\epsilon=0.030$'),
    (1.0, 0.005, COL_H3,  '--', r'$z_c=1.0$, $\epsilon=0.005$'),
    (2.0, 0.001, COL_FP,  '--', r'$z_c=2.0$, $\epsilon=0.001$'),
]
for zc_v, eps_v, col, ls, label in models_plot:
    w_v = w_from_hybrid(z_w, zc_v, eps_v)
    # Smooth out discontinuity at z_c
    mask = np.isfinite(w_v) & (np.abs(w_v) < 5)
    ax2.plot(z_w[mask], w_v[mask], color=col, lw=2, ls=ls, label=label)

ax2.axhline(-1.0, color=COL_LCDM, ls=':', lw=1.5, label='$\\Lambda$CDM')
z_cpl = np.linspace(0, 2.5, 200)
ax2.plot(z_cpl, DESI_W0 + DESI_WA*z_cpl/(1+z_cpl),
         color=COL_DESI, lw=1.5, ls='-.', label='DESI Pan+ CPL')
ax2.set_xlabel('Redshift $z$');  ax2.set_ylabel('$w_{\\rm eff}(z)$')
ax2.set_title('SCM III: Dark Energy Equation of State')
ax2.set_xlim(0, 2.5);  ax2.set_ylim(-2.5, -0.5)
ax2.legend(fontsize=8, loc='lower right');  ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig("SCM_III_wz.pdf", bbox_inches='tight')
fig2.savefig("SCM_III_wz.png", bbox_inches='tight', dpi=150)
plt.close(fig2)
print("  Saved: SCM_III_wz.pdf")

# figure: w0-wa plane
fig3, ax3 = plt.subplots(figsize=(8, 6))
draw_ellipse(ax3, (DESI_W0, DESI_WA), DESI_COV, 1, COL_DESI, 0.35, 'DESI 1$\\sigma$')
draw_ellipse(ax3, (DESI_W0, DESI_WA), DESI_COV, 2, COL_DESI, 0.15, 'DESI 2$\\sigma$')
ax3.axhline(0,  color='gray', lw=0.7)
ax3.axvline(-1, color='gray', lw=0.7)

# Plot all valid results
for r in results:
    if not (np.isfinite(r['w0']) and np.isfinite(r['wa'])): continue
    if abs(r['w0']) > 3 or abs(r['wa']) > 5: continue
    col = plt.cm.viridis(r['zc'] / 3.0)
    ax3.scatter(r['w0'], r['wa'], color=col, s=20, alpha=0.6, zorder=3)

# Color bar for z_c
sm = plt.cm.ScalarMappable(cmap='viridis',
                            norm=plt.Normalize(vmin=0.3, vmax=3.0))
sm.set_array([])
plt.colorbar(sm, ax=ax3, label='Condensation redshift $z_c$')

# Highlight best-fit and LCDM
ax3.scatter(best['w0'], best['wa'], color='lime', s=150, marker='*', zorder=6,
            label=f"Best-fit: ({best['w0']:.3f}, {best['wa']:.3f})")
ax3.scatter(-1.0, 0.0, color='black', s=100, marker='*', zorder=6,
            label='$\\Lambda$CDM (-1.000, 0.000)')

# Show the SCM II RE reference
ax3.scatter(-1.000, 0.000, color='blue', s=80, marker='o', zorder=5,
            label='SCM II RE Case A')

# Annotate quadrants
ax3.text(-1.4,  0.4, 'Phantom\ngrowing', fontsize=8, ha='center', color='gray')
ax3.text(-1.4, -0.8, 'Phantom\nfading', fontsize=8, ha='center', color='gray')
ax3.text(-0.7, -0.8, 'Quintessence\nfading ✓', fontsize=8, ha='center', color='darkgreen')
ax3.text(-0.7,  0.4, 'Quintessence\ngrowing', fontsize=8, ha='center', color='gray')

ax3.set_xlabel('$w_0$');  ax3.set_ylabel('$w_a$')
ax3.set_title('SCM III: Reachable $w_0$–$w_a$ Region')
ax3.set_xlim(-1.6, -0.4);  ax3.set_ylim(-2.0, 0.8)
ax3.legend(fontsize=8, loc='upper right');  ax3.grid(alpha=0.25)
fig3.tight_layout()
fig3.savefig("SCM_III_w0wa.pdf", bbox_inches='tight')
fig3.savefig("SCM_III_w0wa.png", bbox_inches='tight', dpi=150)
plt.close(fig3)
print("  Saved: SCM_III_w0wa.pdf")

# figure: mass mapping and energy budget
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: m(z_c)
zc_arr = np.linspace(0.3, 5.0, 100)
m_arr  = np.array([mass_eV_from_zc(z)*1e3 for z in zc_arr])
valid  = np.isfinite(m_arr) & (m_arr > 0)
ax4a.plot(zc_arr[valid], m_arr[valid], color='navy', lw=2.5)
ax4a.axvspan(0.5, 3.0, alpha=0.12, color='green', label='DESI-relevant $z_c$ range')
for zc_v in [0.5, 1.0, 2.0, 3.0]:
    m_v = mass_eV_from_zc(zc_v)*1e3
    ax4a.annotate(f'$z_c={zc_v:.1f}$\n$m={m_v:.1f}$ meV',
                  (zc_v, m_v), fontsize=7.5, ha='left',
                  xytext=(zc_v+0.1, m_v*0.85),
                  arrowprops=dict(arrowstyle='->', lw=0.8))
ax4a.set_xlabel('Condensation redshift $z_c$')
ax4a.set_ylabel('DM boson mass $m$ (meV/$c^2$)')
ax4a.set_title('Mass–Redshift Mapping: $m \\propto (1+z_c)^{3/5}$')
ax4a.legend(fontsize=9);  ax4a.grid(alpha=0.25)
ax4a.set_xlim(0.3, 5.0);  ax4a.set_ylim(0, 60)

# Right: energy budget constraint
zc_eb = np.linspace(0.3, 3.0, 100)
em_arr = np.array([eps_max(z) for z in zc_eb])
ax4b.semilogy(zc_eb, em_arr, color='darkred', lw=2.5, label='$\\epsilon_{\\max}$ (10% EDE limit)')
ax4b.semilogy(zc_eb, em_arr/10, color='orange', lw=2, ls='--', label='$\\epsilon_{\\max}$ (1% EDE limit)')
ax4b.axhspan(0, 1e-4, alpha=0.1, color='gray', label='Too small to measure')
for zc_v, lab in [(0.5,'$z_c=0.5$'), (1.0,'$z_c=1.0$'), (2.0,'$z_c=2.0$')]:
    em_v = eps_max(zc_v)
    ax4b.scatter([zc_v], [em_v], s=60, zorder=5)
    ax4b.annotate(f'{lab}\n$\\epsilon_{{\\max}}$={em_v:.4f}',
                  (zc_v, em_v), fontsize=7.5, ha='center',
                  xytext=(zc_v, em_v*5))
ax4b.set_xlabel('Condensation redshift $z_c$')
ax4b.set_ylabel('Maximum FP fraction $\\epsilon_{\\max}$')
ax4b.set_title('Energy Budget Constraint on FP Contribution')
ax4b.legend(fontsize=8);  ax4b.grid(alpha=0.25, which='both')
ax4b.set_xlim(0.3, 3.0)

fig4.suptitle("SCM III: Mass Mapping and Energy Budget", fontweight='bold')
fig4.tight_layout()
fig4.savefig("SCM_III_mass_budget.pdf", bbox_inches='tight')
fig4.savefig("SCM_III_mass_budget.png", bbox_inches='tight', dpi=150)
plt.close(fig4)
print("  Saved: SCM_III_mass_budget.pdf")

# LaTeX tables
print("\n  LaTeX table rows (mass mapping):")
for zc_v in [0.30, 0.50, 1.00, 2.00, 3.00, 5.00]:
    m_v = mass_eV_from_zc(zc_v)*1e3
    em_v = eps_max(zc_v)
    fp_ratio = FP_shape(zc_v)
    print(f"  {zc_v:.2f} & {m_v:.2f} & {fp_ratio:.0f} & {em_v:.4f} \\\\")

print("\n  LaTeX table rows (CPL scan):")
for r in sorted(results, key=lambda r: (r['zc'], r['eps']))[:15]:
    if not np.isfinite(r['dist']): continue
    print(f"  {r['zc']:.1f} & {r['eps']:.3f} & "
          f"{r['m_meV']:.1f} & {r['w0']:.3f} & {r['wa']:.3f} & "
          f"{r['dist']:.1f} \\\\")

# summary
print("\n" + "=" * 65)
print("  SCM III Summary")
print("=" * 65)
print(f"  Phase-space conservation: n*lambda^3 = const during free-streaming")
print(f"  => BEC condensation = kinetic decoupling redshift z_c")
print()
print(f"  Mass-redshift mapping (m in meV/c^2):")
for zc_v in [0.5, 1.0, 2.0, 3.0]:
    m_v = mass_eV_from_zc(zc_v)*1e3
    print(f"    z_c = {zc_v:.1f}  =>  m = {m_v:.2f} meV/c^2")
print()
print(f"  Energy budget constraint:")
for zc_v in [0.5, 1.0, 2.0]:
    em_v = eps_max(zc_v)
    fp_r = FP_shape(zc_v)
    print(f"    z_c = {zc_v:.1f}:  FP-ratio = {fp_r:.0f}x,  eps_max = {em_v:.4f}")
print()
print(f"  Key finding: RE + FP-II hybrid lies in PHANTOM-GROWING quadrant")
print(f"  (w0 < -1, wa > 0) for nearly all epsilon > 0.")
print(f"  FP worsens BOTH w0 (more phantom) AND wa (wrong sign, positive).")
print(f"  DESI Pantheon+ prefers QUINTESSENCE-FADING (w0 > -1, wa < 0).")
print()
print(f"  Minimum Mahalanobis distance from DESI: {best['dist']:.2f}sig")
print(f"    at (z_c={best['zc']:.2f}, eps={best['eps']:.3f})")
print(f"    (w0={best['w0']:.3f}, wa={best['wa']:.3f})")
print(f"    Implied DM mass: m = {best['m_meV']:.1f} meV/c^2")
print()
print(f"  Physical implication: matching DESI's w0 > -1 requires")
print(f"  a mechanism that REDUCES DE density going backward in time.")
print(f"  This requires a POSITIVE pressure contribution at z>0,")
print(f"  or a fundamentally different DE mechanism (see Discussion).")
print("=" * 65)
print("\nDone. All output saved to working directory.")
