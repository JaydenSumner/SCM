import os
import sys
import argparse
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

from SCM_camb_de import H0_val, ombh2, omch2, ns_val, As_val, SCMDarkEnergy
from SCM_camb_de_IV import get_scm_camb_results_IV
import camb

# Step 12 posterior mean
H0_12    = 69.09
OMBH2_12 = 0.02236
OMCH2_12 = 0.12013
H_12     = H0_12 / 100.0
OM_M_12  = (OMBH2_12 + OMCH2_12) / H_12**2   # 0.2985
ZC_12    = 2.41
EPS_12   = 0.00122
TAU_12   = 0.051
NS_12    = 0.9649
LOGA_12  = 3.038

# physical constants
# rho_crit,0 = 2.775e11 h^2 M_sun/Mpc^3  →  in h-units: 2.775e11 M_sun (h/Mpc)^3
RHO_CRIT_H0 = 2.775e11    # M_sun (h/Mpc)^3   (independent of h)
C_KMS       = 299792.458  # km/s

# Tinker 2008 Table 2 parameters
_TINKER_DELTA_M = np.array([200,   300,   400,   600,   800,  1200,  1600,  2400,  3200])
_TINKER_A0      = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_TINKER_a0      = np.array([1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66])
_TINKER_b0      = np.array([2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41])
_TINKER_c0      = np.array([1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44])

# Planck PSZ2 cosmological sample (Planck 2015 XXIV, Table 2 / PSZ2 cosmology subset)
# Binned cluster counts: N_obs in z = [0,0.2,0.4,0.6,0.8,1.0]
PSZ2_Z_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
PSZ2_N_OBS   = np.array([22,  53,  61,  38,  15])   # 189 total

# Survey: Planck SZ sky fraction and detection threshold
PLANCK_SKY_FRAC   = 0.65   # effective unmasked sky fraction
PLANCK_M_MIN_H    = 2.0e14 # M_min in M_sun (approximate q>6 effective threshold)

# Matplotlib style matching SCM V
plt.rcParams.update({
    'font.family':     'serif',
    'text.usetex':     True,
    'font.size':       12,
    'axes.labelsize':  13,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top':       True,
    'ytick.right':     True,
})
SCM_ORANGE = '#e87722'
LCDM_BLUE  = '#3366cc'
FIG_DIR    = os.path.join(_HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# CAMB helpers

def _build_camb_results(zc=ZC_12, eps=EPS_12, z_pk=None):
    """
    Run the SCM IV self-consistency loop and return CAMB results.
    For eps=0, returns standard LCDM CAMB results.
    """
    if z_pk is None:
        z_pk = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

    _, w_func, _ = get_scm_camb_results_IV(zc, eps, verbose=False)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0_12, ombh2=OMBH2_12, omch2=OMCH2_12,
        tau=TAU_12, mnu=0.06, num_massive_neutrinos=1,
    )
    pars.InitPower.set_params(ns=NS_12, As=np.exp(LOGA_12) * 1e-10)

    if eps > 0 and w_func is not None:
        de = SCMDarkEnergy.from_w_func(w_func)
        pars.DarkEnergy = de

    pars.set_matter_power(
        redshifts=sorted(set(z_pk), reverse=False),
        kmax=50.0,
        nonlinear=False,
    )
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)
    return results


def _get_pk_at_z(camb_results, z, nk=512):
    """
    Extract linear P(k) at a single redshift from CAMB results.
    Returns (k_h, Pk) with k in h/Mpc, Pk in (Mpc/h)^3.
    """
    kh, zs, Pk_grid = camb_results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=20.0, npoints=nk,
    )
    zs = np.array(zs)
    iz = int(np.argmin(np.abs(zs - z)))
    return np.array(kh), np.array(Pk_grid[iz, :])


# sigma(M)

def _tophat_window(x):
    """Top-hat window function W(x) = 3[sin(x)-x cos(x)]/x^3."""
    W = np.ones_like(x, dtype=float)
    ok = x > 1e-4
    W[ok] = 3.0 * (np.sin(x[ok]) - x[ok]*np.cos(x[ok])) / x[ok]**3
    return W


def sigma_M(M_arr, kh, Pk, om_m):
    """
    RMS linear matter fluctuation in sphere enclosing mass M.

    Parameters
    ----------
    M_arr : array of masses in M_sun
    kh    : k array in h/Mpc  (from CAMB)
    Pk    : P(k) array in (Mpc/h)^3 at the desired redshift
    om_m  : Omega_m

    Returns
    -------
    sigma : dimensionless RMS fluctuation array
    """
    rho_m_h = RHO_CRIT_H0 * om_m   # M_sun (h/Mpc)^3
    Pk = np.maximum(Pk, 0.0)

    sigma = np.zeros(len(M_arr))
    for i, M in enumerate(M_arr):
        R_h = (3.0 * M / (4.0 * np.pi * rho_m_h))**(1.0/3.0)  # Mpc/h
        x   = kh * R_h
        W   = _tophat_window(x)
        integrand = kh**2 * Pk * W**2
        s2 = trapezoid(integrand, kh) / (2.0 * np.pi**2)
        sigma[i] = np.sqrt(max(s2, 0.0))
    return sigma


def dsigma_dM(M_arr, kh, Pk, om_m, dlnM=0.02):
    """Numerical d sigma / d M via central differences in ln M."""
    Mlo = M_arr * np.exp(-dlnM)
    Mhi = M_arr * np.exp(+dlnM)
    s_lo = sigma_M(Mlo, kh, Pk, om_m)
    s_hi = sigma_M(Mhi, kh, Pk, om_m)
    return (s_hi - s_lo) / (Mhi - Mlo)


# Tinker 2008 HMF

def _tinker_interp_params(delta_m):
    """Interpolate Tinker 2008 Table 2 to given Delta_m (relative to mean)."""
    log_d  = np.log(_TINKER_DELTA_M.astype(float))
    log_dt = float(np.log(max(delta_m, _TINKER_DELTA_M[0])))
    log_dt = min(log_dt, np.log(_TINKER_DELTA_M[-1]))

    A0 = float(interp1d(log_d, _TINKER_A0)(log_dt))
    a0 = float(interp1d(log_d, _TINKER_a0)(log_dt))
    b0 = float(interp1d(log_d, _TINKER_b0)(log_dt))
    c0 = float(interp1d(log_d, _TINKER_c0)(log_dt))
    return A0, a0, b0, c0


def tinker_fsigma(sigma, delta_m, z):
    """
    Tinker 2008 multiplicity function f(sigma) at given Delta_m and z.

    delta_m : overdensity relative to mean matter density
    """
    A0, a0, b0, c0 = _tinker_interp_params(delta_m)

    A = A0 * (1.0 + z)**(-0.14)
    a = a0 * (1.0 + z)**(-0.06)
    gamma = (0.75 / np.log10(max(delta_m / 75.0, 1.01)))**1.2
    b = b0 * (1.0 + z)**(-gamma)
    c = c0

    return A * ((sigma / b)**(-a) + 1.0) * np.exp(-c / sigma**2)


def dn_dM(M_arr, z, kh, Pk, camb_results, om_m, h, delta_c=500):
    """
    Tinker 2008 HMF dn/dM at Delta=delta_c (relative to critical density).

    Parameters
    ----------
    kh, Pk : pre-computed k array (h/Mpc) and P(k) ((Mpc/h)^3) at redshift z

    Returns
    -------
    dn_dM : array in h^3/Mpc^3 per M_sun, shape (len(M_arr),)
    """
    rho_m_h = RHO_CRIT_H0 * om_m    # M_sun (h/Mpc)^3

    Ez  = float(camb_results.hubble_parameter(z)) / H0_12

    # Delta_m(z) corresponding to Delta_c at this z: Δm = Δc / Ω_m(z)
    om_m_z  = om_m * (1.0 + z)**3 / Ez**2
    delta_m = delta_c / om_m_z

    sig  = sigma_M(M_arr, kh, Pk, om_m)
    dsig = dsigma_dM(M_arr, kh, Pk, om_m)

    # Guard against zero sigma
    sig = np.maximum(sig, 1e-10)

    fsig = tinker_fsigma(sig, delta_m, z)
    dlnsigma_inv = np.abs(M_arr * dsig / sig)

    # dn/dM = (rho_m / M^2) * |d ln sigma^{-1} / d ln M| * f(sigma)
    return (rho_m_h / M_arr**2) * dlnsigma_inv * fsig


# compute HMF

def compute_hmf_scm_lcdm(zc=ZC_12, eps=EPS_12,
                          z_vals=(0.2, 0.5, 1.0),
                          M_arr=None,
                          verbose=True):
    """
    Compute HMF dn/dM for SCM and LCDM at several redshifts.

    Returns
    -------
    M_arr                      : mass array (M_sun)
    dndM_scm, dndM_lcdm        : dict z → array
    res_scm, res_lcdm          : CAMB results objects
    pk_grids_scm, pk_grids_lcdm: dict z → (kh, Pk) tuples
    """
    if M_arr is None:
        M_arr = np.logspace(13.0, 15.8, 60)

    z_list = sorted(set(list(z_vals) + [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]))

    if verbose:
        print('  [HMF] Building SCM CAMB...')
    res_scm  = _build_camb_results(zc=zc, eps=eps,  z_pk=z_list)
    if verbose:
        print('  [HMF] Building LCDM CAMB (eps=0)...')
    res_lcdm = _build_camb_results(zc=zc, eps=0.0, z_pk=z_list)

    dndM_scm  = {}
    dndM_lcdm = {}
    pk_grids_scm  = {}
    pk_grids_lcdm = {}

    for z in z_vals:
        if verbose:
            print(f'  [HMF] z={z:.1f}...', end=' ', flush=True)
        kh_s, Pk_s = _get_pk_at_z(res_scm,  z)
        kh_l, Pk_l = _get_pk_at_z(res_lcdm, z)
        pk_grids_scm[z]  = (kh_s, Pk_s)
        pk_grids_lcdm[z] = (kh_l, Pk_l)

        dndM_scm[z]  = dn_dM(M_arr, z, kh_s, Pk_s, res_scm,  OM_M_12, H_12)
        dndM_lcdm[z] = dn_dM(M_arr, z, kh_l, Pk_l, res_lcdm, OM_M_12, H_12)
        if verbose:
            ratio = dndM_scm[z] / np.maximum(dndM_lcdm[z], 1e-300)
            idx   = np.argmin(np.abs(M_arr - 3e14))
            print(f'ratio at 3e14 M_sun = {ratio[idx]:.4f}')

    return M_arr, dndM_scm, dndM_lcdm, res_scm, res_lcdm, pk_grids_scm, pk_grids_lcdm


# predicted dN/dz

def predict_dndz(res, om_m, h,
                 z_edges=PSZ2_Z_EDGES,
                 M_min=PLANCK_M_MIN_H,
                 sky_frac=PLANCK_SKY_FRAC,
                 mass_bias=0.22,
                 n_M=40, n_z=6):
    """
    Predict cluster number counts per redshift bin for a Planck SZ-like survey.

    Computes a fresh P(k) grid at each integration redshift via CAMB.
    M_min     : effective mass threshold in M_sun (before mass bias)
    mass_bias : hydrostatic mass bias b; M_eff = M_min / (1-b)
    """
    M_eff  = M_min / (1.0 - mass_bias)
    M_max  = 5e15
    M_bins = np.logspace(np.log10(M_eff), np.log10(M_max), n_M)

    N_bins = np.zeros(len(z_edges) - 1)

    for i_z, (z_lo, z_hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        z_pts       = np.linspace(z_lo + 0.01, z_hi - 0.01, n_z)
        dN_dz_vals  = np.zeros(n_z)

        for j, zz in enumerate(z_pts):
            chi  = float(res.comoving_radial_distance(zz)) * h    # Mpc/h
            Hz   = float(res.hubble_parameter(zz))                  # km/s/Mpc
            Hz_h = Hz / h                                            # km/s/(Mpc/h)
            dVdz = 4.0 * np.pi * sky_frac * chi**2 * C_KMS / Hz_h  # (Mpc/h)^3

            kh, Pk = _get_pk_at_z(res, zz)
            dndM_arr = dn_dM(M_bins, zz, kh, Pk, res, om_m, h)
            N_M = trapezoid(dndM_arr, M_bins)
            dN_dz_vals[j] = dVdz * N_M

        N_bins[i_z] = trapezoid(dN_dz_vals, z_pts)

    return N_bins


# figure: HMF absolute

def fig_hmf(M_arr, dndM_scm, dndM_lcdm, z_vals=(0.2, 0.5, 1.0)):
    """fig_hmf.pdf: HMF absolute value at three redshifts."""
    colors_z = ['#e87722', '#cc4400', '#8800cc']
    M_msun = M_arr  # M_sun

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for col, z in zip(colors_z, z_vals):
        scm  = dndM_scm[z]
        lcdm = dndM_lcdm[z]

        axes[0].loglog(M_msun / 1e14, scm,  '-',  color=col, lw=1.8,
                       label=rf'SCM $z={z:.1f}$')
        axes[0].loglog(M_msun / 1e14, lcdm, '--', color=col, lw=1.2, alpha=0.7,
                       label=rf'$\Lambda$CDM $z={z:.1f}$')

        ratio = scm / np.maximum(lcdm, 1e-300)
        axes[1].semilogx(M_msun / 1e14, ratio, '-', color=col, lw=2.0,
                         label=rf'$z={z:.1f}$')

    axes[0].set_xlabel(r'$M_{500c}\ [10^{14}\,M_\odot]$')
    axes[0].set_ylabel(r'$dn/dM\ [h^3\,\mathrm{Mpc}^{-3}\,(M_\odot/h)^{-1}]$')
    axes[0].set_xlim(0.2, 50)
    axes[0].set_ylim(1e-22, 1e-13)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title(r'Halo mass function: SCM (solid) vs $\Lambda$CDM (dashed)')

    axes[1].axhline(1.0, color='k', lw=0.8, ls='-')
    axes[1].axhline(1.05, color='gray', lw=0.6, ls=':')
    axes[1].axhline(0.95, color='gray', lw=0.6, ls=':')
    axes[1].set_xlabel(r'$M_{500c}\ [10^{14}\,M_\odot]$')
    axes[1].set_ylabel(r'$(dn/dM)_{\rm SCM}\,/\,(dn/dM)_{\Lambda\rm CDM}$')
    axes[1].set_xlim(0.2, 50)
    axes[1].set_ylim(0.97, 1.09)
    axes[1].legend(fontsize=9)
    axes[1].set_title(r'HMF ratio SCM/$\Lambda$CDM')

    for ax in axes:
        ax.grid(alpha=0.25, ls=':')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_hmf.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# figure: dN/dz vs PSZ2

def fig_dndz(N_scm, N_lcdm, z_edges=PSZ2_Z_EDGES, N_obs=PSZ2_N_OBS,
             mass_bias=0.22):
    """fig_dndz.pdf: predicted dN/dz vs Planck PSZ2 observed counts."""
    z_mid  = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz     = z_edges[1:] - z_edges[:-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: absolute counts per bin
    width = 0.35 * dz
    ax = axes[0]
    ax.bar(z_mid - 0.5*width, N_scm,  width, color=SCM_ORANGE, alpha=0.8,
           label=r'SCM (Step~12, $1{-}b=0.78$)')
    ax.bar(z_mid + 0.5*width, N_lcdm, width, color=LCDM_BLUE,  alpha=0.6,
           label=r'$\Lambda$CDM')

    # Observed counts with Poisson error bars
    ax.errorbar(z_mid, N_obs, yerr=np.sqrt(N_obs),
                fmt='ko', ms=6, capsize=4, lw=1.5, zorder=5,
                label=r'Planck PSZ2 (189 clusters)')

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$N_{\rm clusters}$ per bin')
    ax.set_title(r'Cluster counts: SCM vs $\Lambda$CDM vs Planck PSZ2')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.25, ls=':')

    # Right: ratio SCM/LCDM per bin
    ax2 = axes[1]
    ratio = N_scm / np.maximum(N_lcdm, 0.1)
    ax2.bar(z_mid, ratio - 1, dz*0.7, bottom=1.0,
            color=SCM_ORANGE, alpha=0.7, label=r'SCM/$\Lambda$CDM')
    ax2.axhline(1.0, color='k', lw=1.0)
    ax2.axhline(1.05, color='gray', lw=0.7, ls='--', alpha=0.6)
    ax2.set_xlabel(r'Redshift $z$')
    ax2.set_ylabel(r'$N_{\rm SCM}\,/\,N_{\Lambda\rm CDM}$')
    ax2.set_title(r'Predicted count enhancement vs $\Lambda$CDM')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0.97, 1.10)
    ax2.grid(alpha=0.25, ls=':')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_dndz.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# figure: sigma(M)

def fig_sigma_M(M_arr, z_vals, res_scm, res_lcdm, pk_grids_scm, pk_grids_lcdm):
    """fig_sigma_M.pdf: sigma(M) ratio SCM/LCDM."""
    colors_z = ['#e87722', '#cc4400', '#8800cc']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for col, z in zip(colors_z, z_vals):
        kh_s, Pk_s = pk_grids_scm[z]
        kh_l, Pk_l = pk_grids_lcdm[z]
        sig_scm  = sigma_M(M_arr, kh_s, Pk_s, OM_M_12)
        sig_lcdm = sigma_M(M_arr, kh_l, Pk_l, OM_M_12)

        axes[0].semilogx(M_arr / 1e14, sig_scm,  '-',  color=col, lw=1.8,
                         label=rf'SCM $z={z:.1f}$')
        axes[0].semilogx(M_arr / 1e14, sig_lcdm, '--', color=col, lw=1.2, alpha=0.7)

        ratio = sig_scm / np.maximum(sig_lcdm, 1e-10)
        axes[1].semilogx(M_arr / 1e14, ratio, '-', color=col, lw=2.0,
                         label=rf'$z={z:.1f}$')

    axes[0].set_xlabel(r'$M_{500c}\ [10^{14}\,M_\odot]$')
    axes[0].set_ylabel(r'$\sigma(M,z)$')
    axes[0].set_title(r'RMS fluctuation $\sigma(M)$: SCM (solid) vs $\Lambda$CDM (dashed)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, ls=':')

    axes[1].axhline(1.0, color='k', lw=0.8)
    axes[1].set_xlabel(r'$M_{500c}\ [10^{14}\,M_\odot]$')
    axes[1].set_ylabel(r'$\sigma_{\rm SCM}(M)\,/\,\sigma_{\Lambda\rm CDM}(M)$')
    axes[1].set_title(r'$\sigma(M)$ ratio SCM/$\Lambda$CDM')
    axes[1].set_ylim(0.998, 1.012)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25, ls=':')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_sigma_M.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# figure: sigma8-Omega_m

def fig_cluster_s8():
    """fig_cluster_s8.pdf: sigma_8 -- Omega_m plane with constraint bands."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    om_arr = np.linspace(0.20, 0.40, 200)

    # SCM Step 12 posterior (1sigma and 2sigma ellipses approximated as
    # S8 = sigma8 sqrt(Om/0.3) isocontours)
    sigma8_scm = 0.828; s8_scm = 0.826; s8_err = 0.008
    om_scm = OM_M_12

    # S8 = sigma8 sqrt(Om/0.3) isocontours
    for s8_val, lw, ls, alpha in [
        (s8_scm, 2.5, '-',  0.9),
        (s8_scm - s8_err, 1.2, '--', 0.6),
        (s8_scm + s8_err, 1.2, '--', 0.6),
    ]:
        sigma8_iso = s8_val / np.sqrt(om_arr / 0.3)
        ax.plot(om_arr, sigma8_iso, lw=lw, ls=ls, color=SCM_ORANGE, alpha=alpha)

    ax.fill_between(om_arr,
                    (s8_scm - s8_err) / np.sqrt(om_arr/0.3),
                    (s8_scm + s8_err) / np.sqrt(om_arr/0.3),
                    color=SCM_ORANGE, alpha=0.15, label=r'SCM Step~12 $1\sigma$')

    # Planck SZ cluster cosmology (Planck 2015 XXIV with WL calibration)
    # sigma_8 (Om/0.3)^0.3 = 0.782 +/- 0.010 (stat) +/- 0.022 (mass calib)
    # Combined: 0.782 +/- 0.024
    s8_sz = 0.782; s8_sz_err = 0.024
    for s8_val, lw, ls, alpha in [
        (s8_sz, 2.5, '-',  0.9),
        (s8_sz - s8_sz_err, 1.2, '--', 0.6),
        (s8_sz + s8_sz_err, 1.2, '--', 0.6),
    ]:
        sigma8_iso = s8_val / np.sqrt(om_arr / 0.3)
        ax.plot(om_arr, sigma8_iso, lw=lw, ls=ls, color=LCDM_BLUE, alpha=alpha)
    ax.fill_between(om_arr,
                    (s8_sz - s8_sz_err) / np.sqrt(om_arr/0.3),
                    (s8_sz + s8_sz_err) / np.sqrt(om_arr/0.3),
                    color=LCDM_BLUE, alpha=0.15,
                    label=r'Planck PSZ2 (WL calib) $1\sigma$')

    # DES Y1 shear
    s8_des = 0.782; s8_des_err = 0.027
    for s8_val, lw, ls, alpha in [
        (s8_des, 2.0, '-',  0.8),
        (s8_des - s8_des_err, 1.0, '--', 0.5),
        (s8_des + s8_des_err, 1.0, '--', 0.5),
    ]:
        sigma8_iso = s8_val / np.sqrt(om_arr / 0.3)
        ax.plot(om_arr, sigma8_iso, lw=lw, ls=ls, color='#c62026', alpha=alpha)
    ax.fill_between(om_arr,
                    (s8_des - s8_des_err) / np.sqrt(om_arr/0.3),
                    (s8_des + s8_des_err) / np.sqrt(om_arr/0.3),
                    color='#c62026', alpha=0.10,
                    label=r'DES Y1 shear $1\sigma$')

    # Planck CMB
    s8_cmb = 0.832; s8_cmb_err = 0.013
    ax.fill_between(om_arr,
                    (s8_cmb - s8_cmb_err) / np.sqrt(om_arr/0.3),
                    (s8_cmb + s8_cmb_err) / np.sqrt(om_arr/0.3),
                    color='#117733', alpha=0.10,
                    label=r'Planck CMB $1\sigma$')
    ax.plot(om_arr, s8_cmb / np.sqrt(om_arr/0.3), '-', color='#117733', lw=2.0, alpha=0.8)

    # Mark SCM Step 12 best-fit point
    ax.plot(om_scm, sigma8_scm, 'D', color=SCM_ORANGE, ms=9, zorder=5,
            markeredgecolor='k', markeredgewidth=0.7)

    ax.set_xlabel(r'$\Omega_m$')
    ax.set_ylabel(r'$\sigma_8$')
    ax.set_title(r'$\sigma_8$--$\Omega_m$ plane: SCM~VI constraint comparison')
    ax.set_xlim(0.22, 0.38)
    ax.set_ylim(0.72, 0.95)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, ls=':')

    # S8 level labels
    for s8_val, label, color in [
        (s8_scm, rf'$S_8={s8_scm}$', SCM_ORANGE),
        (s8_sz,  rf'$S_8^{{SZ}}={s8_sz}$', LCDM_BLUE),
    ]:
        om_lab = 0.265
        sig_lab = s8_val / np.sqrt(om_lab / 0.3)
        ax.annotate(label, xy=(om_lab, sig_lab), fontsize=8, color=color,
                    xytext=(om_lab + 0.005, sig_lab + 0.015),
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cluster_s8.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# self-consistency test

def run_test():
    """Verify eps=0 gives HMF ratio = 1.000."""
    print('\n  [TEST] eps=0 self-consistency (SCM -> LCDM)...')
    M_test = np.logspace(13.5, 15.5, 20)
    res_a = _build_camb_results(zc=ZC_12, eps=EPS_12, z_pk=[0.0, 0.5])
    res_b = _build_camb_results(zc=ZC_12, eps=0.0,    z_pk=[0.0, 0.5])

    for z in [0.0, 0.5]:
        kh_a, Pk_a = _get_pk_at_z(res_a, z)
        kh_b, Pk_b = _get_pk_at_z(res_b, z)
        dndM_a = dn_dM(M_test, z, kh_a, Pk_a, res_a, OM_M_12, H_12)
        dndM_b = dn_dM(M_test, z, kh_b, Pk_b, res_b, OM_M_12, H_12)
        ratio = dndM_a / np.maximum(dndM_b, 1e-300)
        print(f'    z={z:.1f}: ratio range = [{ratio.min():.4f}, {ratio.max():.4f}]  '
              f'(should be ~1; SCM/LCDM with eps={EPS_12})')

    print('  [TEST] LCDM vs LCDM (eps=0 twice): ratio should be 1.0000...')
    res_c = _build_camb_results(zc=ZC_12, eps=0.0, z_pk=[0.0])
    kh_b0, Pk_b0 = _get_pk_at_z(res_b, 0.0)
    kh_c0, Pk_c0 = _get_pk_at_z(res_c, 0.0)
    dndM_b0 = dn_dM(M_test, 0.0, kh_b0, Pk_b0, res_b, OM_M_12, H_12)
    dndM_c0 = dn_dM(M_test, 0.0, kh_c0, Pk_c0, res_c, OM_M_12, H_12)
    ratio_cc = dndM_b0 / np.maximum(dndM_c0, 1e-300)
    print(f'    LCDM/LCDM ratio range = [{ratio_cc.min():.6f}, {ratio_cc.max():.6f}]')
    print('  [TEST] Done.\n')



def main(test_only=False):
    print()
    print('=' * 70)
    print('  SCM_hmf.py: SCM VI halo mass function prediction figures')
    print('=' * 70)
    print()

    if test_only:
        run_test()
        return

    M_arr  = np.logspace(13.0, 15.8, 50)
    z_vals = (0.2, 0.5, 1.0)

    print('[1/4] Computing HMF for SCM and LCDM...')
    M_arr, dndM_scm, dndM_lcdm, res_scm, res_lcdm, pk_grids_scm, pk_grids_lcdm = \
        compute_hmf_scm_lcdm(zc=ZC_12, eps=EPS_12, z_vals=z_vals, M_arr=M_arr)

    print('[2/4] fig_hmf.pdf ...')
    fig_hmf(M_arr, dndM_scm, dndM_lcdm, z_vals=z_vals)

    print('[3/4] fig_sigma_M.pdf ...')
    fig_sigma_M(M_arr, z_vals, res_scm, res_lcdm, pk_grids_scm, pk_grids_lcdm)

    print('[4a/4] Predicting dN/dz for Planck SZ-like survey...')
    N_scm  = predict_dndz(res_scm,  OM_M_12, H_12)
    N_lcdm = predict_dndz(res_lcdm, OM_M_12, H_12)
    print(f'       N_SCM  = {N_scm}  total={N_scm.sum():.0f}')
    print(f'       N_LCDM = {N_lcdm}  total={N_lcdm.sum():.0f}')
    print(f'       N_obs  = {PSZ2_N_OBS}  total={PSZ2_N_OBS.sum()}')

    print('[4b/4] fig_dndz.pdf ...')
    fig_dndz(N_scm, N_lcdm)

    print('[4c/4] fig_cluster_s8.pdf ...')
    fig_cluster_s8()

    print()
    print(f'  All figures saved to: {FIG_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCM VI HMF prediction module')
    parser.add_argument('--test', action='store_true', help='Run self-consistency test')
    args = parser.parse_args()
    main(test_only=args.test)
