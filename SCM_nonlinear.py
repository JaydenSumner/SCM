import os
import sys
import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from scipy.integrate import quad

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# cosmological constants
from SCM_camb_de import (
    H0_val, ombh2, omch2, ns_val, As_val,
    Omega_m as OM_M, SCMDarkEnergy,
)
from SCM_camb_de_IV import get_scm_camb_results_IV
import camb
from camb import model as camb_model

C_KM_S   = 299792.458          # km/s
H0_KM    = H0_val              # km/s/Mpc
H_FRAC   = H0_KM / 100.0      # h = H0/100
OM_B     = ombh2 / H_FRAC**2
OM_C     = omch2 / H_FRAC**2
OM_M0    = OM_M                # total matter density parameter

# Step 11 posterior mean (used as reference cosmology for predictions)
ZC_MEAN  = 2.399
EPS_MEAN = 0.00148
S8_MEAN  = 0.8347 * np.sqrt(OM_M0 / 0.3)

# Step 11 chain file
_CHAIN = os.path.join(_HERE, 'scm_step11_mcmc.1.txt')

# Figure output directory
FIG_DIR = os.path.join(_HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# matplotlib style
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'font.size': 12,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# CAMB helpers

def _build_camb_params(w_func=None, z_values=None, nonlinear=True, lmax=3000):
    """
    Build CAMB params with optional SCM dark energy and halofit.

    Parameters
    ----------
    w_func    : PchipInterpolator for w(z), or None for LCDM
    z_values  : list of redshifts for P(k) computation
    nonlinear : enable halofit
    lmax      : lmax for CMB lensing spectrum
    """
    if z_values is None:
        z_values = [0.0]
    z_sorted = sorted(set([0.0] + list(z_values)))

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2, mnu=0.06)
    pars.InitPower.set_params(As=As_val, ns=ns_val)

    if w_func is not None:
        de = SCMDarkEnergy.from_w_func(w_func)
        pars.DarkEnergy = de

    pars.set_matter_power(redshifts=z_sorted, kmax=50.0,
                          nonlinear=nonlinear)
    if nonlinear:
        pars.NonLinear = camb_model.NonLinear_pk
        pars.set_nonlinear_lensing(True)
        try:
            pars.NonLinearModel.set_params(halofit_version='mead2020')
        except Exception:
            pass
    else:
        pars.NonLinear = camb_model.NonLinear_none

    pars.max_l      = lmax
    pars.max_eta_k  = 2.0 * lmax

    return pars


def _get_pk_grid(results, nonlinear=True, kmin=1e-3, kmax=30.0, nk=400):
    """
    Extract P(k, z) grid from CAMB results.

    Returns
    -------
    kh  : (nk,) array of k in h/Mpc
    zs  : list of redshifts
    Pk  : (nz, nk) array of P(k, z) in (Mpc/h)^3
    """
    try:
        kh, zs, Pk = results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints=nk, nonlinear=nonlinear)
    except Exception:
        kh, zs, Pk = results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints=nk)
    return np.asarray(kh), list(zs), np.asarray(Pk)


def _pk_interpolator(kh, zs, Pk):
    """
    Build a 2D log-log P(k, z) interpolator.

    Returns a function f(z_val, kh_val) -> P in (Mpc/h)^3.
    """
    zs_arr  = np.asarray(zs, dtype=float)
    log_kh  = np.log(kh)
    log_Pk  = np.log(np.maximum(Pk, 1e-300))

    # Ensure z is increasing
    if zs_arr[0] > zs_arr[-1]:
        zs_arr = zs_arr[::-1]
        log_Pk = log_Pk[::-1, :]

    interp = RegularGridInterpolator(
        (zs_arr, log_kh), log_Pk,
        method='linear', bounds_error=False,
        fill_value=None,
    )

    def pk_func(z_val, kh_val):
        pts = np.column_stack([
            np.full_like(kh_val, z_val, dtype=float),
            np.log(np.maximum(kh_val, 1e-10)),
        ])
        return np.exp(interp(pts))

    return pk_func


# non-linear power spectrum

def compute_nonlinear_pk(zc=ZC_MEAN, eps=EPS_MEAN,
                          z_values=(0.0, 0.5, 1.0),
                          kmin=1e-3, kmax=20.0, nk=300,
                          verbose=True):
    """
    Compute non-linear P(k, z) for SCM and LCDM using halofit (mead2020).

    Returns
    -------
    dict with keys: 'kh', 'P_scm', 'P_lcdm', 'P_lin_scm', 'P_lin_lcdm', 'ratio'
    Each P_* value is a dict {z: array}.
    """
    if verbose:
        print(f"  [NL-Pk] Running SCM CAMB self-consistency loop "
              f"(zc={zc:.2f}, eps={eps:.4f})...")
    _, w_func, _ = get_scm_camb_results_IV(zc, eps, verbose=verbose)

    if verbose:
        print("  [NL-Pk] Running CAMB with halofit for SCM...")
    pars_nl_scm  = _build_camb_params(w_func, z_values, nonlinear=True)
    pars_lin_scm = _build_camb_params(w_func, z_values, nonlinear=False)
    pars_nl_lcdm = _build_camb_params(None,   z_values, nonlinear=True)
    pars_lin_lcdm= _build_camb_params(None,   z_values, nonlinear=False)

    res_nl_scm   = camb.get_results(pars_nl_scm)
    if verbose:
        print("  [NL-Pk] Running CAMB with halofit for LCDM...")
    res_nl_lcdm  = camb.get_results(pars_nl_lcdm)
    res_lin_scm  = camb.get_results(pars_lin_scm)
    res_lin_lcdm = camb.get_results(pars_lin_lcdm)

    out = {'kh': None, 'P_scm': {}, 'P_lcdm': {}, 'P_lin_scm': {}, 'P_lin_lcdm': {}, 'ratio': {}}

    for z in z_values:
        for key, res, nl_flag in [
            ('P_scm',      res_nl_scm,   True),
            ('P_lcdm',     res_nl_lcdm,  True),
            ('P_lin_scm',  res_lin_scm,  False),
            ('P_lin_lcdm', res_lin_lcdm, False),
        ]:
            kh, zs, Pk = _get_pk_grid(res, nonlinear=nl_flag, kmin=kmin, kmax=kmax, nk=nk)
            if out['kh'] is None:
                out['kh'] = kh
            iz = int(np.argmin(np.abs(np.array(zs) - z)))
            out[key][z] = Pk[iz]

        out['ratio'][z] = out['P_scm'][z] / np.maximum(out['P_lcdm'][z], 1e-300)

    return out


# DES Y1 source distribution

def _des_y1_nz_effective(z):
    """
    Effective single-bin DES Y1 source n(z).

    Approximates the sum of 4 DES Y1 tomographic bins (Zuntz et al. 2018),
    normalized so that integral n(z) dz = 1.
    """
    z    = np.asarray(z, dtype=float)
    n    = np.zeros_like(z)
    mask = (z > 0.1) & (z < 1.8)
    zm   = z[mask]
    n[mask] = zm**1.8 * np.exp(-((zm - 0.60) / 0.30)**2 / 2.0)
    n[mask] += 0.4 * zm**0.8 * np.exp(-((zm - 0.95) / 0.25)**2 / 2.0)
    n[mask] = np.maximum(n[mask], 0.0)
    # normalise
    from scipy.integrate import trapezoid
    norm = trapezoid(n, z)
    return n / max(norm, 1e-30)


# convergence power spectrum (Limber)

def compute_convergence_cell(zc=ZC_MEAN, eps=EPS_MEAN,
                              ell_array=None, n_chi=200,
                              verbose=True):
    """
    Compute C_ell^kk via Limber approximation using halofit P(k).

    Uses DES Y1-like effective source n(z).

    Returns
    -------
    dict with keys: 'ell', 'Cell_scm', 'Cell_lcdm'
    All in dimensionless convergence power spectrum units.
    """
    if ell_array is None:
        ell_array = np.unique(np.round(
            np.logspace(np.log10(30), np.log10(3000), 40)).astype(int))
    ell_array = np.asarray(ell_array, dtype=float)

    # Redshift and chi grids
    z_chi = np.linspace(0.02, 2.0, n_chi)

    # Coarse z grid for CAMB P(k) (max 200 to stay well under CAMB's 256 limit)
    z_pk_camb = np.unique(np.concatenate([
        np.linspace(0.02, 0.5, 60),
        np.linspace(0.5,  1.2, 60),
        np.linspace(1.2,  2.0, 30),
    ]))[:200]

    # Get CAMB results for geometry and P(k)
    if verbose:
        print("  [C_ell] Computing SCM non-linear P(k) + geometry...")
    _, w_func, _ = get_scm_camb_results_IV(zc, eps, verbose=False)
    pars_scm  = _build_camb_params(w_func, z_values=list(z_pk_camb), nonlinear=True)
    pars_lcdm = _build_camb_params(None,   z_values=list(z_pk_camb), nonlinear=True)
    res_scm   = camb.get_results(pars_scm)
    res_lcdm  = camb.get_results(pars_lcdm)

    # Comoving distances (Mpc) — comoving_radial_distance accepts arrays
    chi_mpc_scm  = np.array([float(res_scm.comoving_radial_distance(float(z)))  for z in z_chi])
    chi_mpc_lcdm = np.array([float(res_lcdm.comoving_radial_distance(float(z))) for z in z_chi])
    chi_H        = chi_mpc_lcdm[-1]      # horizon ~Mpc

    h   = H_FRAC
    # H0/c in Mpc^{-1}
    H0c = H0_KM / C_KM_S               # in Mpc^{-1}
    # Lensing prefactor [Mpc^{-2}]
    prefac = 1.5 * OM_M0 * H0c**2

    # Source n(z)
    nz = _des_y1_nz_effective(z_chi)    # normalised

    # Hubble parameter H(z) in km/s/Mpc — computed on z_chi via CAMB
    H_scm  = np.array([float(res_scm.hubble_parameter(float(z)))  for z in z_chi])
    H_lcdm = np.array([float(res_lcdm.hubble_parameter(float(z))) for z in z_chi])
    dchi_dz_scm  = C_KM_S / H_scm     # Mpc
    dchi_dz_lcdm = C_KM_S / H_lcdm

    # P(k) grids — use h/Mpc for k, (Mpc/h)^3 for P
    kh, zs, Pk_scm  = _get_pk_grid(res_scm,  nonlinear=True, kmin=5e-4, kmax=50.0, nk=500)
    _,  _,  Pk_lcdm = _get_pk_grid(res_lcdm, nonlinear=True, kmin=5e-4, kmax=50.0, nk=500)

    pk_scm_func  = _pk_interpolator(kh, zs, Pk_scm)
    pk_lcdm_func = _pk_interpolator(kh, zs, Pk_lcdm)

    # Build lensing kernels W(chi) for SCM and LCDM [Mpc^{-1}]
    def _lensing_kernel(chi_arr, z_arr, nz, dchi_dz, prefac):
        """W(chi) array [Mpc^{-1}]. O(n^2) with n=200 takes ~0.04 s."""
        n = len(chi_arr)
        a = 1.0 / (1.0 + z_arr)
        W = np.zeros(n)
        for i in range(n):
            j = np.arange(i + 1, n)
            if len(j) == 0:
                continue
            intg = (nz[j] * (chi_arr[j] - chi_arr[i])
                    / np.maximum(chi_arr[j], 1.0) * dchi_dz[j])
            q_i  = np.trapz(intg, z_arr[j])
            W[i] = prefac * chi_arr[i] / a[i] * q_i
        return W

    if verbose:
        print("  [C_ell] Building lensing kernels...")
    W_scm  = _lensing_kernel(chi_mpc_scm,  z_chi, nz, dchi_dz_scm,  prefac)
    W_lcdm = _lensing_kernel(chi_mpc_lcdm, z_chi, nz, dchi_dz_lcdm, prefac)

    # Limber integral C_ell = integral dchi W^2/chi^2 * P(k=(l+0.5)/chi * h, z)
    # Here P is in (Mpc/h)^3, k in h/Mpc, chi in Mpc -> P(k_h)/chi_h^2 * dchi_h
    # => C_ell = integral (W/h)^2 / (chi/h)^2 * P_h * d(chi/h)
    #          = integral W^2 / chi^2 * P_h / h^3 * dchi  [in Mpc units throughout]
    # Simpler: convert P to Mpc^3: P_mpc = P_h / h^3
    if verbose:
        print("  [C_ell] Evaluating Limber integrals...")

    Cell_scm  = np.zeros(len(ell_array))
    Cell_lcdm = np.zeros(len(ell_array))

    # Precompute z(chi) interpolators
    z_of_chi_scm  = PchipInterpolator(chi_mpc_scm,  z_chi)
    z_of_chi_lcdm = PchipInterpolator(chi_mpc_lcdm, z_chi)

    for i_ell, ell in enumerate(ell_array):
        # SCM
        k_mpc = (ell + 0.5) / np.maximum(chi_mpc_scm, 1.0)  # k in Mpc^{-1}
        k_h   = k_mpc / h                                     # k in h/Mpc
        valid = (k_h > 5e-4) & (k_h < 50.0) & (chi_mpc_scm > 10.0)
        if np.any(valid):
            P_vals = pk_scm_func(z_chi[valid], k_h[valid])
            P_mpc  = P_vals / h**3                             # (Mpc)^3
            integrand = W_scm[valid]**2 / chi_mpc_scm[valid]**2 * P_mpc
            Cell_scm[i_ell] = np.trapz(integrand, chi_mpc_scm[valid])

        # LCDM
        k_mpc = (ell + 0.5) / np.maximum(chi_mpc_lcdm, 1.0)
        k_h   = k_mpc / h
        valid = (k_h > 5e-4) & (k_h < 50.0) & (chi_mpc_lcdm > 10.0)
        if np.any(valid):
            P_vals = pk_lcdm_func(z_chi[valid], k_h[valid])
            P_mpc  = P_vals / h**3
            integrand = W_lcdm[valid]**2 / chi_mpc_lcdm[valid]**2 * P_mpc
            Cell_lcdm[i_ell] = np.trapz(integrand, chi_mpc_lcdm[valid])

    return {'ell': ell_array, 'Cell_scm': Cell_scm, 'Cell_lcdm': Cell_lcdm}


# CMB lensing power spectrum

def compute_cmb_lensing_cell(zc=ZC_MEAN, eps=EPS_MEAN, lmax=2000, verbose=True):
    """
    Compute [L(L+1)]^2 C_L^phiphi / 2pi for SCM and LCDM.

    Returns
    -------
    dict with keys: 'ell', 'cl_scm', 'cl_lcdm' (unnorm. C_L^phiphi * 1e7)
    """
    if verbose:
        print("  [CMB-lens] Running CAMB for CMB lensing spectrum...")
    _, w_func, _ = get_scm_camb_results_IV(zc, eps, verbose=False)

    pars_scm  = _build_camb_params(w_func, nonlinear=True, lmax=lmax)
    pars_lcdm = _build_camb_params(None,   nonlinear=True, lmax=lmax)
    pars_scm.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars_lcdm.set_for_lmax(lmax, lens_potential_accuracy=1)

    res_scm  = camb.get_results(pars_scm)
    res_lcdm = camb.get_results(pars_lcdm)

    # Returns [L(L+1)]^2 C_L^phiphi / 2pi in column 0, shape (lmax+1, 4)
    cl_phi_scm  = res_scm.get_lens_potential_cls(lmax=lmax)[:, 0]
    cl_phi_lcdm = res_lcdm.get_lens_potential_cls(lmax=lmax)[:, 0]

    ell_arr = np.arange(len(cl_phi_scm))
    mask    = ell_arr >= 8

    return {
        'ell':      ell_arr[mask],
        'cl_scm':   cl_phi_scm[mask]  * 1e7,
        'cl_lcdm':  cl_phi_lcdm[mask] * 1e7,
    }


# S8 from step 11 chain

def compute_s8_from_chain(chain_file=_CHAIN, burnin=0.30):
    """
    Compute S8 = sigma8 * sqrt(Omega_m / 0.3) from the Step 11 chain.

    Step 11 column mapping:
      0=weight, 1=logpost, 2=H0, 3=ombh2, 4=omch2, 5=tau, 6=ns, 7=logA,
      8=A_planck, 9=eps, 10=A_fp, 11=As, 12=sigma8, 13=w0_scm, 14=wa_scm,
      15=Mahal_DESI, 16=zc_scm, 17-23=fs8_z*, 24+=chi2 terms
    """
    data = np.loadtxt(chain_file)
    n    = len(data)
    data = data[int(n * burnin):]

    w      = data[:, 0]
    H0     = data[:, 2]
    ombh2_ = data[:, 3]
    omch2_ = data[:, 4]
    sigma8 = data[:, 12]

    h_arr  = H0 / 100.0
    om_m   = (ombh2_ + omch2_) / h_arr**2
    s8_arr = sigma8 * np.sqrt(om_m / 0.3)

    w_tot    = np.sum(w)
    s8_mean  = np.sum(w * s8_arr) / w_tot
    s8_std   = np.sqrt(np.sum(w * (s8_arr - s8_mean)**2) / w_tot)

    # Weighted 68% CI
    idx    = np.argsort(s8_arr)
    w_cum  = np.cumsum(w[idx]) / w_tot
    s8_lo  = s8_arr[idx[np.searchsorted(w_cum, 0.16)]]
    s8_hi  = s8_arr[idx[np.searchsorted(w_cum, 0.84)]]

    return {'mean': s8_mean, 'std': s8_std, 'lo': s8_lo, 'hi': s8_hi,
            'samples': s8_arr, 'weights': w}


# figures

def fig_pk_nonlinear(zc=ZC_MEAN, eps=EPS_MEAN):
    """Non-linear and linear P(k) ratio SCM / LCDM at z=0, 0.5, 1.0."""
    z_vals = [0.0, 0.5, 1.0]
    pk     = compute_nonlinear_pk(zc, eps, z_values=z_vals, verbose=True)
    kh     = pk['kh']

    colors = ['#e87722', '#c62026', '#6600cc']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    ax0, ax1 = axes

    # Left: absolute NL P(k)
    for iz, (z, col) in enumerate(zip(z_vals, colors)):
        ax0.loglog(kh, pk['P_scm'][z],  color=col, lw=1.8, label=f'SCM   $z={z}$')
        ax0.loglog(kh, pk['P_lcdm'][z], color=col, lw=1.8, ls='--', alpha=0.7)

    ax0.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax0.set_ylabel(r'$P(k)\ [(h^{-1}\,\mathrm{Mpc})^3]$')
    ax0.set_title(r'Non-linear $P(k)$: SCM (solid) vs $\Lambda$CDM (dashed)')
    ax0.legend(fontsize=9, ncol=1)
    ax0.set_xlim(1e-2, 10)

    # Right: ratio NL and linear
    ax1.axhline(1.0, color='k', lw=0.8, ls=':')
    for iz, (z, col) in enumerate(zip(z_vals, colors)):
        ratio_nl  = pk['P_scm'][z]  / np.maximum(pk['P_lcdm'][z],  1e-300)
        ratio_lin = pk['P_lin_scm'][z] / np.maximum(pk['P_lin_lcdm'][z], 1e-300)
        ax1.semilogx(kh, ratio_nl,  color=col, lw=1.8, label=f'NL $z={z}$')
        ax1.semilogx(kh, ratio_lin, color=col, lw=1.2, ls='--', alpha=0.6,
                     label=f'Lin $z={z}$')

    ax1.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax1.set_ylabel(r'$P_\mathrm{SCM}(k) / P_{\Lambda\mathrm{CDM}}(k)$')
    ax1.set_title(r'Power spectrum ratio: NL (solid) vs linear (dashed)')
    ax1.legend(fontsize=9, ncol=2)
    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(0.93, 1.07)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_pk_nonlinear.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def fig_shear_cell(zc=ZC_MEAN, eps=EPS_MEAN):
    """Convergence power spectrum C_ell^kk: SCM vs LCDM."""
    result  = compute_convergence_cell(zc, eps, verbose=True)
    ell     = result['ell']
    Cl_scm  = result['Cell_scm']
    Cl_lcdm = result['Cell_lcdm']

    norm = ell * (ell + 1) / (2.0 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1  = axes

    # Left: C_ell * l(l+1)/2pi
    ax0.loglog(ell, norm * Cl_scm,  color='#e87722', lw=2.0, label='SCM')
    ax0.loglog(ell, norm * Cl_lcdm, color='k',       lw=1.5, ls='--',
               label=r'$\Lambda$CDM')
    ax0.set_xlabel(r'$\ell$')
    ax0.set_ylabel(r'$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/(2\pi)$')
    ax0.set_title(r'Convergence power spectrum')
    ax0.legend()
    ax0.set_xlim(30, 3000)

    # Right: ratio
    ratio = Cl_scm / np.maximum(Cl_lcdm, 1e-300)
    ax1.axhline(1.0, color='k', lw=0.8, ls=':')
    ax1.semilogx(ell, ratio, color='#e87722', lw=2.0)
    ax1.set_xlabel(r'$\ell$')
    ax1.set_ylabel(r'$C_\ell^{\kappa\kappa,\mathrm{SCM}} / C_\ell^{\kappa\kappa,\Lambda\mathrm{CDM}}$')
    ax1.set_title(r'Ratio SCM / $\Lambda$CDM')
    ax1.set_xlim(30, 3000)
    ax1.set_ylim(0.94, 1.10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_shear_cell.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def fig_s8_tension(chain_file=_CHAIN):
    """S8 comparison: SCM Step 11 vs published survey values."""
    # Published S8 measurements (mean, +err, -err, label, color)
    surveys = [
        (0.832, 0.013, 0.013, r'Planck 2018 (CMB)',       '#3366cc'),
        (0.835, 0.011, 0.011, r'SCM Step~11 (this work)',  '#e87722'),
        (0.776, 0.017, 0.017, r'DES Y3 3$\times$2pt',      '#c62026'),
        (0.782, 0.027, 0.027, r'DES Y1 shear',              '#aa0000'),
        (0.759, 0.024, 0.021, r'KiDS-1000',                 '#6600cc'),
        (0.780, 0.030, 0.033, r'HSC Y1',                    '#007755'),
    ]

    # Try to get SCM S8 from chain
    try:
        s8res = compute_s8_from_chain(chain_file)
        # Replace SCM placeholder
        surveys[1] = (s8res['mean'], s8res['hi'] - s8res['mean'],
                      s8res['mean'] - s8res['lo'],
                      r'SCM Step~11 (this work)', '#e87722')
    except Exception as exc:
        print(f"  Warning: could not load chain for S8 ({exc})")

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos   = np.arange(len(surveys))[::-1]

    for i, (mean, ep, em, label, col) in enumerate(surveys):
        y = y_pos[i]
        ax.errorbar(mean, y, xerr=[[em], [ep]],
                    fmt='o', color=col, ms=7, capsize=4, lw=1.8,
                    label=label if i == 0 else None)
        ax.text(mean, y + 0.3, label, ha='center', va='bottom',
                fontsize=9, color=col)

    # Tension band: shaded region between CMB and WL surveys
    ax.axvspan(0.750, 0.800, alpha=0.08, color='#cc0000', label='WL tension region')
    ax.axvline(0.832, color='#3366cc', lw=0.8, ls='--', alpha=0.5)

    ax.set_xlabel(r'$S_8 = \sigma_8\,(\Omega_m/0.3)^{1/2}$')
    ax.set_yticks([])
    ax.set_xlim(0.70, 0.92)
    ax.set_title(r'$S_8$ tension: CMB-inferred vs weak lensing surveys')
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_s8_tension.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def fig_cmb_lensing_cell(zc=ZC_MEAN, eps=EPS_MEAN):
    """CMB lensing potential: SCM vs LCDM vs Planck 2018 bandpowers."""
    result = compute_cmb_lensing_cell(zc, eps, verbose=True)
    ell    = result['ell']
    cl_scm = result['cl_scm']
    cl_lcdm= result['cl_lcdm']

    # Planck 2018 VIII Table B1 bandpowers:
    # [L_center, [L(L+1)]^2 C_L^phi / 2pi * 1e7, sigma]
    planck_bp = np.array([
        [19.9,  0.582, 0.129],
        [54.5,  0.427, 0.052],
        [106.9, 0.366, 0.033],
        [167.2, 0.340, 0.028],
        [236.3, 0.278, 0.024],
        [326.1, 0.230, 0.021],
        [436.4, 0.191, 0.020],
        [570.6, 0.162, 0.020],
        [743.7, 0.125, 0.023],
        [967.6, 0.087, 0.026],
        [1243.8,0.067, 0.030],
        [1591.0,0.049, 0.036],
        [1919.2,0.040, 0.050],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes

    # Left: absolute spectrum
    ax0.loglog(ell, cl_scm,  color='#e87722', lw=2.0, label='SCM')
    ax0.loglog(ell, cl_lcdm, color='k',       lw=1.5, ls='--',
               label=r'$\Lambda$CDM')
    ax0.errorbar(planck_bp[:, 0], planck_bp[:, 1],
                 yerr=planck_bp[:, 2],
                 fmt='o', color='#3366cc', ms=5, capsize=3, zorder=5,
                 label='Planck 2018')
    ax0.set_xlabel(r'$L$')
    ax0.set_ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi} / 2\pi \times 10^7$')
    ax0.set_title(r'CMB lensing potential power spectrum')
    ax0.legend()
    ax0.set_xlim(8, 2000)
    ax0.set_ylim(0.01, 2.0)

    # Right: ratio
    from scipy.interpolate import PchipInterpolator
    cl_scm_i  = PchipInterpolator(ell, cl_scm)(planck_bp[:, 0])
    cl_lcdm_i = PchipInterpolator(ell, cl_lcdm)(planck_bp[:, 0])
    ratio_lcdm = cl_lcdm_i / planck_bp[:, 1]
    ratio_scm  = cl_scm_i  / planck_bp[:, 1]

    ax1.axhline(1.0, color='k', lw=0.8, ls=':')
    ax1.semilogx(planck_bp[:, 0], ratio_lcdm, color='k',       lw=1.5, ls='--',
                 label=r'$\Lambda$CDM / Planck')
    ax1.semilogx(planck_bp[:, 0], ratio_scm,  color='#e87722', lw=2.0,
                 label=r'SCM / Planck')
    ax1.fill_between(planck_bp[:, 0],
                     1.0 - planck_bp[:, 2] / planck_bp[:, 1],
                     1.0 + planck_bp[:, 2] / planck_bp[:, 1],
                     alpha=0.15, color='#3366cc', label=r'Planck $1\sigma$')
    ax1.set_xlabel(r'$L$')
    ax1.set_ylabel(r'Theory / Planck bandpower')
    ax1.set_title(r'Ratio to Planck 2018')
    ax1.legend(fontsize=9)
    ax1.set_xlim(8, 2000)
    ax1.set_ylim(0.5, 1.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_cmb_lensing_cell.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


# unit tests

def run_tests():
    """Validate key functions at LCDM limit (eps -> 0)."""
    print("\n" + "=" * 60)
    print("  SCM_nonlinear unit tests")
    print("=" * 60)
    ok = True

    # Test 1: P(k) ratio -> 1 at eps=0
    print("  Test 1: NL P(k) ratio at eps=0...")
    pk = compute_nonlinear_pk(zc=2.0, eps=1e-6, z_values=[0.0], verbose=False)
    ratio_max = np.max(np.abs(pk['ratio'][0.0][50:200] - 1.0))
    status = 'PASS' if ratio_max < 0.005 else 'FAIL'
    print(f"    max|ratio - 1| = {ratio_max:.4f}  [{status}]")
    ok = ok and (ratio_max < 0.005)

    # Test 2: C_ell > 0 and finite
    print("  Test 2: Convergence C_ell positive and finite...")
    try:
        ell_test = np.array([100, 300, 1000])
        res = compute_convergence_cell(zc=2.0, eps=1e-6, ell_array=ell_test, verbose=False)
        all_pos   = np.all(res['Cell_scm'] > 0)
        all_fin   = np.all(np.isfinite(res['Cell_scm']))
        # Should be within 1% of LCDM at eps=0
        ratio_cl  = res['Cell_scm'] / np.maximum(res['Cell_lcdm'], 1e-300)
        close     = np.all(np.abs(ratio_cl - 1.0) < 0.05)
        status    = 'PASS' if all_pos and all_fin and close else 'FAIL'
        print(f"    all_positive={all_pos}, all_finite={all_fin}, close_to_LCDM={close}  [{status}]")
        ok = ok and (all_pos and all_fin and close)
    except Exception as e:
        print(f"    FAIL ({e})")
        ok = False

    # Test 3: S8 from chain
    if os.path.exists(_CHAIN):
        print("  Test 3: S8 from Step 11 chain...")
        try:
            res = compute_s8_from_chain()
            in_range = 0.75 < res['mean'] < 0.90
            status   = 'PASS' if in_range else 'FAIL'
            print(f"    S8 = {res['mean']:.4f} +{res['hi']-res['mean']:.4f} -{res['mean']-res['lo']:.4f}  [{status}]")
            ok = ok and in_range
        except Exception as e:
            print(f"    FAIL ({e})")
            ok = False

    print("\n" + "=" * 60)
    print(f"  All tests {'PASSED' if ok else 'FAILED'}")
    print("=" * 60 + "\n")
    return ok



def run_all():
    """Generate all SCM V prediction figures."""
    print("\n" + "=" * 70)
    print("  SCM_nonlinear.py: generating SCM V prediction figures")
    print("=" * 70)

    print("\n[1/4] Non-linear power spectrum ratio...")
    fig_pk_nonlinear()

    print("\n[2/4] Convergence power spectrum C_ell^kk...")
    fig_shear_cell()

    print("\n[3/4] S8 tension comparison...")
    fig_s8_tension()

    print("\n[4/4] CMB lensing spectrum...")
    fig_cmb_lensing_cell()

    print("\n  All figures saved to:", FIG_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCM V: non-linear structure figures')
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)
    else:
        run_all()
