import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood

# Planck PSZ2 data
PSZ2_N_OBS   = np.array([22, 53, 61, 38, 15])
PSZ2_Z_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
PSZ2_SKY_FRAC = 0.65

# Planck SZ scaling relation parameters (Arnaud et al. 2010; Planck 2015 XX/XXIV)
_Q_MIN   = 6.0      # S/N threshold
_ALPHA   = 1.79     # mass–SZ slope
_BETA_SZ = 0.66     # redshift evolution exponent E(z)^beta
_THETA_STAR = 6.997 # arcmin angular scale
_D_STAR_MPC = 500.0 # reference distance in Mpc
_M_REF   = 6.0e14   # reference mass M_sun h_70^{-1} (Planck XXIV Appendix A pivot)

# h-unit cosmology constants
RHO_CRIT_H0 = 2.775e11   # M_sun (h/Mpc)^3 (independent of h)
C_KMS        = 2.998e5    # km/s

# Tinker 2008 Table 2 parameters (Delta_m relative to mean matter density)
_TINKER_DELTA_M = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200],
                            dtype=float)
_TINKER_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_TINKER_a0 = np.array([1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66])
_TINKER_b0 = np.array([2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41])
_TINKER_c0 = np.array([1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44])


# HMF helpers

def _tophat_window(x):
    """Top-hat window W(x) = 3[sin(x) - x cos(x)] / x^3, safe at x→0."""
    out = np.ones_like(x)
    nz  = x > 1e-4
    xn  = x[nz]
    out[nz] = 3.0 * (np.sin(xn) - xn * np.cos(xn)) / xn**3
    return out


def _tinker_interp(delta_m):
    log_d  = np.log(_TINKER_DELTA_M)
    log_dt = float(np.clip(np.log(delta_m), log_d[0], log_d[-1]))
    A0 = float(interp1d(log_d, _TINKER_A0)(log_dt))
    a0 = float(interp1d(log_d, _TINKER_a0)(log_dt))
    b0 = float(interp1d(log_d, _TINKER_b0)(log_dt))
    c0 = float(interp1d(log_d, _TINKER_c0)(log_dt))
    return A0, a0, b0, c0


def _fsigma_tinker(sigma, delta_m, z):
    A0, a0, b0, c0 = _tinker_interp(delta_m)
    A = A0 * (1.0 + z)**(-0.14)
    a = a0 * (1.0 + z)**(-0.06)
    gam = (0.75 / np.log10(max(delta_m / 75.0, 1.01)))**1.2
    b = b0 * (1.0 + z)**(-gam)
    return A * ((sigma / b)**(-a) + 1.0) * np.exp(-c0 / sigma**2)


def _sigma_M(M_arr, kh, Pk, om_m):
    """RMS matter fluctuation σ(M) in top-hat sphere."""
    rho_m_h = RHO_CRIT_H0 * om_m
    sigma = np.zeros(len(M_arr))
    for i, M in enumerate(M_arr):
        R_h = (3.0 * M / (4.0 * np.pi * rho_m_h))**(1.0 / 3.0)
        W   = _tophat_window(kh * R_h)
        s2  = trapezoid(kh**2 * Pk * W**2, kh) / (2.0 * np.pi**2)
        sigma[i] = np.sqrt(max(s2, 0.0))
    return sigma


def _dsigma_dM(M_arr, kh, Pk, om_m, dlnM=0.02):
    s_lo = _sigma_M(M_arr * np.exp(-dlnM), kh, Pk, om_m)
    s_hi = _sigma_M(M_arr * np.exp(+dlnM), kh, Pk, om_m)
    return (s_hi - s_lo) / (M_arr * (np.exp(dlnM) - np.exp(-dlnM)))


def _dndM(M_arr, z, kh, Pk, H0, om_m, Hz, delta_c=500):
    """
    Tinker 2008 dn/dM at Delta=delta_c (critical density).
    Returns h³/Mpc³/M_sun.
    """
    rho_m_h = RHO_CRIT_H0 * om_m
    Ez      = Hz / H0
    om_m_z  = om_m * (1.0 + z)**3 / Ez**2
    delta_m = delta_c / om_m_z   # Δm = Δc × ρ_crit/ρ_mean = Δc / Ω_m(z)

    sig  = _sigma_M(M_arr, kh, Pk, om_m)
    sig  = np.maximum(sig, 1e-10)
    dsig = _dsigma_dM(M_arr, kh, Pk, om_m)
    dlnsigma_inv = np.abs(M_arr * dsig / sig)

    fsig = _fsigma_tinker(sig, delta_m, z)
    return (rho_m_h / M_arr**2) * dlnsigma_inv * fsig


def _get_pk_at_z(provider, z, H0, nk=256):
    """
    Extract linear P(k) at redshift z from Cobaya provider.
    Returns (kh [h/Mpc], Pk [(Mpc/h)^3]).

    Cobaya's Pk_interpolator.P(z,k) expects k in 1/Mpc and returns P in Mpc^3.
    We query at physical k = k_h * h and convert P back to (Mpc/h)^3 via P_h = P_p / h^3.
    """
    h = H0 / 100.0
    PK_interp = provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
                                              extrap_kmax=20.0,
                                              nonlinear=False)
    k_h = np.logspace(-4, np.log10(20.0), nk)   # desired grid in h/Mpc
    k_p = k_h * h                                 # physical k in 1/Mpc
    Pk_p = PK_interp.P(z, k_p)                   # Mpc^3
    Pk_h = Pk_p * h**3                            # (Mpc/h)^3 via P_h = h^3 * P_p
    return k_h, Pk_h


def _effective_M_min(z, Y_star, mass_bias_b, H0, om_m, chi_Mpc, H_z_kms,
                     q_min=_Q_MIN, alpha=_ALPHA):
    """
    Effective mass threshold from Planck SZ selection function.

    M_eff is the mass below which the SZ signal S/N < q_min.
    Uses the simplified scaling: Y_SZ = Y_star (M / (1-b))^alpha E(z)^(2/3) (D_A/D_*)^2
    """
    Ez         = H_z_kms / H0
    D_A_Mpc    = chi_Mpc / (1.0 + z)        # angular diameter distance [Mpc]
    Y_star_lin = 10.0**Y_star
    h70        = H0 / 70.0                   # h in units of h_70

    # Planck 2015 XXIV eq (6): Y_500 * E(z)^{-2/3} * (D_A/D_*)^2 = Y_* * (M/M_ref)^alpha
    # Invert for M_eff given q_min = Y_500 threshold (noise absorbed into Y_*)
    Q_z   = Ez**(2.0 / 3.0) * (D_A_Mpc / _D_STAR_MPC)**2
    M_eff = (_M_REF / h70) * (1.0 - mass_bias_b) * (q_min / (Y_star_lin * Q_z))**(1.0 / alpha)
    return max(M_eff, 1e13)   # floor at 10^13 M_sun


# Cobaya likelihood

class PlanckSZClusters(Likelihood):
    """
    Poisson log-likelihood for Planck PSZ2 cluster number counts.

    Requires from theory provider:
      - Pk_interpolator (linear, delta_tot–delta_tot)
      - comoving_radial_distance
      - Hubble (H(z))
      - ombh2, omch2, H0 (sampled params)

    Nuisance parameters (auto-registered via params class dict):
      mass_bias_b  ~ N(0.22, 0.06)
      log_Y_star   ~ N(-0.19, 0.02)
    """

    # Declare nuisance params so cobaya assigns them to this likelihood
    params = {
        'mass_bias_b': {
            'prior': {'dist': 'norm', 'loc': 0.22, 'scale': 0.06},
            'ref':   {'dist': 'norm', 'loc': 0.22, 'scale': 0.01},
            'proposal': 0.01,
            'latex': r'b_{\rm hyd}',
        },
        'log_Y_star': {
            'prior': {'dist': 'norm', 'loc': -0.19, 'scale': 0.02},
            'ref':   {'dist': 'norm', 'loc': -0.19, 'scale': 0.005},
            'proposal': 0.005,
            'latex': r'\log_{10}Y_\star',
        },
    }

    # Gaussian prior parameters (must match params dict above)
    _PRIOR_MASS_BIAS_MU  = 0.22;  _PRIOR_MASS_BIAS_SIG = 0.06
    _PRIOR_LOG_YSTAR_MU  = -0.19; _PRIOR_LOG_YSTAR_SIG = 0.02

    # Integration resolution
    n_M: int = 32   # mass nodes per z-bin
    n_z: int = 5    # redshift nodes per z-bin

    # Pre-computed integration z points (must be declared in get_requirements)
    # n_z points per z-bin: np.linspace(z_lo+0.02, z_hi-0.02, n_z) for each bin
    _z_int: list = []  # populated in __post_init__

    def initialize(self):
        """Pre-compute the integration z points used across all bins."""
        z_pts = []
        for z_lo, z_hi in zip(PSZ2_Z_EDGES[:-1], PSZ2_Z_EDGES[1:]):
            z_pts.extend(list(np.linspace(z_lo + 0.02, z_hi - 0.02, self.n_z)))
        self._z_int = sorted(set(round(z, 6) for z in z_pts))

    def get_requirements(self):
        zs_needed = self._z_int if self._z_int else list(np.linspace(0.02, 0.98, 25))
        return {
            "Pk_interpolator": {
                "z": zs_needed,
                "k_max": 20.0,
                "nonlinear": False,
                "vars_pairs": [("delta_tot", "delta_tot")],
            },
            "comoving_radial_distance": {"z": zs_needed},
            "Hubble": {"z": zs_needed},
            "ombh2": None,
            "omch2": None,
            "H0": None,
        }

    def logp(self, **params_values):
        b      = params_values.get("mass_bias_b", 0.22)
        log_Ys = params_values.get("log_Y_star",  -0.19)
        ombh2  = self.provider.get_param("ombh2")
        omch2  = self.provider.get_param("omch2")
        H0     = self.provider.get_param("H0")
        h      = H0 / 100.0
        om_m   = (ombh2 + omch2) / h**2

        # Gaussian priors on nuisance parameters
        lp_prior  = -0.5 * ((b      - self._PRIOR_MASS_BIAS_MU) / self._PRIOR_MASS_BIAS_SIG)**2
        lp_prior += -0.5 * ((log_Ys - self._PRIOR_LOG_YSTAR_MU) / self._PRIOR_LOG_YSTAR_SIG)**2

        # Predict N_th per z-bin
        N_th = self._predict_N_bins(b, log_Ys, om_m, H0, h)

        # Guard against negative or zero predictions
        N_th = np.maximum(N_th, 1e-6)

        # Poisson log-likelihood: sum_i [N_obs * ln(N_th) - N_th]
        lp_data = float(np.sum(PSZ2_N_OBS * np.log(N_th) - N_th))

        return lp_data + lp_prior

    def _predict_N_bins(self, b, log_Y_star, om_m, H0, h):
        N_bins = np.zeros(len(PSZ2_N_OBS))

        for i_z, (z_lo, z_hi) in enumerate(zip(PSZ2_Z_EDGES[:-1], PSZ2_Z_EDGES[1:])):
            z_pts = np.linspace(z_lo + 0.02, z_hi - 0.02, self.n_z)
            dN_dz = np.zeros(self.n_z)

            for j, zz in enumerate(z_pts):
                chi_Mpc = float(self.provider.get_comoving_radial_distance(zz))
                Hz_kms  = float(self.provider.get_Hubble(zz))

                chi_h = chi_Mpc * h                        # Mpc/h
                Hz_h  = Hz_kms / h                         # km/s per Mpc/h

                # Comoving volume element (Mpc/h)^3 per unit z (full sky × sky_frac)
                dVdz = 4.0 * np.pi * PSZ2_SKY_FRAC * chi_h**2 * C_KMS / Hz_h

                # Effective mass threshold from SZ selection
                M_eff = _effective_M_min(zz, log_Y_star, b, H0, om_m,
                                         chi_Mpc, Hz_kms)
                M_max  = 5e15
                M_bins_int = np.logspace(np.log10(M_eff), np.log10(M_max), self.n_M)

                # P(k) at this z
                kh, Pk = _get_pk_at_z(self.provider, zz, H0)

                dndM_arr = _dndM(M_bins_int, zz, kh, Pk, H0, om_m, Hz_kms)
                N_M      = trapezoid(dndM_arr, M_bins_int)

                dN_dz[j] = dVdz * N_M

            N_bins[i_z] = trapezoid(dN_dz, z_pts)

        return N_bins
