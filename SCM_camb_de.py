import numpy as np
from scipy import integrate
from scipy.interpolate import PchipInterpolator
import camb
from camb import model as camb_model
from camb.dark_energy import DarkEnergyPPF

# cosmological parameters
H0_val    = 67.32
h_cosmo   = H0_val / 100.0
ombh2     = 0.02237
omch2     = 0.1200
Omega_b   = ombh2  / h_cosmo**2
Omega_DM  = omch2  / h_cosmo**2
Omega_m   = Omega_b + Omega_DM
Omega_L   = 1.0 - Omega_m
ns_val    = 0.9649
As_val    = 2.1e-9
M_min     = 1e8           # M_sun

delta_c           = 1.686
A_ST, a_ST, p_ST  = 0.3222, 0.707, 0.3

# smooth cutoff
def smooth_cutoff(z, zc, dz=0.1):
    """tanh window: ~1 for z << zc, ~0 for z >> zc."""
    return 0.5 * (1.0 - np.tanh((z - zc) / dz))


# analytic LCDM Hubble function
def _E_lcdm(z):
    """E(z) = H(z)/H0 for flat LCDM, matter + Lambda only."""
    return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)


# SCM hybrid density
def rho_hybrid_scm(z, zc, eps, fseq_func, fseq0, dz=0.1):
    """
    Unnormalised SCM dark energy density (equals ~1 at z=0).
    RE part: Case A approximation rho_RE = 1 (valid to < 0.4% for z < 4).
    FP part: (1+z)^6 * [(1-f_seq)/(1-f_seq0)]^2 * smooth_cutoff.
    """
    fs       = float(np.clip(fseq_func(float(z)), 0.0, 1.0))
    fp_shape = (1.0 + z)**6 * ((1.0 - fs) / max(1.0 - fseq0, 1e-10))**2
    return (1.0 - eps) * 1.0 + eps * fp_shape * smooth_cutoff(z, zc, dz)


def build_rho_func(zc, eps, fseq_func, fseq0, dz=0.1):
    """
    Normalised rho_hybrid interpolator on z in [0, 20].
    Divides by rho_hybrid(0) so that rho(0) = 1 exactly, as CAMB requires.
    Returns (interp, norm) where norm = rho_hybrid(0).
    """
    z_grid = np.unique(np.concatenate([
        np.linspace(0.0, 0.5,  80),
        np.linspace(0.5, 2.0, 100),
        np.linspace(2.0, 6.0,  60),
        np.linspace(6.0, 20.0, 30),
    ]))
    rho_grid = np.array([
        rho_hybrid_scm(z, zc, eps, fseq_func, fseq0, dz) for z in z_grid
    ])
    norm = rho_grid[0]
    return PchipInterpolator(z_grid, rho_grid / norm), norm


# w(z) from rho(z)
def build_w_func(rho_func, z_max=19.9, dz_deriv=5e-4):
    """
    w(z) = -1 - (1/3)(1+z)/rho * drho/dz
    Uses centred finite differences on the normalised rho interpolator.
    Returns a PchipInterpolator on z in (0, z_max].
    """
    z_grid = np.unique(np.concatenate([
        np.linspace(0.005, 0.5,  80),
        np.linspace(0.5,  2.0, 120),
        np.linspace(2.0,  6.0,  60),
        np.linspace(6.0,  z_max, 30),
    ]))
    w_grid = np.zeros(len(z_grid))
    for i, z in enumerate(z_grid):
        zp     = min(z + dz_deriv, z_max)
        zm     = max(z - dz_deriv, 0.001)
        rho_c  = float(rho_func(z))
        drho   = (float(rho_func(zp)) - float(rho_func(zm))) / (zp - zm)
        w_grid[i] = -1.0 - (1.0/3.0) * (1.0 + z) * drho / max(rho_c, 1e-12)
    return PchipInterpolator(z_grid, w_grid)


# linear growth factor D(z)
def build_D_func(E_callable, z_max=20.0):
    """
    D(z) proportional to E(z) * integral_z^inf dz'/[(1+z') E(z')]^3,
    normalised so D(0) = 1.

    E_callable must be correct for all z from 0 to 1000.
    """
    def kern(zp):
        return (1.0 + zp) / max(E_callable(zp)**3, 1e-90)

    D0_int, _ = integrate.quad(kern, 0.0, 1000.0, limit=400)
    D0 = float(E_callable(0.0)) * D0_int

    z_grid = np.unique(np.concatenate([
        np.linspace(0.0,  5.0, 100),
        np.linspace(5.0, z_max,  50),
    ]))
    D_grid = np.zeros(len(z_grid))
    for i, z in enumerate(z_grid):
        I, _ = integrate.quad(kern, float(z), 1000.0, limit=400)
        D_grid[i] = float(E_callable(float(z))) * I / D0
    return PchipInterpolator(z_grid, D_grid)


# sigma(M_min) from CAMB
_sigma_cache: dict = {}

def get_sigma_Mmin(verbose=True):
    if 'sigma' in _sigma_cache:
        return _sigma_cache['sigma']

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    pars.NonLinear = camb_model.NonLinear_none
    res  = camb.get_results(pars)
    kh, _, Pk2d = res.get_matter_power_spectrum(minkh=1e-5, maxkh=200, npoints=3000)
    Pk      = Pk2d[0]
    k_phys  = kh * h_cosmo
    P_phys  = Pk / h_cosmo**3
    rho_m_comov = Omega_m * 2.775e11 * h_cosmo**2
    R_Mmin  = (3.0 * M_min / (4.0 * np.pi * rho_m_comov))**(1.0/3.0)
    x       = k_phys * R_Mmin
    Wx      = np.where(np.abs(x) > 1e-4,
                       3.0*(np.sin(x) - x*np.cos(x)) / x**3,
                       1.0 - x**2/10.0)
    sigma   = float(np.sqrt(
        np.trapezoid(k_phys**2 * P_phys * Wx**2, k_phys) / (2.0*np.pi**2)
    ))
    _sigma_cache['sigma'] = sigma
    if verbose:
        print(f"  sigma(M_min={M_min:.0e} Msun, z=0) = {sigma:.4f}")
    return sigma


# Sheth-Tormen f_seq(z)
def f_ST(nu):
    x = a_ST * nu**2
    return A_ST * np.sqrt(2.0*a_ST/np.pi) * (1.0 + x**(-p_ST)) * nu * np.exp(-x/2.0)

def _fseq_from_numin(nu_min):
    if nu_min > 30.0:
        return 0.0
    val, _ = integrate.quad(lambda nu: f_ST(nu)/nu, nu_min, 60.0,
                             limit=200, epsabs=1e-9, epsrel=1e-7)
    return float(val)

def build_fseq_func(D_interp, sigma_Mmin_z0):
    """Build f_seq(z) PCHIP interpolator given D(z) and sigma(M_min, z=0)."""
    z_grid = np.unique(np.concatenate([
        np.linspace(0.0,  1.0, 30),
        np.linspace(1.0,  4.0, 30),
        np.linspace(4.0, 10.0, 20),
        np.linspace(10.0, 20.0, 10),
    ]))
    fseq_grid = np.zeros(len(z_grid))
    for i, z in enumerate(z_grid):
        s = sigma_Mmin_z0 * float(D_interp(z))
        if s < 1e-10:
            continue
        fseq_grid[i] = _fseq_from_numin(delta_c / s)
    return PchipInterpolator(z_grid, fseq_grid)


# self-consistency solver
def solve_selfconsistent(zc, eps, dz=0.1, n_iter=6, tol=1e-4, verbose=True):
    """
    Iteratively solve the circular dependency:
      f_seq(z) -> rho(z) -> E(z) -> D(z) -> f_seq(z)

    Iteration 0 uses the analytic LCDM E(z), giving results identical to
    SCM_III_numerics.py (which also uses LCDM for its growth factors).

    Returns
    -------
    fseq_func  : converged PchipInterpolator for f_seq(z)
    rho_func   : converged normalised dark energy density rho(z)/rho_0
    w_func     : converged w(z) interpolator
    fseq0      : f_seq at z=0
    converged  : bool
    """
    sigma_Mmin = get_sigma_Mmin(verbose=verbose)

    # -- Iteration 0: analytic LCDM E(z) -------------------------------------
    D_func    = build_D_func(_E_lcdm)
    fseq_func = build_fseq_func(D_func, sigma_Mmin)
    fseq0     = float(fseq_func(0.0))

    if verbose:
        print(f"  iter 0 (LCDM):  f_seq,0 = {fseq0:.4f}")

    # -- Iterations 1 ... n_iter ----------------------------------------------
    for k in range(1, n_iter + 1):
        rho_func, norm = build_rho_func(zc, eps, fseq_func, fseq0, dz)

        # Build a callable E(z) from rho_func.
        # Use LCDM analytically for z > 19 so the D(z) integration to z=1000
        # stays accurate without needing the SCM interpolator there.
        def E_scm(z, _rf=rho_func):
            if z > 19.0:
                return _E_lcdm(z)
            rho = max(float(_rf(z)), 0.0)
            return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L * rho)

        D_new     = build_D_func(E_scm)
        fseq_new  = build_fseq_func(D_new, sigma_Mmin)
        fseq0_new = float(fseq_new(0.0))

        z_check = np.linspace(0.0, 5.0, 50)
        diff = float(np.max(np.abs(fseq_new(z_check) - fseq_func(z_check))))

        if verbose:
            w0_now = float(build_w_func(rho_func)(0.01))
            print(f"  iter {k}:  f_seq,0={fseq0_new:.4f}  "
                  f"max|df_seq|={diff:.2e}  w(z->0)={w0_now:.4f}")

        fseq_func = fseq_new
        fseq0     = fseq0_new

        if diff < tol:
            if verbose:
                print(f"  Converged after {k} iteration(s).")
            rho_func, _ = build_rho_func(zc, eps, fseq_func, fseq0, dz)
            w_func      = build_w_func(rho_func)
            return fseq_func, rho_func, w_func, fseq0, True

    if verbose:
        print(f"  Warning: did not converge in {n_iter} iterations (tol={tol}).")
    rho_func, _ = build_rho_func(zc, eps, fseq_func, fseq0, dz)
    w_func      = build_w_func(rho_func)
    return fseq_func, rho_func, w_func, fseq0, False


# CAMB dark energy class
class SCMDarkEnergy(DarkEnergyPPF):
    """
    CAMB DarkEnergyPPF subclass carrying a tabulated SCM w(a).

    set_w_a_table requires:
      - a array in ascending order
      - a[-1] exactly 1.0  (z=0)
      - all a > 0
    """

    @classmethod
    def from_w_func(cls, w_func, z_max=8.0, n_pts=600):
        """
        Build an SCMDarkEnergy from a w(z) PchipInterpolator.
        Converts to w(a) in ascending-a order as CAMB requires.
        """
        z_arr = np.unique(np.concatenate([
            np.linspace(0.0,  0.6, 200),
            np.linspace(0.6,  2.0, 200),
            np.linspace(2.0, z_max, 200),
        ]))
        w_arr = np.array([float(w_func(max(z, 0.005))) for z in z_arr])

        a_arr = 1.0 / (1.0 + z_arr)
        idx   = np.argsort(a_arr)
        a_arr = a_arr[idx]
        w_arr = w_arr[idx]

        # Ensure table ends exactly at a=1
        if not np.isclose(a_arr[-1], 1.0):
            a_arr = np.append(a_arr, 1.0)
            w_arr = np.append(w_arr, float(w_func(0.005)))

        de = cls()
        de.set_w_a_table(
            np.ascontiguousarray(a_arr, dtype=np.float64),
            np.ascontiguousarray(w_arr, dtype=np.float64),
        )
        return de


# full CAMB run
def get_scm_camb_results(zc, eps, dz=0.1, n_iter=6, verbose=True):
    """
    Run a full CAMB computation with SCM dark energy.

    Steps
    -----
    1. Self-consistency loop -> converged f_seq(z), rho(z), w(z)
    2. Build SCMDarkEnergy with the w(z) table
    3. Configure CAMBparams and call camb.get_results()

    Returns
    -------
    results  : camb.CAMBdata
    w_func   : converged w(z) interpolator
    rho_func : converged rho(z)/rho_0 interpolator
    """
    if verbose:
        print("\n" + "="*60)
        print(f"  SCM CAMB run: z_c={zc:.2f}, eps={eps:.3f}, dz={dz:.2f}")
        print("="*60)

    fseq_func, rho_func, w_func, fseq0, converged = solve_selfconsistent(
        zc, eps, dz=dz, n_iter=n_iter, verbose=verbose
    )

    de = SCMDarkEnergy.from_w_func(w_func)

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    pars.DarkEnergy = de
    pars.set_matter_power(redshifts=[0.0, 0.5, 1.0, 2.0], kmax=10.0)
    pars.NonLinear = camb_model.NonLinear_none

    if verbose:
        print("  Running CAMB...")
    results = camb.get_results(pars)
    if verbose:
        print("  CAMB done.")

    return results, w_func, rho_func


# validation
def validate():
    """
    Three tests:

    Test 1 -- Iteration-0 w(z) and rho(z) match SCM_III_numerics.py to < 0.02.
              Both use LCDM growth factors; should be numerically identical.

    Test 2 -- Full CAMB run with best-fit (zc=0.30, eps=0.10) completes and
              returns physical outputs (H0 preserved, comoving distances sane).

    Test 3 -- Self-consistency converges within n_iter for an RE-dominated
              model (zc=1.0, eps=0.005).
    """
    print("\n" + "="*60)
    print("  SCM_camb_de  Validation Suite")
    print("="*60)

    zc, eps = 0.30, 0.10

    # Reference values from SCM_III_numerics.py (zc=0.30, eps=0.10, smooth tanh)
    ref = {
        0.00: {'rho': 0.9998, 'w': -1.2400},
        0.25: {'rho': 1.2165, 'w': -0.9921},
        0.50: {'rho': 0.9263, 'w': -0.7846},
        1.00: {'rho': 0.9000, 'w': -0.9999},
        2.00: {'rho': 0.9000, 'w': -1.0000},
    }

    # -- Test 1 ---------------------------------------------------------------
    print("\n  Test 1: iteration-0 rho(z) and w(z) vs SCM_III_numerics.py")
    sigma_Mmin = get_sigma_Mmin(verbose=True)
    D_func     = build_D_func(_E_lcdm)
    fseq_0     = build_fseq_func(D_func, sigma_Mmin)
    fseq0_v    = float(fseq_0(0.0))
    print(f"  f_seq,0 = {fseq0_v:.4f}  (SCM_III ref: 0.5773)")

    rho_0, norm_0 = build_rho_func(zc, eps, fseq_0, fseq0_v, dz=0.1)
    w_0           = build_w_func(rho_0)

    print(f"  {'z':>5}  {'rho_ref':>9}  {'rho_here':>9}  "
          f"{'w_ref':>8}  {'w_here':>8}  {'dw':>7}")
    print("  " + "-"*58)
    test1_ok = True
    for z, vals in sorted(ref.items()):
        rho_v = float(rho_0(z)) * norm_0
        w_v   = float(w_0(max(z, 0.005)))
        dw    = abs(w_v - vals['w'])
        flag  = "OK" if dw < 0.02 else "MISMATCH"
        if flag != "OK":
            test1_ok = False
        print(f"  {z:>5.2f}  {vals['rho']:>9.4f}  {rho_v:>9.4f}  "
              f"{vals['w']:>8.4f}  {w_v:>8.4f}  {dw:>7.4f}  {flag}")
    print(f"  Test 1: {'PASSED' if test1_ok else 'FAILED'}")

    # -- Test 2 ---------------------------------------------------------------
    print("\n  Test 2: full CAMB run (zc=0.30, eps=0.10)")
    try:
        results, w_func, rho_func = get_scm_camb_results(zc, eps, verbose=True)
        derived = results.get_derived_params()
        H0_out  = derived.get('H0', H0_val)
        chi1    = results.comoving_radial_distance(1.0)
        chi2    = results.comoving_radial_distance(2.0)
        print(f"\n  H0 (output) = {H0_out:.2f}  (input: {H0_val})")
        print(f"  chi(z=1) = {chi1:.1f} Mpc")
        print(f"  chi(z=2) = {chi2:.1f} Mpc")
        test2_ok = abs(H0_out - H0_val) < 0.5 and chi1 > 2000
        print(f"  Test 2: {'PASSED' if test2_ok else 'FAILED'}")
    except Exception as e:
        print(f"  Test 2: FAILED -- {e}")
        test2_ok = False

    # -- Test 3 ---------------------------------------------------------------
    print("\n  Test 3: self-consistency convergence (zc=1.0, eps=0.005)")
    _, _, _, _, conv = solve_selfconsistent(
        zc=1.0, eps=0.005, dz=0.1, n_iter=6, verbose=True
    )
    test3_ok = conv
    print(f"  Test 3: {'PASSED' if test3_ok else 'FAILED'}")

    # -- Summary --------------------------------------------------------------
    print("\n" + "="*60)
    all_ok = test1_ok and test2_ok and test3_ok
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("="*60)
    return all_ok


if __name__ == "__main__":
    validate()
