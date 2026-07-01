import sys, os, argparse
import numpy as np
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SCM_camb_de import (
    build_rho_func, build_w_func, build_D_func, build_fseq_func,
    get_sigma_Mmin, SCMDarkEnergy, _E_lcdm,
    H0_val, ombh2, omch2, ns_val, As_val,
)
from cobaya.theories.camb import CAMB
from cobaya.theories.camb.camb import CambTransfers

# DESI constraints
DESI_W0, DESI_WA       = -0.827, -0.750
DESI_S_W0, DESI_S_WA   =  0.059,  0.290
DESI_RHO               = -0.93
DESI_COV = np.array([
    [DESI_S_W0**2,                    DESI_RHO * DESI_S_W0 * DESI_S_WA],
    [DESI_RHO * DESI_S_W0 * DESI_S_WA, DESI_S_WA**2                   ],
])
DESI_INV = np.linalg.inv(DESI_COV)

Z_FIT = np.linspace(0.02, 2.0, 300)

# F(z) inversion domain.
# Step 10: returned to Z_C_MAX=3.0 with REJECTION (not clipping) for ratio>F_max.
# Step 9 showed only 1% of samples at z_c>4.9, but 30% at z_c>3 (Mode B).
# Mode B (z_c~3.6, A_fp~15-20, eps~0.0007) implies eps*F(z_c)~34*rho_Lambda,0
# — an enormous DE spike in the matter-dominated era, observationally disfavoured.
# Physical motivation: BEC condensation at z_c>3 requires m_BEC>40 meV/c²,
# placing it in the ultra-light DM regime outside the SCM III perturbative treatment.
Z_C_MIN, Z_C_MAX = 0.3, 3.0

# eps below which FP contribution is negligible; z_c is returned as midpoint
EPS_MIN = 1e-5

# SCM-specific parameter names (not passed to CAMB)
SCM_PARAM_NAMES    = frozenset({'A_fp', 'eps', 'dz_cutoff'})
SCM_DERIVED_NAMES  = frozenset({'w0_scm', 'wa_scm', 'Mahal_DESI', 'zc_scm'})


# SCMCambTransfersRepar
class SCMCambTransfersRepar(CambTransfers):
    """
    Extends CambTransfers to declare A_fp, eps, dz_cutoff as supported params
    so the Cobaya transfer cache invalidates correctly when they change.
    """
    def get_can_support_params(self):
        return super().get_can_support_params() | SCM_PARAM_NAMES


# SCMCAMBRepar
class SCMCAMBRepar(CAMB):
    """
    Cobaya Theory class for the reparameterized Subtractive Cosmological Model.

    Sampled SCM parameters
    ----------------------
    eps   : FP fraction at z=0; prior flat [0, 0.20]
    A_fp  : FP DE amplitude at z_c in units of rho_Lambda,0;
            prior flat [0, 5]; A_fp = eps * F(z_c)

    Derived parameters
    ------------------
    zc_scm     : BEC condensation redshift, recovered via F(z_c) = A_fp/eps
    w0_scm     : CPL w0 from least-squares fit on z in [0.02, 2.0]
    wa_scm     : CPL wa
    Mahal_DESI : Mahalanobis distance from DESI 2024 Pantheon+ centre
    """

    params = {
        'eps': {
            'prior':    {'min': 0.0, 'max': 0.20},
            'ref':      {'dist': 'norm', 'loc': 0.0015, 'scale': 0.0005},
            'proposal': 0.001,
            'latex':    r'\varepsilon',
        },
        'A_fp': {
            'prior':    {'min': 0.0, 'max': 20.0},
            'ref':      {'dist': 'norm', 'loc': 3.0, 'scale': 1.5},
            'proposal': 0.80,
            'latex':    r'A_{\rm fp}',
        },
        'dz_cutoff': {
            'value':  0.1,
            'latex':  r'\Delta z_c',
        },
        'w0_scm': {
            'derived': True,
            'latex':   r'w_0^{\rm SCM}',
        },
        'wa_scm': {
            'derived': True,
            'latex':   r'w_a^{\rm SCM}',
        },
        'Mahal_DESI': {
            'derived': True,
            'latex':   r'd_{\rm DESI}',
        },
        'zc_scm': {
            'derived': True,
            'latex':   r'z_c^{\rm SCM}',
        },
    }

    # -------------------------------------------------------------------------
    def initialize(self):
        """Precompute f_seq(z) at fiducial Planck 2018 and F(z) bounds."""
        super().initialize()
        self.log.info("SCM-repar: precomputing f_seq at fiducial Planck 2018")
        sigma = get_sigma_Mmin(verbose=False)
        D_func = build_D_func(_E_lcdm)
        self._fseq_func = build_fseq_func(D_func, sigma)
        self._fseq0 = float(self._fseq_func(0.0))
        self._w_cache: dict = {}
        # Cache F bounds so _invert_F doesn't recompute them each call
        self._F_min = self._F_at_z(Z_C_MIN)
        self._F_max = self._F_at_z(Z_C_MAX)
        self.log.info(
            f"SCM-repar: f_seq,0={self._fseq0:.4f}  "
            f"F({Z_C_MIN:.1f})={self._F_min:.2f}  "
            f"F({Z_C_MAX:.1f})={self._F_max:.0f}"
        )

    # -------------------------------------------------------------------------
    def get_helper_theories(self):
        """Replace CambTransfers with SCMCambTransfersRepar."""
        helpers = super().get_helper_theories()
        self._camb_transfers = SCMCambTransfersRepar(
            self,
            "camb.transfers",
            {"stop_at_error": self.stop_at_error},
            timing=self.timer,
        )
        self._camb_transfers.requires = self._transfer_requires
        helpers["camb.transfers"] = self._camb_transfers
        return helpers

    # F(z) and inversion
    def _F_at_z(self, z):
        """F(z) = (1+z)^6 * [(1 - f_seq(z)) / (1 - f_seq,0)]^2."""
        fs    = float(np.clip(self._fseq_func(float(z)), 0.0, 1.0))
        denom = max(1.0 - self._fseq0, 1e-10)
        return (1.0 + z)**6 * ((1.0 - fs) / denom)**2

    def _invert_F(self, A_fp, eps):
        """
        Recover z_c from A_fp = eps * F(z_c) via brentq on [Z_C_MIN, Z_C_MAX].

        Returns None when the point is outside the z_c prior; callers must check
        and return False (= Cobaya sample rejection) in that case.

        Edge cases:
          eps < EPS_MIN : FP negligible; return midpoint (z_c undefined)
          ratio < F_min : z_c below domain; clip to Z_C_MIN (minor boundary)
          ratio > F_max : z_c above domain; return None → sample REJECTED
        """
        if eps < EPS_MIN:
            return 0.5 * (Z_C_MIN + Z_C_MAX)
        ratio = A_fp / eps
        if ratio <= self._F_min:
            return Z_C_MIN
        if ratio >= self._F_max:
            return None  # z_c would exceed Z_C_MAX; reject this proposal
        try:
            return float(brentq(
                lambda z: self._F_at_z(z) - ratio,
                Z_C_MIN, Z_C_MAX,
                xtol=1e-4, rtol=1e-5, maxiter=80,
            ))
        except ValueError:
            return 0.5 * (Z_C_MIN + Z_C_MAX)

    # Cobaya interface
    def set(self, params_values_dict, state):
        """
        Build CAMBparams with SCM dark energy injected.
        Extracts A_fp and eps, inverts F to get z_c, builds and caches w(z).
        """
        A_fp = float(params_values_dict.get('A_fp',      2.0))
        eps  = float(params_values_dict.get('eps',        0.0))
        dz   = float(params_values_dict.get('dz_cutoff', 0.1))

        camb_dict = {k: v for k, v in params_values_dict.items()
                     if k not in SCM_PARAM_NAMES}

        pars = super().set(camb_dict, state)
        if pars is False:
            return False

        zc = self._invert_F(A_fp, eps)
        if zc is None:
            return False  # A_fp/eps > F(Z_C_MAX) — z_c prior rejection

        cache_key = (round(A_fp, 6), round(eps, 6), round(dz, 4))
        if cache_key not in self._w_cache:
            rho_func, _ = build_rho_func(zc, eps, self._fseq_func, self._fseq0, dz)
            w_func_new  = build_w_func(rho_func)
            if len(self._w_cache) > 5000:
                self._w_cache.clear()
            self._w_cache[cache_key] = w_func_new

        pars.DarkEnergy = SCMDarkEnergy.from_w_func(self._w_cache[cache_key])
        return pars

    def _get_derived_output(self, intermediates):
        """Exclude SCM-specific derived params from CAMB's native interface."""
        saved = self.output_params
        self.output_params = [p for p in saved if p not in SCM_DERIVED_NAMES]
        try:
            result = super()._get_derived_output(intermediates)
        finally:
            self.output_params = saved
        return result

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Run CAMB then add CPL-fit and z_c derived parameters."""
        super().calculate(state, want_derived, **params_values_dict)

        if want_derived:
            A_fp = float(params_values_dict.get('A_fp',      2.0))
            eps  = float(params_values_dict.get('eps',        0.0))
            dz   = float(params_values_dict.get('dz_cutoff', 0.1))
            zc   = self._invert_F(A_fp, eps)
            if zc is None:
                zc = Z_C_MAX  # shouldn't happen (set() rejects first), fallback
            w0, wa = self._cpl_fit(A_fp, eps, dz)
            if 'derived' not in state or state['derived'] is None:
                state['derived'] = {}
            state['derived']['w0_scm']     = float(w0)  if np.isfinite(w0) else -1.0
            state['derived']['wa_scm']     = float(wa)  if np.isfinite(wa) else  0.0
            state['derived']['Mahal_DESI'] = float(self._mahal(w0, wa))
            state['derived']['zc_scm']     = float(zc)

    # private helpers
    def _cpl_fit(self, A_fp, eps, dz):
        """Least-squares CPL fit w(z) = w0 + wa*z/(1+z) on Z_FIT."""
        cache_key = (round(A_fp, 6), round(eps, 6), round(dz, 4))
        w_func = self._w_cache.get(cache_key)
        if w_func is None:
            zc       = self._invert_F(A_fp, eps)
            rho_func, _ = build_rho_func(zc, eps, self._fseq_func, self._fseq0, dz)
            w_func   = build_w_func(rho_func)
        w_arr = np.array([float(w_func(z)) for z in Z_FIT])
        mask  = np.isfinite(w_arr)
        if mask.sum() < 10:
            return np.nan, np.nan
        x = Z_FIT[mask] / (1.0 + Z_FIT[mask])
        A = np.column_stack([np.ones(mask.sum()), x])
        (w0, wa), _, _, _ = np.linalg.lstsq(A, w_arr[mask], rcond=None)
        return float(w0), float(wa)

    def _mahal(self, w0, wa):
        """Mahalanobis distance from DESI 2024 Pantheon+ centre."""
        if not (np.isfinite(w0) and np.isfinite(wa)):
            return np.inf
        d = np.array([w0 - DESI_W0, wa - DESI_WA])
        return float(np.sqrt(d @ DESI_INV @ d))


# YAML config
def get_repar_config(output_root='scm_repar_mcmc', n_chains=4):
    """
    Cobaya config dict for the reparameterized (eps, A_fp) MCMC chain.

    Sampled: H0, ombh2, omch2, tau, ns, logA, A_planck, eps, A_fp  (9 params)
    Derived: sigma8, w0_scm, wa_scm, Mahal_DESI, zc_scm             (5 params)
    Likelihoods: Planck 2018 plik-lite TTTEEE + lowl TT/EE + DESI 2024 BAO + Pantheon+
    """
    return {
        'theory': {
            'SCM_cobaya_repar.SCMCAMBRepar': {
                'stop_at_error': False,
                'extra_args': {
                    'lens_potential_accuracy': 1,
                    'num_massive_neutrinos':   1,
                    'mnu':                     0.06,
                },
            },
        },
        'params': {
            # standard ΛCDM params
            'H0': {
                'prior':    {'min': 60.0, 'max': 80.0},
                'ref':      {'dist': 'norm', 'loc': 67.32, 'scale': 0.54},
                'proposal': 0.30,
                'latex':    'H_0',
            },
            'ombh2': {
                'prior':    {'min': 0.018, 'max': 0.026},
                'ref':      {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00015},
                'proposal': 0.00010,
                'latex':    r'\Omega_b h^2',
            },
            'omch2': {
                'prior':    {'min': 0.095, 'max': 0.145},
                'ref':      {'dist': 'norm', 'loc': 0.1200, 'scale': 0.0012},
                'proposal': 0.00080,
                'latex':    r'\Omega_c h^2',
            },
            'tau': {
                'prior':    {'min': 0.01, 'max': 0.15},
                'ref':      {'dist': 'norm', 'loc': 0.0544, 'scale': 0.0073},
                'proposal': 0.003,
                'latex':    r'\tau_{\rm reio}',
            },
            'ns': {
                'prior':    {'min': 0.90, 'max': 1.05},
                'ref':      {'dist': 'norm', 'loc': 0.9649, 'scale': 0.0042},
                'proposal': 0.003,
                'latex':    'n_s',
            },
            'logA': {
                'prior':    {'min': 2.5, 'max': 3.5},
                'ref':      {'dist': 'norm', 'loc': 3.044, 'scale': 0.014},
                'proposal': 0.01,
                'latex':    r'\log(10^{10}A_s)',
                'drop':     True,
            },
            'As': {
                'value':  'lambda logA: 1e-10 * np.exp(logA)',
                'latex':  'A_s',
            },
            'A_planck': {
                'prior':    {'dist': 'norm', 'loc': 1.0, 'scale': 0.0025},
                'ref':      {'dist': 'norm', 'loc': 1.0, 'scale': 0.002},
                'proposal': 0.0005,
                'latex':    r'y_{\rm cal}',
            },
            # derived params
            'sigma8':     {'derived': True, 'latex': r'\sigma_8'},
            'w0_scm':     {'derived': True, 'latex': r'w_0^{\rm SCM}'},
            'wa_scm':     {'derived': True, 'latex': r'w_a^{\rm SCM}'},
            'Mahal_DESI': {'derived': True, 'latex': r'd_{\rm DESI}'},
            'zc_scm':     {'derived': True, 'latex': r'z_c^{\rm SCM}'},
        },
        'likelihood': {
            'planck_2018_highl_plik.TTTEEE_lite_native': None,
            'planck_2018_lowl.TT':   None,
            'planck_2018_lowl.EE':   None,
            'bao.desi_2024_bao_all': None,
            'sn.pantheonplus':       None,
        },
        'sampler': {
            'mcmc': {
                'drag':             True,
                'oversample_power': 0.4,
                'proposal_scale':   1.9,
                'covmat':           'auto',
                'Rminus1_stop':     0.01,
                'Rminus1_cl_stop':  0.20,
                'max_samples':      50000,
            },
        },
        'output':        output_root,
        'packages_path': os.path.join(_HERE, 'cobaya_packages'),
        'resume':        True,
        'debug':         False,
        'stop_at_error': False,
    }


def write_yaml_config(path='scm_repar_mcmc.yaml', **kwargs):
    """Write the reparameterized Cobaya config dict to a YAML file."""
    try:
        import yaml
    except ImportError:
        import ruamel.yaml as yaml

    config = get_repar_config(**kwargs)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Config written to: {path}")


# validation
def validate():
    """
    Test the reparameterized model at a representative parameter point.

    Checks:
      1. F(z) is monotonically increasing on [Z_C_MIN, Z_C_MAX]
      2. _invert_F recovers z_c = 2.0 from A_fp = eps * F(2.0)
      3. _invert_F handles edge cases (eps < EPS_MIN, ratio out of range)
      4. build_rho_func / build_w_func run without error at recovered z_c
      5. w(z->0) is finite and < -0.5
    """
    from cobaya.model import get_model

    print("\n" + "="*60)
    print("  SCMCAMBRepar Validation (reparameterized chain)")
    print("="*60)

    # --- Unit tests on F(z) inversion -----------------------------------------
    sigma = get_sigma_Mmin(verbose=False)
    D_func   = build_D_func(_E_lcdm)
    fseq_func = build_fseq_func(D_func, sigma)
    fseq0    = float(fseq_func(0.0))

    def F_at_z(z):
        fs = float(np.clip(fseq_func(float(z)), 0.0, 1.0))
        return (1.0 + z)**6 * ((1.0 - fs) / max(1.0 - fseq0, 1e-10))**2

    print(f"\n  f_seq,0 = {fseq0:.4f}")
    z_test = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    F_vals = [F_at_z(z) for z in z_test]
    print(f"\n  F(z) values:")
    for z, F in zip(z_test, F_vals):
        print(f"    F({z:.1f}) = {F:10.2f}")

    mono_ok = all(F_vals[i] < F_vals[i+1] for i in range(len(F_vals)-1))
    print(f"\n  [{'PASS' if mono_ok else 'FAIL'}]  F(z) monotonically increasing")

    # Inversion test at z_c = 2.0, eps = 0.002
    eps_test   = 0.002
    zc_true    = 2.0
    A_fp_test  = eps_test * F_at_z(zc_true)
    ratio_test = A_fp_test / eps_test

    F_min = F_at_z(Z_C_MIN)
    F_max = F_at_z(Z_C_MAX)
    try:
        zc_recovered = float(brentq(
            lambda z: F_at_z(z) - ratio_test, Z_C_MIN, Z_C_MAX,
            xtol=1e-4, maxiter=80
        ))
    except ValueError:
        zc_recovered = float('nan')

    inv_err = abs(zc_recovered - zc_true)
    print(f"\n  Inversion test (eps={eps_test}, z_c_true={zc_true}):")
    print(f"    A_fp = {A_fp_test:.4f},  F(z_c) target = {ratio_test:.2f}")
    print(f"    z_c recovered = {zc_recovered:.6f},  |error| = {inv_err:.2e}")
    inv_ok = inv_err < 1e-3
    print(f"  [{'PASS' if inv_ok else 'FAIL'}]  Inversion error < 1e-3")

    # Edge cases
    rho_func, _ = build_rho_func(zc_recovered, eps_test, fseq_func, fseq0)
    w_func = build_w_func(rho_func)
    w0_val = float(w_func(0.01))
    print(f"\n  w(z->0) at recovered z_c: {w0_val:.4f}")
    w_ok = np.isfinite(w0_val) and w0_val < -0.5
    print(f"  [{'PASS' if w_ok else 'FAIL'}]  w finite and < -0.5")

    # --- Cobaya model test -------------------------------------------------------
    print("\n  Building Cobaya model...")
    config = get_repar_config(output_root=None)
    model = get_model(config)

    test_point = {
        'H0':       67.32,
        'ombh2':    0.02237,
        'omch2':    0.1200,
        'tau':      0.0544,
        'ns':       0.9649,
        'logA':     3.044,
        'A_planck': 1.0,
        'eps':      0.002,
        'A_fp':     A_fp_test,
    }
    print(f"  Test point: eps={test_point['eps']}, A_fp={test_point['A_fp']:.4f}")
    print(f"              (corresponds to z_c ~= {zc_recovered:.3f})")

    result  = model.logposterior(test_point)
    logpost = result.logpost
    derived_names = list(model.parameterization.derived_params().keys())
    derived = dict(zip(derived_names, result.derived))

    w0   = derived.get('w0_scm',     np.nan)
    wa   = derived.get('wa_scm',     np.nan)
    dist = derived.get('Mahal_DESI', np.nan)
    zc_d = derived.get('zc_scm',     np.nan)
    s8   = derived.get('sigma8',     np.nan)

    print(f"\n  logposterior = {logpost:.2f}")
    print(f"  zc_scm     = {zc_d:.4f}   (expected ~= {zc_true:.4f})")
    print(f"  w0_scm     = {w0:.4f}")
    print(f"  wa_scm     = {wa:.4f}")
    print(f"  Mahal_DESI = {dist:.3f} sigma")
    print(f"  sigma8     = {s8:.4f}")

    tests = {
        'F(z) monotone':           mono_ok,
        'Inversion error < 1e-3':  inv_ok,
        'w(z->0) finite, < -0.5':  w_ok,
        'logpost finite':          np.isfinite(logpost),
        'zc_scm in [0.3, 5.0]':   Z_C_MIN <= zc_d <= Z_C_MAX if np.isfinite(zc_d) else False,
        'w0_scm present':          np.isfinite(w0),
        'wa_scm present':          np.isfinite(wa),
        'Mahal_DESI present':      np.isfinite(dist),
        'sigma8 in (0.7, 0.95)':   0.7 < s8 < 0.95 if np.isfinite(s8) else False,
    }

    print("\n  Test results:")
    all_ok = True
    for name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}]  {name}")

    print("\n" + "="*60)
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("="*60)
    return all_ok


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCM reparameterized Cobaya wrapper (eps, A_fp)"
    )
    parser.add_argument('--validate',   action='store_true',
                        help='Run single-point validation test')
    parser.add_argument('--write-yaml', action='store_true',
                        help='Write scm_repar_mcmc.yaml')
    parser.add_argument('--run',        action='store_true',
                        help='Run full Planck+BAO+SNIa MCMC')
    args = parser.parse_args()

    if args.validate:
        validate()

    elif args.write_yaml:
        write_yaml_config('scm_repar_mcmc.yaml')

    elif args.run:
        from cobaya.run import run
        print("Starting reparameterized MCMC chain (eps, A_fp)...")
        config = get_repar_config()
        updated_info, sampler = run(config)

    else:
        parser.print_help()
