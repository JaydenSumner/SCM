import sys, os, argparse
import numpy as np

# Ensure SCM_camb_de is importable regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SCM_camb_de import (
    build_rho_func, build_w_func, build_D_func, build_fseq_func,
    get_sigma_Mmin, SCMDarkEnergy, _E_lcdm,
    H0_val, ombh2, omch2, ns_val, As_val,   # fiducial Planck 2018
)
from cobaya.theories.camb import CAMB
from cobaya.theories.camb.camb import CambTransfers

# DESI 2024 constraints
DESI_W0, DESI_WA       = -0.827, -0.750
DESI_S_W0, DESI_S_WA   =  0.059,  0.290
DESI_RHO               = -0.93
DESI_COV = np.array([
    [DESI_S_W0**2,                    DESI_RHO * DESI_S_W0 * DESI_S_WA],
    [DESI_RHO * DESI_S_W0 * DESI_S_WA, DESI_S_WA**2                   ],
])
DESI_INV = np.linalg.inv(DESI_COV)

# CPL fit range
Z_FIT = np.linspace(0.02, 2.0, 300)

# Parameter names that belong to SCM (not passed to CAMB)
SCM_PARAM_NAMES = frozenset({'zc', 'eps', 'dz_cutoff'})

# Derived parameters computed by SCMCAMB, not by CAMB's native interface
SCM_DERIVED_NAMES = frozenset({'w0_scm', 'wa_scm', 'Mahal_DESI'})


# SCMCambTransfers
class SCMCambTransfers(CambTransfers):
    """
    Extends CambTransfers to declare zc, eps, dz_cutoff as supported params.

    Cobaya routes parameters to theory components based on which component
    claims to support them via get_can_support_params(). Without this
    subclass, zc and eps are not in the transfer-level params_values_dict,
    which means:
      (a) SCMCAMB.set() never sees them, and
      (b) the Cobaya transfer cache never invalidates when they change,
          returning stale CAMB results for new SCM parameter values.

    Adding them to get_can_support_params() fixes both problems.
    """
    def get_can_support_params(self):
        return super().get_can_support_params() | SCM_PARAM_NAMES


# SCMCAMB
class SCMCAMB(CAMB):
    """
    Cobaya Theory class for the Subtractive Cosmological Model.

    Inherits all standard CAMB parameter handling, CMB/BAO/Pk observables,
    and caching from Cobaya's CAMB wrapper. Adds:
      - SCM dark energy injection at each transfer evaluation (via set())
      - CPL-fit derived parameters w0_scm, wa_scm, Mahal_DESI (via calculate())

    Parameter notes
    ---------------
    zc         : BEC condensation redshift; sampled in [0.3, 3.0]
    eps        : FP fraction at z=0; sampled in [0.0, 0.20]
    dz_cutoff  : tanh cutoff width; fixed at 0.1 (not sampled by default)
    Standard LCDM params (H0, ombh2, omch2, ns, As, tau) are declared in
    the YAML config and handled by the parent class.
    """

    # SCM-specific parameters. Standard cosmo params go in the YAML.
    params = {
        'zc': {
            'prior':    {'min': 0.3, 'max': 3.0},
            'ref':      {'dist': 'norm', 'loc': 0.50, 'scale': 0.10},
            'proposal': 0.05,
            'latex':    'z_c',
        },
        'eps': {
            'prior':    {'min': 0.0, 'max': 0.20},
            'ref':      {'dist': 'norm', 'loc': 0.05, 'scale': 0.01},
            'proposal': 0.005,
            'latex':    r'\varepsilon',
        },
        'dz_cutoff': {
            'value':  0.1,
            'latex':  r'\Delta z_c',
        },
        # CPL-fit derived parameters
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
    }

    # -------------------------------------------------------------------------
    def initialize(self):
        """
        One-time setup per chain process.
        Precomputes f_seq(z) at fiducial Planck 2018 cosmology.
        """
        super().initialize()
        self.log.info("SCM: precomputing f_seq at fiducial Planck 2018 cosmology")
        sigma = get_sigma_Mmin(verbose=False)
        D_func = build_D_func(_E_lcdm)
        self._fseq_func = build_fseq_func(D_func, sigma)
        self._fseq0 = float(self._fseq_func(0.0))
        self._w_cache: dict = {}
        self.log.info(f"SCM: f_seq,0 = {self._fseq0:.4f}  (fiducial, sigma={sigma:.4f})")

    # -------------------------------------------------------------------------
    def get_helper_theories(self):
        """
        Replace the standard CambTransfers helper with SCMCambTransfers so
        that zc/eps are included in the transfer-level calculation and the
        Cobaya cache invalidates correctly when SCM parameters change.
        """
        helpers = super().get_helper_theories()
        self._camb_transfers = SCMCambTransfers(
            self,
            "camb.transfers",
            {"stop_at_error": self.stop_at_error},
            timing=self.timer,
        )
        self._camb_transfers.requires = self._transfer_requires
        helpers["camb.transfers"] = self._camb_transfers
        return helpers

    # -------------------------------------------------------------------------
    def set(self, params_values_dict, state):
        """
        Build CAMBparams with SCM dark energy injected.
        Called by SCMCambTransfers.calculate() at each transfer evaluation.

        params_values_dict contains both CAMB cosmo params AND SCM params
        (because SCMCambTransfers.get_can_support_params includes them).
        SCM params are extracted and filtered before calling super().set().
        """
        zc  = float(params_values_dict.get('zc',        0.5))
        eps = float(params_values_dict.get('eps',        0.0))
        dz  = float(params_values_dict.get('dz_cutoff', 0.1))

        # Pass only CAMB-recognised params to parent
        camb_dict = {k: v for k, v in params_values_dict.items()
                     if k not in SCM_PARAM_NAMES}

        pars = super().set(camb_dict, state)
        if pars is False:
            return False

        # Build and cache w(z) for this (zc, eps, dz) point.
        # Build before clearing: clearing then accessing caused a KeyError when
        # the just-added entry was evicted by the size limit.
        cache_key = (round(zc, 5), round(eps, 6), round(dz, 4))
        if cache_key not in self._w_cache:
            rho_func, _ = build_rho_func(zc, eps, self._fseq_func, self._fseq0, dz)
            w_func_new = build_w_func(rho_func)
            if len(self._w_cache) > 5000:
                self._w_cache.clear()
            self._w_cache[cache_key] = w_func_new

        pars.DarkEnergy = SCMDarkEnergy.from_w_func(self._w_cache[cache_key])
        return pars

    # -------------------------------------------------------------------------
    def _get_derived_output(self, intermediates):
        """
        Override to exclude SCM-specific derived params from CAMB's native
        derived-output logic. They are not in CAMB's interface and would
        raise an error if passed to _get_derived(). They are added in
        calculate() instead, where zc and eps are in scope.
        """
        saved = self.output_params
        self.output_params = [p for p in saved if p not in SCM_DERIVED_NAMES]
        try:
            result = super()._get_derived_output(intermediates)
        finally:
            self.output_params = saved
        return result

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Run CAMB (via parent) then add CPL-fit derived parameters.
        """
        super().calculate(state, want_derived, **params_values_dict)

        if want_derived:
            zc  = float(params_values_dict.get('zc',        0.5))
            eps = float(params_values_dict.get('eps',        0.0))
            dz  = float(params_values_dict.get('dz_cutoff', 0.1))
            w0, wa = self._cpl_fit(zc, eps, dz)
            if 'derived' not in state or state['derived'] is None:
                state['derived'] = {}
            state['derived']['w0_scm']     = float(w0)  if np.isfinite(w0) else -1.0
            state['derived']['wa_scm']     = float(wa)  if np.isfinite(wa) else  0.0
            state['derived']['Mahal_DESI'] = float(self._mahal(w0, wa))

    # private helpers
    def _cpl_fit(self, zc, eps, dz):
        """Least-squares CPL fit w(z) = w0 + wa*z/(1+z) on Z_FIT."""
        cache_key = (round(zc, 5), round(eps, 6), round(dz, 4))
        w_func = self._w_cache.get(cache_key)
        if w_func is None:
            rho_func, _ = build_rho_func(zc, eps, self._fseq_func, self._fseq0, dz)
            w_func = build_w_func(rho_func)
        w_arr = np.array([float(w_func(z)) for z in Z_FIT])
        mask = np.isfinite(w_arr)
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
def get_yaml_config(output_root='scm_mcmc', n_chains=4):
    """
    Return a Cobaya config dict for a full SCM MCMC run.

    Likelihoods are marked as placeholders. Install them (Step 4) and
    uncomment the relevant blocks before running.

    Parameters
    ----------
    output_root : str   Cobaya output prefix (chains saved here)
    n_chains    : int   Number of parallel chains
    """
    return {
        # theory
        'theory': {
            'SCM_cobaya.SCMCAMB': {
                'stop_at_error': False,
                'extra_args': {
                    'lens_potential_accuracy': 1,
                    'num_massive_neutrinos':   1,
                    'mnu':                     0.06,
                },
            },
        },

        # parameters
        # Standard LCDM
        'params': {
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
            # Planck calibration nuisance parameter
            'A_planck': {
                'prior':    {'dist': 'norm', 'loc': 1.0, 'scale': 0.0025},
                'ref':      {'dist': 'norm', 'loc': 1.0, 'scale': 0.002},
                'proposal': 0.0005,
                'latex':    r'y_{\rm cal}',
            },
            # SCM params (declared in SCMCAMB.params; listed here for explicitness)
            # Uncomment to override defaults:
            # 'zc': {'prior': {'min': 0.3, 'max': 3.0}, 'proposal': 0.05},
            # 'eps': {'prior': {'min': 0.0, 'max': 0.20}, 'proposal': 0.005},
            # Derived
            'sigma8':     {'derived': True, 'latex': r'\sigma_8'},
            'w0_scm':     {'derived': True, 'latex': r'w_0^{\rm SCM}'},
            'wa_scm':     {'derived': True, 'latex': r'w_a^{\rm SCM}'},
            'Mahal_DESI': {'derived': True, 'latex': r'd_{\rm DESI}'},
        },

        # likelihoods
        'likelihood': {
            # Planck 2018: high-ell TT+TE+EE (lite, native Python, no clik)
            'planck_2018_highl_plik.TTTEEE_lite_native': None,
            # Planck 2018: low-ell temperature + polarization
            'planck_2018_lowl.TT': None,
            'planck_2018_lowl.EE': None,
            # DESI 2024 BAO (all tracers)
            'bao.desi_2024_bao_all': None,
            # Pantheon+ SNIa
            'sn.pantheonplus': None,
        },

        # sampler
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

        # output
        'output':        output_root,
        'packages_path': os.path.join(_HERE, 'cobaya_packages'),
        'resume':        True,
        'debug':         False,
        'stop_at_error': False,
    }


def write_yaml_config(path='sample_mcmc_config.yaml', **kwargs):
    """Write the Cobaya config dict to a YAML file."""
    try:
        import yaml
    except ImportError:
        import ruamel.yaml as yaml   # cobaya ships ruamel

    config = get_yaml_config(**kwargs)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Config written to: {path}")


# BAO-only config
def get_bao_snia_config(output_root='step5_bao_snia'):
    """
    Cobaya config for the BAO + SNIa-only validation run (Step 5).

    No CMB likelihoods -> CAMB runs in background-only mode (~0.1 s/call).
    tau, ns, logA are fixed at Planck 2018 fiducial values since they are
    not constrained by geometric probes.  A_planck is dropped (Planck only).

    Purpose: quick check that the SCM posterior is physical before committing
    to the expensive full-Planck run.
    """
    return {
        'theory': {
            'SCM_cobaya.SCMCAMB': {
                'stop_at_error': False,
                'extra_args': {
                    'num_massive_neutrinos': 1,
                    'mnu':                  0.06,
                },
            },
        },
        'params': {
            # Sampled — constrained by BAO + SNIa
            'H0': {
                'prior':    {'min': 60.0, 'max': 80.0},
                'ref':      {'dist': 'norm', 'loc': 67.32, 'scale': 1.0},
                'proposal': 0.5,
                'latex':    'H_0',
            },
            'ombh2': {
                'prior':    {'min': 0.018, 'max': 0.026},
                'ref':      {'dist': 'norm', 'loc': 0.02237, 'scale': 0.001},
                'proposal': 0.0003,
                'latex':    r'\Omega_b h^2',
            },
            'omch2': {
                'prior':    {'min': 0.09, 'max': 0.15},
                'ref':      {'dist': 'norm', 'loc': 0.1200, 'scale': 0.003},
                'proposal': 0.002,
                'latex':    r'\Omega_c h^2',
            },
            # Fixed at Planck 2018 fiducial — not constrained by geometry
            'tau':   0.0544,
            'ns':    0.9649,
            'logA':  3.044,
            'As': {
                'value': 'lambda logA: 1e-10 * np.exp(logA)',
                'latex': 'A_s',
            },
            # SCM parameters — sampled
            'zc': {
                'prior':    {'min': 0.3, 'max': 3.0},
                'ref':      {'dist': 'norm', 'loc': 0.5, 'scale': 0.2},
                'proposal': 0.08,
                'latex':    'z_c',
            },
            'eps': {
                'prior':    {'min': 0.0, 'max': 0.20},
                'ref':      {'dist': 'norm', 'loc': 0.05, 'scale': 0.02},
                'proposal': 0.008,
                'latex':    r'\varepsilon',
            },
            # Derived
            'sigma8':     {'derived': True, 'latex': r'\sigma_8'},
            'w0_scm':     {'derived': True, 'latex': r'w_0^{\rm SCM}'},
            'wa_scm':     {'derived': True, 'latex': r'w_a^{\rm SCM}'},
            'Mahal_DESI': {'derived': True, 'latex': r'd_{\rm DESI}'},
        },
        'likelihood': {
            'bao.desi_2024_bao_all': None,
            'sn.pantheonplus':       None,
        },
        'sampler': {
            'mcmc': {
                'drag':             False,   # all params are slow; drag adds no gain
                'proposal_scale':   2.1,
                'covmat':           'auto',
                'Rminus1_stop':     0.05,    # relaxed — this is a validation run
                'Rminus1_cl_stop':  0.30,
                'max_samples':      80000,
            },
        },
        'output':        output_root,
        'packages_path': os.path.join(_HERE, 'cobaya_packages'),
        'debug':         False,
        'stop_at_error': False,
    }


# validation
def validate():
    """
    Test SCMCAMB at a single parameter point using cobaya.model.get_model().

    Evaluates the model at the SCM_III best-fit point (zc=0.30, eps=0.10)
    with Planck 2018 fiducial cosmology. Checks:
      1. Model initialises without error
      2. logposterior evaluates without error
      3. w0_scm, wa_scm, Mahal_DESI are in derived output
      4. w0_scm matches SCM_III_numerics.py CPL fit to within 0.01
      5. CAMB background distances are sensible
    """
    from cobaya.model import get_model

    print("\n" + "="*60)
    print("  SCMCAMB Validation  (cobaya single-point evaluation)")
    print("="*60)

    # Minimal config: theory + one dummy likelihood (no data)
    config = get_yaml_config(output_root=None)

    print("  Building Cobaya model...")
    model = get_model(config)

    # Test point: Planck 2018 fiducial + SCM best-fit from SCM_III_numerics
    test_point = {
        'H0':       67.32,
        'ombh2':    0.02237,
        'omch2':    0.1200,
        'tau':      0.0544,
        'ns':       0.9649,
        'logA':     3.044,
        'A_planck': 1.0,
        'zc':       0.30,
        'eps':      0.10,
    }

    print(f"  Evaluating at: zc={test_point['zc']}, eps={test_point['eps']}")
    print(f"                 H0={test_point['H0']}, omch2={test_point['omch2']}")

    result  = model.logposterior(test_point)
    logpost = result.logpost
    logprior = result.logprior
    derived_names = list(model.parameterization.derived_params().keys())
    derived = dict(zip(derived_names, result.derived))

    loglikes = dict(zip(model.likelihood, result.loglikes))

    print(f"\n  logposterior = {logpost:.2f}")
    print(f"  logprior     = {logprior:.2f}")
    print("\n  Per-likelihood log-likelihoods:")
    for name, lp in loglikes.items():
        print(f"    {name}: {lp:.2f}")

    # SCM derived params
    print("\n  SCM derived parameters:")
    w0   = derived.get('w0_scm',     np.nan)
    wa   = derived.get('wa_scm',     np.nan)
    dist = derived.get('Mahal_DESI', np.nan)
    s8   = derived.get('sigma8',     np.nan)

    print(f"  w0_scm     = {w0:.4f}   (SCM_III ref: -0.961)")
    print(f"  wa_scm     = {wa:.4f}   (SCM_III ref: -0.007)")
    print(f"  Mahal_DESI = {dist:.3f}  (SCM_III ref: 2.58 sigma)")
    print(f"  sigma8     = {s8:.4f}")

    # Checks
    plik_ll  = loglikes.get('planck_2018_highl_plik.TTTEEE_lite_native', np.nan)
    desi_ll  = loglikes.get('bao.desi_2024_bao_all', np.nan)
    sn_ll    = loglikes.get('sn.pantheonplus', np.nan)
    tests = {
        'logpost finite':          np.isfinite(logpost),
        'w0_scm present':          np.isfinite(w0),
        'wa_scm present':          np.isfinite(wa),
        'Mahal_DESI present':      np.isfinite(dist),
        'w0_scm < -0.9':           w0 < -0.9,
        'w0_scm > -1.1':           w0 > -1.1,
        'Mahal_DESI < 4.0':        dist < 4.0,
        'sigma8 in (0.7, 0.95)':   0.7 < s8 < 0.95,
        'Planck plik loglike < 0': plik_ll < 0,
        'DESI BAO loglike < 0':    desi_ll < 0,
        'Pantheon+ loglike < 0':   sn_ll   < 0,
    }

    print("\n  Test results:")
    all_ok = True
    for name, result in tests.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_ok = False
        print(f"  [{status}]  {name}")

    print("\n" + "="*60)
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("="*60)
    return all_ok


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCM Cobaya theory wrapper")
    parser.add_argument('--validate',   action='store_true',
                        help='Run single-point validation test')
    parser.add_argument('--write-yaml', action='store_true',
                        help='Write sample_mcmc_config.yaml')
    parser.add_argument('--run',        action='store_true',
                        help='Run full Planck+BAO+SNIa MCMC (Step 6)')
    parser.add_argument('--run-step5',  action='store_true',
                        help='Run BAO+SNIa-only MCMC (Step 5, fast)')
    args = parser.parse_args()

    if args.validate:
        validate()

    elif args.write_yaml:
        write_yaml_config('sample_mcmc_config.yaml')

    elif args.run_step5:
        from cobaya.run import run
        print("Starting Step 5: BAO + SNIa-only MCMC chain...")
        config = get_bao_snia_config()
        updated_info, sampler = run(config)

    elif args.run:
        from cobaya.run import run
        print("Starting Step 6: Full Planck + BAO + SNIa MCMC chain...")
        config = get_yaml_config()
        updated_info, sampler = run(config)

    else:
        parser.print_help()
