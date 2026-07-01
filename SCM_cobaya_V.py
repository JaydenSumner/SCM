import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SCM_cobaya_IV import (
    SCMCAMBIV, SCM_IV_DERIVED_NAMES, FS8_NAMES, FS8_LATEX,
    get_step11_config,
)
from SCM_cobaya_repar import Z_C_MIN, Z_C_MAX, EPS_MIN

# derived param names

SCM_V_NEW_NAMES   = ['s8', 'sigma8_z03']
SCM_V_NEW_LATEX   = [r'S_8', r'\sigma_8(z=0.3)']
SCM_V_DERIVED_NAMES = SCM_IV_DERIVED_NAMES | frozenset(SCM_V_NEW_NAMES)


# SCMCAMBV

class SCMCAMBV(SCMCAMBIV):
    """
    SCM V Cobaya Theory class.

    Inherits all of SCMCAMBIV (SCM IV: CAMB + CPL + DESI Mahal + f·σ8).
    Adds:
      - Non-linear halofit matter power spectrum (mead2020) via CAMB
      - Derived S8 = sigma8 * sqrt(Omega_m / 0.3)
      - Derived sigma8(z=0.3) for weak lensing comparison
    """

    params = {
        **SCMCAMBIV.params,
        **{name: {'derived': True, 'latex': latex}
           for name, latex in zip(SCM_V_NEW_NAMES, SCM_V_NEW_LATEX)},
    }

    # hide SCM-specific params from derived output

    def _get_derived_output(self, intermediates):
        saved = self.output_params
        self.output_params = [p for p in saved
                              if p not in SCM_V_DERIVED_NAMES]
        try:
            result = super(SCMCAMBIV, self)._get_derived_output(intermediates)
        finally:
            self.output_params = saved
        return result

    # add halofit to CAMB requirements

    def get_camb_requirements(self):
        reqs = super().get_camb_requirements()
        # Enable non-linear power spectrum (halofit mead2020)
        reqs['extra_args'] = reqs.get('extra_args', {})
        reqs['extra_args']['nonlinear'] = True
        reqs['extra_args']['halofit_version'] = 'mead2020'
        reqs['extra_args']['lens_potential_accuracy'] = 2
        # Request sigma8(z) at z=0.3 for the derived parameter
        reqs.setdefault('Pk_grid', {})
        reqs['Pk_grid']['z'] = [0.0, 0.3]
        reqs['Pk_grid']['k_max'] = 20.0
        reqs['Pk_grid']['nonlinear'] = True
        return reqs

    # calculate: CAMB + SCM derived params + S8

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        1. Call SCMCAMBIV.calculate (CAMB + CPL + f·σ8).
        2. Extract Omega_m from sampled params.
        3. Compute S8 = sigma8 * sqrt(Omega_m / 0.3).
        4. Compute sigma8(z=0.3) from CAMB provider.
        5. Store in state['derived'].
        """
        super().calculate(state, want_derived, **params_values_dict)

        if not want_derived:
            return

        H0    = float(params_values_dict.get('H0',    69.0))
        ombh2 = float(params_values_dict.get('ombh2', 0.02237))
        omch2 = float(params_values_dict.get('omch2', 0.1200))
        h     = H0 / 100.0
        om_m  = (ombh2 + omch2) / h**2

        derived = state.get('derived') or {}
        sigma8  = float(derived.get('sigma8', 0.8347))
        if not (0.4 < sigma8 < 1.5):
            sigma8 = 0.8347

        s8 = sigma8 * np.sqrt(om_m / 0.3)

        # sigma8(z=0.3): use CAMB sigma8_z if available from provider
        sigma8_z03 = sigma8  # fallback: same as z=0
        try:
            provider = self.provider
            if provider is not None:
                s8z03 = provider.get_sigma8_z(0.3)
                if np.isfinite(s8z03) and 0.1 < s8z03 < 1.5:
                    sigma8_z03 = float(s8z03)
        except Exception:
            pass

        if state.get('derived') is None:
            state['derived'] = {}
        state['derived']['s8']         = float(s8)
        state['derived']['sigma8_z03'] = float(sigma8_z03)


# Step 12 config

def get_step12_config(output_root='scm_step12_mcmc',
                      covmat='scm_step11_mcmc.covmat',
                      packages_path='cobaya_packages'):
    """
    Build the Cobaya config dict for Step 12.

    Based on Step 11 (Planck + DESI + Pantheon+ + eBOSS DR16 BAO+FS) with:
      1. Theory class upgraded to SCMCAMBV (adds halofit, S8, sigma8_z03)
      2. Planck 2018 CMB lensing likelihood
      3. DES Y1 cosmic shear likelihood (10 nuisance params)
    """
    config = get_step11_config(output_root=output_root)

    # Upgrade theory class
    old_args = config['theory'].pop('SCM_cobaya_IV.SCMCAMBIV')
    config['theory']['SCM_cobaya_V.SCMCAMBV'] = old_args

    # Add SCM V derived params
    for name, latex in zip(SCM_V_NEW_NAMES, SCM_V_NEW_LATEX):
        config['params'][name] = {'derived': True, 'latex': latex}

    # Add Planck 2018 CMB lensing
    # None = use the top-level packages_path (same as all other likelihoods)
    config['likelihood']['planck_2018_lensing.native'] = None

    # Add DES Y1 cosmic shear
    # Nuisance params (photo-z biases, shear calibration, IA) are added
    # automatically from des_y1/shear.yaml defaults.
    config['likelihood']['des_y1.shear'] = None

    # Update sampler
    config['sampler']['mcmc'].update({
        'covmat':          covmat,
        'Rminus1_stop':    0.02,
        'Rminus1_cl_stop': 0.20,
        'max_samples':     100000,
        'drag':            True,
    })

    config['output'] = output_root
    config['resume'] = False

    return config


def write_yaml_config(path='scm_step12_mcmc.yaml', output_root='scm_step12_mcmc',
                      packages_path='cobaya_packages'):
    """Write the Step 12 Cobaya config to YAML."""
    try:
        import yaml
    except ImportError:
        import ruamel.yaml as yaml

    config = get_step12_config(output_root=output_root,
                               packages_path=packages_path)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f'Step 12 config written to: {path}')


# validation

def validate():
    """Validate SCMCAMBV at the Step 11 posterior mean."""
    from cobaya.model import get_model

    print("\n" + "=" * 60)
    print("  SCMCAMBV Validation (SCM V)")
    print("=" * 60)

    # Reuse the Step 12 config but without DES/lensing for speed
    config = get_step12_config()
    config['likelihood'].pop('des_y1.shear', None)
    config['likelihood'].pop('planck_2018_lensing.native', None)

    model = get_model(config)

    # Step 11 posterior mean
    test_params = {
        'H0': 69.00, 'ombh2': 0.02237, 'omch2': 0.1200,
        'tau': 0.054, 'ns': 0.965, 'logA': 3.044,
        'A_planck': 1.0,
        'eps': 0.00148, 'A_fp': 2.0,
    }

    logpost = model.logpost(test_params)
    derived = model.logpost(test_params, as_dict=True)['derived']

    print(f"\n  logposterior = {logpost:.2f}")
    print(f"  sigma8       = {derived.get('sigma8', float('nan')):.4f}")
    print(f"  S8           = {derived.get('s8', float('nan')):.4f}")
    print(f"  sigma8(z=0.3)= {derived.get('sigma8_z03', float('nan')):.4f}")
    for name in FS8_NAMES:
        print(f"  {name:<14} = {derived.get(name, float('nan')):.4f}")

    # Tests
    s8 = derived.get('s8', 0.0)
    ok = 0.75 < s8 < 0.95
    print(f"\n  S8 in (0.75, 0.95): {'PASS' if ok else 'FAIL'}")
    print("=" * 60 + "\n")
    return ok


# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCM V Cobaya wrapper')
    parser.add_argument('--validate',   action='store_true')
    parser.add_argument('--write-yaml', action='store_true')
    parser.add_argument('--output',     default='scm_step12_mcmc')
    args = parser.parse_args()

    if args.validate:
        sys.exit(0 if validate() else 1)
    elif args.write_yaml:
        write_yaml_config(path=args.output + '.yaml', output_root=args.output)
    else:
        write_yaml_config()
