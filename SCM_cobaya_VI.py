import sys
import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SCM_cobaya_V import (
    SCMCAMBV, SCM_V_DERIVED_NAMES, SCM_V_NEW_NAMES, SCM_V_NEW_LATEX,
    get_step12_config,
)
from SCM_cobaya_IV import FS8_NAMES, FS8_LATEX, SCM_IV_DERIVED_NAMES
from SCM_cobaya_repar import Z_C_MIN, Z_C_MAX, EPS_MIN

# derived param names

SCM_VI_NEW_NAMES  = ['s8_cluster', 'sigma8_z01', 'sigma8_z05', 'sigma8_z08']
SCM_VI_NEW_LATEX  = [r'S_{8,\rm cl}', r'\sigma_8(z=0.1)',
                     r'\sigma_8(z=0.5)', r'\sigma_8(z=0.8)']
SCM_VI_DERIVED_NAMES = SCM_V_DERIVED_NAMES | frozenset(SCM_VI_NEW_NAMES)

# Cluster likelihood redshifts
_CL_REDSHIFTS = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# SCMCAMBVI

class SCMCAMBVI(SCMCAMBV):
    """
    SCM VI Cobaya Theory class.

    Extends SCMCAMBV (Step 12) with:
      - Linear Pk at cluster-relevant redshifts for PlanckSZClusters likelihood
      - Derived: s8_cluster = sigma8 * (Omega_m/0.3)^0.3
      - Derived: sigma8 at z = 0.1, 0.5, 0.8
    """

    # Tell Cobaya's param-assignment that this theory provides these derived params
    provides = ['s8_cluster', 'sigma8_z01', 'sigma8_z05', 'sigma8_z08']

    # filter SCM-specific derived params from CAMB

    def _get_derived_output(self, intermediates):
        saved = self.output_params
        self.output_params = [p for p in saved if p not in frozenset(SCM_VI_NEW_NAMES)]
        try:
            result = super()._get_derived_output(intermediates)
        finally:
            self.output_params = saved
        return result

    # extra CAMB requirements for cluster likelihood

    def get_camb_requirements(self):
        reqs = super().get_camb_requirements()
        # Add linear Pk at cluster redshifts (non-linear already added by V)
        reqs.setdefault('Pk_interpolator', {})
        reqs['Pk_interpolator'].update({
            'z': _CL_REDSHIFTS,
            'k_max': 20.0,
            'nonlinear': False,
            'vars_pairs': [('delta_tot', 'delta_tot')],
        })
        # sigma8 at several redshifts
        for zz in [0.1, 0.5, 0.8]:
            reqs.setdefault('sigma8_z', [])
            if isinstance(reqs['sigma8_z'], list) and zz not in reqs['sigma8_z']:
                reqs['sigma8_z'].append(zz)
        return reqs

    # calculate: SCMCAMBV + cluster-derived params

    def calculate(self, state, want_derived=True, **params_values_dict):
        super().calculate(state, want_derived=want_derived, **params_values_dict)

        if not want_derived:
            return
        if state.get('derived') is None:
            state['derived'] = {}

        sigma8 = state['derived'].get('sigma8', float('nan'))
        s8     = state['derived'].get('s8', float('nan'))

        # Omega_m from sampled params (same logic as SCMCAMBV)
        om_m = float('nan')
        try:
            ombh2 = float(params_values_dict.get('ombh2', 0.0))
            omch2 = float(params_values_dict.get('omch2', 0.0))
            H0    = float(params_values_dict.get('H0', 70.0))
            h     = H0 / 100.0
            om_m  = (ombh2 + omch2) / h**2
        except Exception:
            pass

        # s8_cluster = sigma8 * (Omega_m / 0.3)^0.3
        try:
            s8_cluster = sigma8 * (om_m / 0.3)**0.3
        except Exception:
            s8_cluster = float('nan')

        # sigma8 at z = 0.1, 0.5, 0.8
        def _get_s8z(z):
            try:
                val = float(self.provider.get_sigma8_z(z))
                if np.isfinite(val) and 0.05 < val < 2.0:
                    return val
            except Exception:
                pass
            return sigma8  # fallback

        state['derived']['s8_cluster']  = float(s8_cluster)
        state['derived']['sigma8_z01']  = _get_s8z(0.1)
        state['derived']['sigma8_z05']  = _get_s8z(0.5)
        state['derived']['sigma8_z08']  = _get_s8z(0.8)


# Step 13 YAML config

def get_step13_config(output='scm_step13_mcmc'):
    """
    Build Cobaya config dict for Step 13.

    Extends Step 12 dataset with Planck PSZ2 cluster number counts.
    Adds 2 nuisance params: mass_bias_b, log_Y_star (23 free params total).
    """
    cfg = get_step12_config()

    # Upgrade theory class to SCMCAMBVI
    cfg['theory'] = {
        'SCM_cobaya_VI.SCMCAMBVI': None,
    }

    # Add cluster likelihood
    cfg.setdefault('likelihood', {})
    cfg['likelihood']['SCM_planck_sz_likelihood.PlanckSZClusters'] = None

    # New derived params (nuisance params are auto-registered by PlanckSZClusters.params)
    cfg['params']['s8_cluster']  = {'derived': True, 'latex': r'S_{8,\rm cl}'}
    cfg['params']['sigma8_z01']  = {'derived': True, 'latex': r'\sigma_8(0.1)'}
    cfg['params']['sigma8_z05']  = {'derived': True, 'latex': r'\sigma_8(0.5)'}
    cfg['params']['sigma8_z08']  = {'derived': True, 'latex': r'\sigma_8(0.8)'}

    # Use Step 12 covariance matrix as warm start (extended for 2 new params)
    cfg['sampler'] = {
        'mcmc': {
            'drag':          True,
            'oversample_power': 0.4,
            'proposal_scale': 1.9,
            'Rminus1_stop':  0.02,
            'max_samples':   200000,
            'covmat':        'scm_step12_mcmc.covmat',
            'covmat_params': None,   # auto-extend for new params
        }
    }

    # Sampler output
    cfg['output'] = output

    # Python path so cobaya-run finds our custom modules
    cfg['python_path'] = _HERE

    # Debug-level logging
    cfg['debug'] = False

    return cfg


def write_yaml_config(output='scm_step13_mcmc', path=None):
    """Write Step 13 YAML config file."""
    import yaml

    cfg = get_step13_config(output=output)
    if path is None:
        path = os.path.join(_HERE, output + '.yaml')

    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'  Written: {path}')
    return path


# validate and write YAML

def _validate():
    """Quick sanity check: build config and test theory class instantiation."""
    from cobaya.model import get_model

    print('\n  [VALIDATE] Building Step 13 Cobaya model...')
    cfg = get_step13_config(output='_scm_step13_test', resume=False)

    # Point theory to this module
    info = dict(cfg)
    try:
        model = get_model(info)
        lp, derived = model.logposterior({})
        print(f'  logposterior = {lp:.2f}')
        print(f'  s8_cluster   = {derived.get("s8_cluster", float("nan")):.4f}')
        print('  [VALIDATE] PASS\n')
    except Exception as exc:
        print(f'  [VALIDATE] FAIL: {exc}\n')
        raise


def main():
    parser = argparse.ArgumentParser(description='SCM VI Cobaya theory class')
    parser.add_argument('--validate',   action='store_true')
    parser.add_argument('--write-yaml', action='store_true')
    parser.add_argument('--output', default='scm_step13_mcmc')
    args = parser.parse_args()

    if args.validate:
        _validate()
    if args.write_yaml:
        write_yaml_config(output=args.output)


if __name__ == '__main__':
    main()
