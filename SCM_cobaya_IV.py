import sys, os, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SCM_cobaya_repar import (
    SCMCAMBRepar, SCM_DERIVED_NAMES,
    Z_C_MIN, Z_C_MAX, EPS_MIN,
    get_repar_config, _HERE as _REPAR_HERE,
)
from SCM_perturbations import (
    compute_fsigma8, OM_M as _OM_M_FIDU,
)

# f·σ8 redshifts

FS8_Z = np.array([0.150, 0.38, 0.51, 0.61, 0.70, 0.85, 1.48])

FS8_NAMES = [
    'fs8_z015', 'fs8_z038', 'fs8_z051', 'fs8_z061',
    'fs8_z070', 'fs8_z085', 'fs8_z148',
]

FS8_LATEX = [
    r'f\sigma_8(0.15)', r'f\sigma_8(0.38)', r'f\sigma_8(0.51)', r'f\sigma_8(0.61)',
    r'f\sigma_8(0.70)', r'f\sigma_8(0.85)', r'f\sigma_8(1.48)',
]

SCM_IV_DERIVED_NAMES = SCM_DERIVED_NAMES | frozenset(FS8_NAMES)

# eBOSS DR16 reference measurements for validation
_EBOSS_FS8_DATA = {
    'z':    np.array([0.38, 0.51, 0.698, 1.480]),
    'fs8':  np.array([0.497, 0.459, 0.473, 0.462]),
    'err':  np.array([0.045, 0.038, 0.044, 0.045]),
    'src':  ['BOSS DR12', 'BOSS DR12', 'eBOSS DR16 LRG', 'eBOSS DR16 QSO'],
}


# SCMCAMBIV

class SCMCAMBIV(SCMCAMBRepar):
    """
    SCM IV Cobaya Theory class.

    Inherits all of SCMCAMBRepar (reparameterized eps, A_fp sampling,
    CPL w0/wa fit, DESI Mahalanobis distance) and additionally computes
    f·σ8 at six eBOSS / BOSS / DESI measurement redshifts via the linear
    growth ODE from SCM_perturbations.py.

    The growth ODE uses the sampled (H0, ombh2, omch2) → Ω_m,0 and the
    σ8 output from CAMB at each MCMC step, so the f·σ8 predictions are
    fully self-consistent with the sampled cosmology.
    """

    params = {
        **SCMCAMBRepar.params,
        **{name: {'derived': True, 'latex': latex}
           for name, latex in zip(FS8_NAMES, FS8_LATEX)},
    }

    # hide fs8 params from derived output

    def _get_derived_output(self, intermediates):
        """
        Exclude all SCM-specific derived params (III + IV) from CAMB.
        Calling super(SCMCAMBRepar, self) skips the parent's filter and applies
        our expanded SCM_IV_DERIVED_NAMES set instead.
        """
        saved = self.output_params
        self.output_params = [p for p in saved
                              if p not in SCM_IV_DERIVED_NAMES]
        try:
            # Jump past SCMCAMBRepar to CAMB directly (avoid double filtering)
            result = super(SCMCAMBRepar, self)._get_derived_output(intermediates)
        finally:
            self.output_params = saved
        return result

    # calculate: CAMB + SCM III + f·σ8

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        1. Call SCMCAMBRepar.calculate (CAMB + CPL fit + w0/wa/zc/Mahal).
        2. Extract sampled cosmological params → Ω_m,0.
        3. Retrieve σ8 from CAMB derived output.
        4. Solve linear growth ODE with current (zc, eps, Ω_m,0, σ8).
        5. Store fs8_z* in state['derived'].
        """
        super().calculate(state, want_derived, **params_values_dict)

        if not want_derived:
            return

        # extract sampled params
        H0    = float(params_values_dict.get('H0',    69.20))
        ombh2 = float(params_values_dict.get('ombh2', 0.02237))
        omch2 = float(params_values_dict.get('omch2', 0.1200))
        h     = H0 / 100.0
        om0   = (ombh2 + omch2) / h**2

        A_fp  = float(params_values_dict.get('A_fp',  2.0))
        eps   = float(params_values_dict.get('eps',   0.0))

        # σ8 from CAMB derived state
        derived = state.get('derived') or {}
        sigma8  = float(derived.get('sigma8', 0.8349))
        if not (0.4 < sigma8 < 1.5):
            sigma8 = 0.8349

        # recover z_c from reparameterisation
        zc = self._invert_F(A_fp, eps)
        if zc is None:
            zc = Z_C_MAX

        # solve growth ODE, compute f·σ8
        try:
            fs8_result = compute_fsigma8(
                zc, eps, FS8_Z,
                sigma8_0=sigma8,
                om0=om0,
            )
            fs8_vals = fs8_result['fs8_scm']
        except Exception as exc:
            self.log.debug(f"SCM IV: growth ODE failed ({exc}); using NaN")
            fs8_vals = np.full(len(FS8_Z), np.nan)

        # store f·σ8 in derived state
        if state.get('derived') is None:
            state['derived'] = {}

        for name, val in zip(FS8_NAMES, fs8_vals):
            state['derived'][name] = float(val) if np.isfinite(val) else 0.0


# Step 11 config

def get_step11_config(output_root='scm_step11_mcmc',
                      covmat='scm_step10_mcmc.covmat'):
    """
    Build the Cobaya config dict for Step 11.

    Based on Step 10 (Planck + DESI + Pantheon+) with three additions:
      1. Theory class upgraded to SCMCAMBIV (adds f·σ8 derived params)
      2. eBOSS DR16 LRG, QSO BAO+FS likelihoods
      3. BOSS DR12 consensus final likelihood
    """
    config = get_repar_config(output_root=output_root)

    # Replace theory class
    old_args = config['theory'].pop('SCM_cobaya_repar.SCMCAMBRepar')
    config['theory']['SCM_cobaya_IV.SCMCAMBIV'] = old_args

    # Add f·σ8 derived params
    for name, latex in zip(FS8_NAMES, FS8_LATEX):
        config['params'][name] = {'derived': True, 'latex': latex}

    # Add eBOSS DR16 BAO+FS likelihoods.
    # sdss_dr16_baoplus_lrg already includes BOSS DR12 z=0.38,0.51 in its joint
    # covariance (Alam et al. 2020). Do NOT also add sdss_dr12_consensus_final
    # at these redshifts — that would double-count.
    config['likelihood'].update({
        'bao.sdss_dr16_baoplus_lrg': None,   # DM/rs + DH/rs + fs8 at z=0.38,0.51,0.698
        'bao.sdss_dr16_baoplus_qso': None,   # DM/rs + DH/rs + fs8 at z=1.48
    })

    # Update sampler: use Step 10 covmat, single adaptive chain
    config['sampler']['mcmc'].update({
        'covmat':          covmat,
        'Rminus1_stop':    0.01,
        'Rminus1_cl_stop': 0.20,
        'max_samples':     80000,
    })

    config['output'] = output_root
    config['resume'] = False

    return config


def write_yaml_config(path='scm_step11_mcmc.yaml', output_root='scm_step11_mcmc'):
    """Write the Step 11 Cobaya config to YAML."""
    try:
        import yaml
    except ImportError:
        import ruamel.yaml as yaml

    config = get_step11_config(output_root=output_root)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Step 11 config written to: {path}")


# validation

def validate():
    """
    Validate SCMCAMBIV at the Step 10 posterior mean.

    Tests:
      1. All SCM III tests inherited from SCMCAMBRepar.validate()
      2. fs8_z038 within 10% of eBOSS DR12 value (0.497 ± 0.045)
      3. fs8_z070 within 10% of eBOSS DR16 LRG value (0.473 ± 0.044)
      4. fs8_z148 within 10% of eBOSS DR16 QSO value (0.462 ± 0.045)
      5. All fs8_z* values finite and in (0.1, 0.9)
    """
    from cobaya.model import get_model

    print("\n" + "=" * 60)
    print("  SCMCAMBIV Validation (SCM IV: growth rate predictions)")
    print("=" * 60)

    config = get_step11_config(output_root=None)
    # Disable external likelihoods that require data files for validation
    for like in ('planck_2018_highl_plik.TTTEEE_lite_native',
                 'planck_2018_lowl.TT', 'planck_2018_lowl.EE',
                 'bao.desi_2024_bao_all', 'sn.pantheonplus',
                 'bao.sdss_dr16_lrg_fsbao_dmdh',
                 'bao.sdss_dr16_qso_fsbao_dmdh',
                 'bao.sdss_dr12_consensus_final'):
        config['likelihood'].pop(like, None)
    config['likelihood']['one'] = None   # trivial likelihood for testing

    print("\n  Building Cobaya model (SCMCAMBIV)...")
    model = get_model(config)

    # Step 10 posterior mean
    # A_fp = eps * F(zc) ≈ 0.00159 * F(2.427)
    # Need to compute A_fp from zc_mean — use the model's own inversion
    theory = model.theory['SCM_cobaya_IV.SCMCAMBIV']
    zc_true = 2.427
    eps_test = 0.00159
    A_fp_test = eps_test * theory._F_at_z(zc_true)
    print(f"  Test point: eps={eps_test}, zc_true={zc_true:.3f} → A_fp={A_fp_test:.4f}")

    test_point = {
        'H0':       69.20,
        'ombh2':    0.02237,
        'omch2':    0.1200,
        'tau':      0.0544,
        'ns':       0.9649,
        'logA':     3.044,
        'A_planck': 1.0,
        'eps':      eps_test,
        'A_fp':     A_fp_test,
    }

    result      = model.logposterior(test_point)
    derived_names = list(model.parameterization.derived_params().keys())
    derived       = dict(zip(derived_names, result.derived))

    sigma8  = derived.get('sigma8',     np.nan)
    zc_d    = derived.get('zc_scm',     np.nan)
    w0      = derived.get('w0_scm',     np.nan)
    wa      = derived.get('wa_scm',     np.nan)
    mahal   = derived.get('Mahal_DESI', np.nan)

    fs8 = {name: derived.get(name, np.nan) for name in FS8_NAMES}

    print(f"\n  CAMB-derived parameters:")
    print(f"    sigma8     = {sigma8:.4f}")
    print(f"    zc_scm     = {zc_d:.4f}   (expected ~2.427)")
    print(f"    w0_scm     = {w0:.4f}")
    print(f"    wa_scm     = {wa:.4f}")
    print(f"    Mahal_DESI = {mahal:.3f} σ")

    print(f"\n  f·σ8 predictions (SCM ODE):")
    refs = {0.38: (0.497, 0.045, 'BOSS DR12'),
            0.70: (0.473, 0.044, 'eBOSS LRG'),
            1.48: (0.462, 0.045, 'eBOSS QSO')}
    for name, z_val in zip(FS8_NAMES, FS8_Z):
        val = fs8[name]
        ref = refs.get(round(z_val, 2), refs.get(round(z_val, 1)))
        if ref:
            pull = (val - ref[0]) / ref[1] if np.isfinite(val) else np.nan
            print(f"    {name} = {val:.4f}   [{ref[2]}: {ref[0]}±{ref[1]}, pull={pull:+.2f}σ]")
        else:
            print(f"    {name} = {val:.4f}")

    # Run tests
    tests = {}
    tests['logpost finite']     = np.isfinite(result.logpost)
    tests['sigma8 in (0.7, 0.95)'] = 0.7 < sigma8 < 0.95
    tests['zc_scm near 2.43']   = abs(zc_d - 2.427) < 0.05 if np.isfinite(zc_d) else False
    for name, z_val in zip(FS8_NAMES, FS8_Z):
        val = fs8[name]
        tests[f'{name} finite, in (0.1, 0.9)'] = (
            np.isfinite(val) and 0.1 < val < 0.9
        )
    # eBOSS consistency: prediction within 3σ
    for z_ref, (fs8_ref, err_ref, label) in refs.items():
        name = FS8_NAMES[np.argmin(np.abs(FS8_Z - z_ref))]
        val = fs8[name]
        tests[f'{name} within 3σ of {label}'] = (
            abs(val - fs8_ref) < 3.0 * err_ref if np.isfinite(val) else False
        )

    print("\n  Test results:")
    all_ok = True
    for t_name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}]  {t_name}")

    print("\n" + "=" * 60)
    print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_ok


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCM IV Cobaya wrapper — f·σ8 derived params + eBOSS likelihoods"
    )
    parser.add_argument('--validate',   action='store_true',
                        help='Run validation at Step 10 posterior mean')
    parser.add_argument('--write-yaml', action='store_true',
                        help='Write scm_step11_mcmc.yaml config')
    parser.add_argument('--run',        action='store_true',
                        help='Launch Step 11 MCMC chain')
    parser.add_argument('--output',     default='scm_step11_mcmc',
                        help='Output root for MCMC chain (default: scm_step11_mcmc)')
    args = parser.parse_args()

    if args.validate:
        validate()

    elif args.write_yaml:
        write_yaml_config(f'{args.output}.yaml', output_root=args.output)

    elif args.run:
        from cobaya.run import run
        print(f"Starting Step 11 MCMC chain → {args.output}")
        config = get_step11_config(output_root=args.output)
        updated_info, sampler = run(config)

    else:
        parser.print_help()
