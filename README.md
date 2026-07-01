# The Subtractive Cosmological Model

Dark energy as an emergent consequence of dark matter depletion, not a vacuum energy constant.

---

## The basic idea

The standard cosmological model treats dark energy as an unexplained constant — it just is, and its value happens to be what it is. The Subtractive Cosmological Model (SCM) proposes instead that dark energy is sourced by dark matter progressively condensing into Bose-Einstein condensate (BEC) halos throughout cosmic history.

As structure forms, dark matter falls into gravitational wells and undergoes BEC condensation at some critical redshift z_c. This removes field-state dark matter from the background, and the energy released from that phase transition contributes to what we observe as dark energy. The mechanism is genuinely subtractive: dark energy grows *because* dark matter is being depleted from the field.

The sequestration fraction f_seq(z) tracks how much DM has condensed into halos at each epoch. It's computed from the Sheth-Tormen halo mass function using CAMB power spectra, and sits around 57% today. The dark energy density takes the form

```
ρ_DE(z) = ρ_Λ,0 [ (1-ε)·ρ_RE + ε · F(z) · Θ(z; z_c) ]
```

where F(z) = (1+z)⁶ · [(1 − f_seq(z)) / (1 − f_seq,0)]² is the field pressure shape function, Θ is a smooth tanh cutoff at z_c, and ε is the FP fraction — how much of the DE comes from the BEC transition versus the residual vacuum term.

This gives a time-varying equation of state w(z) that looks like ΛCDM at low redshift but departs significantly near z_c. The CPL fit at the Step 10 posterior mean gives w₀ ≈ −0.57, w_a ≈ −1.88, which falls close to where DESI 2024 is pointing.

The boson mass required for BEC condensation at z_c ≈ 2.4 works out to m ≈ 22–23 meV/c².

---

## Code structure

The development follows a chain of papers (SCM I through VI), and the scripts map roughly onto that progression:

**Core physics**

- `SCM_camb_de.py` — Background: builds the SCM dark energy density and w(z), implements the `SCMDarkEnergy` class (a CAMB `DarkEnergyPPF` subclass), and computes the self-consistent sequestration fraction using the linear growth factor from CAMB P(k,z).
- `SCM_camb_de_IV.py` — The full self-consistency loop: iterates the inner loop (E(z) ↔ f_seq) and outer loop (f_seq from CAMB P(k,z)) until convergence. This is the production CAMB runner used by the Cobaya wrappers from Step 10 onwards.
- `SCM_perturbations.py` — Linear growth factor D(z) and f·σ₈ predictions from the growth ODE with the SCM background. Also handles the BEC quantum Jeans scale.
- `SCM_hmf.py` — Tinker 2008 HMF at Δ=500c, predicted cluster counts dN/dz, and comparison to Planck PSZ2.

**Paper numerics**

- `SCM_II_numerics.py` — SCM II: corrected field density (expansion factor), Sheth-Tormen f_seq(z), w(z) in both RE and FP formulations, CPL fit and DESI comparison.
- `SCM_III_numerics.py` — SCM III: BEC condensation mass m(z_c) as a function of redshift, the hybrid DE density with the tanh cutoff, CPL parameterization of the full SCM w(z).
- `SCM_nonlinear.py` — SCM V: halofit non-linear P(k,z), cosmic shear convergence C_ℓ (Limber), CMB lensing potential, and S8 from the posterior chain.

**MCMC / Cobaya wrappers**

- `SCM_cobaya.py` — Basic (ε, z_c) parameterization, Steps 1–6.
- `SCM_cobaya_repar.py` — Reparameterized to (ε, A_fp) where A_fp = ε·F(z_c), breaking the ε–z_c flat direction. Step 7+.
- `SCM_cobaya_IV.py` — Adds f·σ₈ at eBOSS redshifts as derived parameters. Step 11.
- `SCM_cobaya_V.py` — Adds halofit and S8 derived parameters. Step 12.
- `SCM_cobaya_VI.py` — Adds Planck PSZ2 cluster likelihood compatibility. Step 13.
- `SCM_planck_sz_likelihood.py` — Cobaya likelihood class for Planck PSZ2 cluster counts using the Tinker HMF.

**Analysis and figures**

- `scm_step11_analysis.py`, `scm_step12_analysis.py`, `scm_step13_analysis.py` — Chain loading, summary tables, and publication figures for each MCMC step.
- `generate_figures.py` — All paper figures: R−1 convergence, parameter posteriors, w(z) evolution, BEC mass constraint, etc.

---

## Results so far

The chain has gone through 13 steps, progressively adding physics and fixing convergence problems. The main milestones:

- **Steps 1–6**: Basic (ε, z_c) parameterization. R−1 never converged below 0.13 due to a near-flat ε–z_c direction (any z_c is acceptable when ε → 0).
- **Step 7**: Reparameterized to A_fp = ε·F(z_c). R−1 dropped to 0.036 — degeneracy broken.
- **Steps 8–9**: Discovered bimodal posterior (Mode A: z_c ~ 2.2, Mode B: z_c > 3). Mode B is physically disfavoured — at z_c = 3.6, the FP term produces ~46% of the DE budget at z = 3.6, which Planck CMB rules out.
- **Step 10**: First fully converged run (R−1 = 0.027). Mode B excluded by z_c prior. **Posterior: ε = 0.00159 ± 0.00091, z_c = 2.427 ± 0.382, w₀ = −0.570 ± 0.357, w_a = −1.883 ± 1.218, H₀ = 69.20 ± 0.61.**
- **Step 11**: Added eBOSS f·σ₈. Consistent with Step 10.
- **Step 12**: Added halofit and S8 from DES/KiDS. σ₈ slightly reduced.
- **Step 13**: Added Planck PSZ2 cluster likelihood. **ε → 0, σ₈ = 0.806, S8 = 0.805, mass bias b = 0.188.** R−1 converged (v4 chain).

The model does not resolve the Hubble tension — H₀ sits at 68.9–69.2 across all steps, not 73. What it does do is produce a time-varying w(z) that matches the DESI 2024 preference for dynamical dark energy, without adding any new dark energy field — the energy budget is already there in the dark matter sector.

---

## Dependencies

```
python >= 3.9
numpy
scipy
matplotlib
camb
cobaya
```

CAMB and Cobaya can be installed via pip. The Planck 2018 likelihoods (plik-lite TTTEEE, lowl TT/EE) and DESI 2024 BAO likelihood need to be installed through Cobaya's package manager:

```bash
cobaya-install planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lowl.TT planck_2018_lowl.EE bao.desi_2024_bao_all sn.pantheonplus --path cobaya_packages
```

---

## Running the numerics

Each paper script can be run standalone to reproduce the figures:

```bash
python SCM_II_numerics.py     # SCM II: f_seq, w(z), CPL fit, DESI comparison
python SCM_III_numerics.py    # SCM III: BEC mass, hybrid DE, w0-wa plane
python SCM_perturbations.py   # growth factor, f*sigma8, Jeans scale
python SCM_nonlinear.py       # non-linear P(k), shear C_ell, CMB lensing
python SCM_hmf.py             # HMF, cluster counts, S8 constraints
```

To reproduce the Step 13 MCMC:

```bash
python SCM_cobaya_VI.py --validate    # check config
python SCM_cobaya_VI.py --yaml scm_step13.yaml
cobaya-run scm_step13.yaml
python scm_step13_analysis.py
```

Chains take roughly 10–12 hours on a modern desktop with 4 parallel chains.

---

## Status

This is active research — the code is functional but has rough edges. The physics is developed iteratively and the scripts retain the step numbering from the development history. Papers are in preparation.

SCM VI (cluster likelihood + sigma_8 tension) is the current focus. The next open questions are whether the BEC sound speed modifies the matter power spectrum at observable scales, and whether 21cm tomography around z_c could directly detect the condensation transition.
