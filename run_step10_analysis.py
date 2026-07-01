import os, warnings, numpy as np
warnings.filterwarnings('ignore')

from getdist import loadMCSamples
import getdist.plots as gdp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scm_step10_mcmc')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'step10_analysis_summary.txt')

s = loadMCSamples(ROOT, settings={'ignore_rows': 0.3})
pdict = s.getParams().__dict__

lines = []
def w(x=''):
    lines.append(x)
    print(x)

w('Step 10 Analysis Summary')
w('========================')
w(f'Chain:   scm_step10_mcmc.1.txt')
w(f'Config:  scm_step10_mcmc.yaml')
w(f'Post-burn samples: {s.numrows}  (30% burnin removed)')
w()

params = ['eps', 'A_fp', 'zc_scm', 'w0_scm', 'wa_scm', 'H0', 'sigma8']
w('=== POSTERIOR SUMMARY ===')
w('%-14s %10s %10s %10s %10s %10s %10s' % (
    'Param', 'Mean', 'Std', '68CI_lo', '68CI_hi', '95CI_lo', '95CI_hi'))
w('-'*78)
for p in params:
    v = pdict[p]
    w('%-14s %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f' % (
        p, np.mean(v), np.std(v),
        np.percentile(v,16), np.percentile(v,84),
        np.percentile(v,2.5), np.percentile(v,97.5)))
w()

eps  = pdict['eps']
Afp  = pdict['A_fp']
zc   = pdict['zc_scm']
w0   = pdict['w0_scm']
w()
w('=== BOUNDARY DIAGNOSTICS ===')
w('Samples at Z_C_MAX (zc>2.99): %d/%d = %.2f%%' % (
    np.sum(zc > 2.99), len(zc), 100*np.mean(zc > 2.99)))
w('Samples at Z_C_MIN (zc<0.31): %d/%d = %.2f%%' % (
    np.sum(zc < 0.31), len(zc), 100*np.mean(zc < 0.31)))
w('Samples at A_fp > 18: %d/%d = %.2f%%' % (
    np.sum(Afp > 18), len(Afp), 100*np.mean(Afp > 18)))
w('eps 95th pct: %.5f' % np.percentile(eps, 95))
w()
w('eps pctls: 5th=%.5f  16th=%.5f  50th=%.5f  84th=%.5f  95th=%.5f' % (
    np.percentile(eps,5), np.percentile(eps,16),
    np.percentile(eps,50), np.percentile(eps,84), np.percentile(eps,95)))
w('zc  pctls: 5th=%.3f  16th=%.3f  50th=%.3f  84th=%.3f  95th=%.3f' % (
    np.percentile(zc,5), np.percentile(zc,16),
    np.percentile(zc,50), np.percentile(zc,84), np.percentile(zc,95)))
w('A_fp pctls: 5th=%.3f 16th=%.3f  50th=%.3f  84th=%.3f  95th=%.3f' % (
    np.percentile(Afp,5), np.percentile(Afp,16),
    np.percentile(Afp,50), np.percentile(Afp,84), np.percentile(Afp,95)))
w()
w('=== CORRELATIONS ===')
for (p1,p2) in [('eps','A_fp'),('eps','zc_scm'),('A_fp','zc_scm'),
                ('eps','w0_scm'),('zc_scm','w0_scm')]:
    w('corr(%s, %s) = %.4f' % (p1, p2, np.corrcoef(pdict[p1], pdict[p2])[0,1]))
w()
w('=== BEC MASS ESTIMATE ===')
# m_BEC ~ (1+z_c)^(3/5); at z_c=2.20, m=22 meV (from Step 7 posterior)
zc_ref, m_ref = 2.20, 22.0  # meV/c^2
m_bec = m_ref * ((1 + zc) / (1 + zc_ref))**(3/5)
w('m_BEC (meV/c^2): mean=%.1f +/- %.1f   [16th,84th]=[%.1f, %.1f]' % (
    np.mean(m_bec), np.std(m_bec),
    np.percentile(m_bec,16), np.percentile(m_bec,84)))
w()
w('=== COMPARISON ACROSS STEPS ===')
w('Step | eps UL (95%)  | R-1 min | Note')
w('-----|---------------|---------|------------------------------')
w('  6  | 0.0034        | 0.13    | eps-zc flat ridge')
w('  7  | 0.0028        | 0.036   | A_fp prior wall at 5')
w('  8  | 0.0029        | 0.035   | 34.5%% clipped at zc=3')
w('  9  | 0.0029        | 0.798*  | Bimodal; not converged')
w(' 10  | %.4f        | TBD     | Z_C_MAX=3 rejection; converged!' % np.percentile(eps,95))

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))

print('\nSaved to', OUT)
