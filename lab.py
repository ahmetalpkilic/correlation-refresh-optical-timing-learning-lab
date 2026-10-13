"""A regime change, fixed-cost refresh policies, and fresh optical-delay risk."""
from pathlib import Path
import json
import numpy as np
import optics as o
import clock_model as c
import repeat_model as r
from characterization import ar_sequence, fit_rho

OLD_RHO = .4
LEVELS = (0., .4, .7, .9)
SEED = 20261013
# Old characterization is a common sunk cost: 2*80 pairs. Fresh pair counts
# and repeat counts below exhaust the same 640-pair lifetime budget.
POLICIES = {'stale': (0, 12), 'refresh40': (40, 10), 'refresh80': (80, 8)}


def transfer_factor(m, assumed, actual):
    """Conditional mean covariance scale: E[S | assumed] = factor * C."""
    return float(np.trace(r.residual_precision(m, assumed) @ r.correlation(m, actual))/(m-1))


def estimate_policy(repeats, old_rhos, fresh_rhos, name):
    """Feasible policy: never receives the true current rho or evaluation data."""
    n, m = POLICIES[name]
    used = old_rhos if n == 0 else fresh_rhos[n]
    return r.estimate(repeats[:, :m], np.asarray(used)[r.LABELS])


def experiment(trials=300, seed=SEED):
    if not isinstance(trials, int) or trials < 2:
        raise ValueError('trials must be an integer >=2')
    image_rng, eval_rng, cal_rng, old_rng, fresh_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    cov = c.covariances(4.)
    means = np.column_stack((c.SYNC, c.RATE*c.SYNC+c.OFFSET))
    center = o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers = [o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,33)) for _ in range(trials)]
    evaluation = eval_rng.normal(size=(trials,40,2))
    calibration = cal_rng.normal(size=(trials,40,12,2))
    old_normal = old_rng.normal(size=(trials,2,80,2))
    fresh_normal = fresh_rng.normal(size=(trials,2,80,2))
    # Old characterization happens once, before the regime change. Reusing
    # these estimates across scenarios couples comparisons, not data roles.
    old_fits = np.array([[fit_rho(ar_sequence(old_normal[k,g], [0.,OLD_RHO][g], cov[g], means[g]))
                          for g in range(2)] for k in range(trials)])
    rows = []
    for rho in LEVELS:
        actual = np.array([0.,rho]); event_rhos = actual[r.LABELS]
        names = ['oracle', *POLICIES, 'known12', 'known10', 'known8']
        errors = {name: [] for name in names}
        fits = {'stale':old_fits.tolist(), 'refresh40':[], 'refresh80':[]}
        factors = {name:[] for name in POLICIES}
        for k,u in enumerate(centers):
            z = means+np.einsum('nij,nj->ni', np.linalg.cholesky(cov), evaluation[k])
            repeats = r.draw_repeats(calibration[k],cov,event_rhos,means)
            fresh_sequences = [ar_sequence(fresh_normal[k,g],actual[g],cov[g],means[g]) for g in range(2)]
            fresh = {n: np.array([fit_rho(seq[:n]) for seq in fresh_sequences]) for n in (40,80)}
            estimates = {'oracle':cov}
            for name,(n,m) in POLICIES.items():
                used = old_fits[k] if n == 0 else fresh[n]
                if n: fits[name].append(used.tolist())
                factors[name].append([transfer_factor(m,float(used[g]),float(actual[g])) for g in range(2)])
                estimates[name] = estimate_policy(repeats,old_fits[k],fresh,name)
                estimates[f'known{m}'] = r.estimate(repeats[:,:m],event_rhos)
            for name,s in estimates.items():
                fit = c.calibrate(*z.T,s)
                errors[name].append(1000*(c.delay_from_center(u,fit)-c.DELAY))
        paired = {}
        for a,b in [('refresh40','stale'),('refresh80','stale'),('refresh80','refresh40'),
                    ('stale','known12'),('refresh40','known10'),('refresh80','known8')]:
            d = np.square(errors[a])-np.square(errors[b])
            paired[a+' minus '+b] = dict(difference_ms2=float(d.mean()),mcse_ms2=float(d.std(ddof=1)/np.sqrt(trials)))
        rows.append(dict(current_rho=rho,
            methods={name:dict(rmse_ms=float(np.sqrt(np.mean(np.square(e)))),bias_ms=float(np.mean(e)),sd_ms=float(np.std(e,ddof=1))) for name,e in errors.items()},
            rho_estimates={name:dict(mean=np.mean(x,axis=0).tolist(),sd=np.std(x,axis=0,ddof=1).tolist(),boundary_fraction=np.mean(np.abs(x)>=.95-1e-7,axis=0).tolist()) for name,x in fits.items()},
            conditional_scale_mean={name:np.mean(x,axis=0).tolist() for name,x in factors.items()},
            exact_stale_scale=transfer_factor(12,OLD_RHO,rho), paired_risk=paired))
    return dict(seed=seed,trials=trials,old_rho=OLD_RHO,total_pairs=640,
                allocations={name:dict(old_pairs=160,fresh_pairs=2*n,repeat_pairs=40*m,repeats=m) for name,(n,m) in POLICIES.items()},
                fresh_evaluation_pairs=40,sigma_px=o.SIGMA,center_sd_px=float(np.std(centers,ddof=1)),failed_fits=0,rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = experiment(); out = Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes = plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for name in POLICIES:
        axes[0].plot(LEVELS,[row['conditional_scale_mean'][name][1] for row in result['rows']],'o-',label=name)
        axes[1].plot(LEVELS,[row['methods'][name]['rmse_ms'] for row in result['rows']],'o-',label=name)
    axes[0].axhline(1,color='k',ls=':'); axes[0].set(title='Noisier class: covariance scale',ylabel='Mean conditional E[S] / C')
    axes[1].plot(LEVELS,[row['methods']['oracle']['rmse_ms'] for row in result['rows']],'k:',label='oracle')
    axes[1].set(title='Fresh optical-delay measurement',ylabel='Delay RMSE (ms)')
    for name in ('refresh40 minus stale','refresh80 minus stale','refresh80 minus refresh40'):
        axes[2].errorbar(LEVELS,[row['paired_risk'][name]['difference_ms2'] for row in result['rows']],yerr=[2*row['paired_risk'][name]['mcse_ms2'] for row in result['rows']],fmt='o-',capsize=3,label=name)
    axes[2].axhline(0,color='k',ls=':'); axes[2].set(title='Paired risk difference ±2 MCSE',ylabel='MSE difference (ms²)')
    for ax in axes:
        ax.set(xlabel='Current rho (old rho = 0.4)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__ == '__main__': demo()
