"""
Figure 4 of the manuscript, submitted Figure 3.
The cost of conservative strategies in scale-free networks.

Sweeps the imitation probability pr from 0.1 to 0.9 at two exploration levels,
pe = 0.1 and pe = 0.9, under conditional propagation on BA scale-free networks,
and adds the three intermediate exploration levels pe = 0.3, 0.5 and 0.7 at
pr = 0.1 for the ternary row.

Revision protocol: R = 10 independent dynamic realisations per condition at
T = 1,000,000 steps, in place of R = 3 at T = 100,000. Per-realisation values
and 95% t half-widths are saved next to the standard deviations. The exploration
advantage is reported with a bootstrap 95% interval. The ternary row uses
simulated stationary states and the simulated transient rather than interpolation.

The inner simulation is the vectorised implementation of the same model and the
same step order as the loop code in core/. Realisations of one condition run in
parallel with joblib.

Outputs
    fig2_data.npz           pr sweep. Keys pr_values, cm_pe01, cm_pe09,
                            cm_pe01_std, cm_pe09_std, f_pe01, f_pe09,
                            f_pe01_std, f_pe09_std, with *_ci (95% t
                            half-widths), *_runs (per realisation) and fp_*.
    fig3_ternary_data.npz   five pe levels at pr = 0.1. Stationary c_m, f, f_p,
                            protection cost and strategy diversity, with the
                            population trajectories and the 100-step block
                            series used by the attractor landscape and the
                            ternary row.

Runtime: 210 runs at about 2 min each, near 2 h with N_JOBS = 4.

Usage
    python3 fig3_sim.py
"""
import networkx as nx
import numpy as np
import time
import os
from scipy import stats
from joblib import Parallel, delayed

# =============================================================================
# FIXED PARAMETERS (identical to the submitted version except T and R)
# =============================================================================
n = 100
m = 10
T = 1_000_000
T_half = T // 2

pmax = 1.0
cp = 0.05
pn = 0.1
pl = 0.3
cin = 1.0
fm = 0.1
memory = 0.99
s = 10          # selection intensity (manuscript specification)
mu = 0.0
sigma = 0.001   # exploration noise SD

# Sweep parameters
pr_values = np.arange(0.1, 1.0, 0.1)  # 0.1, 0.2, ..., 0.9
pe_values = [0.1, 0.9]
pe_ternary = [0.1, 0.3, 0.5, 0.7, 0.9]  # ternary row, all at pr = 0.1
pr_ternary = 0.1
R = 10                                  # realisations per condition
SEED0 = 30_000                          # dynamics seed = SEED0 + 100*condition + r
N_JOBS = max(1, os.cpu_count() - 1)     # parallel realisations (set 1 for serial)

# Trajectory windows (geometric, for the transient shown in the ternary row)
win_edges = np.unique(np.concatenate([[0], np.round(np.logspace(0, np.log10(T), 121)).astype(int)]))
win_mid = 0.5 * (win_edges[:-1] + win_edges[1:])
BLOCK = 100   # fixed averaging window (steps) of the population time series kept for the attractor landscape

# =============================================================================
# GENERATE NETWORK (shared across all runs for consistency)
# =============================================================================
G = nx.barabasi_albert_graph(n, m, seed=42)
eig_centrality = nx.eigenvector_centrality_numpy(G)
Centrality = np.array(list(eig_centrality.values()))
src = np.array([u for u, v in G.edges()] + [v for u, v in G.edges()], dtype=np.int64)
dst = np.array([v for u, v in G.edges()] + [u for u, v in G.edges()], dtype=np.int64)

print(f"Network: BA({n}, {m}), seed=42")
print(f"Centrality range: {Centrality.min():.4f} to {Centrality.max():.4f}")
print(f"Parameters: s={s}, T={T:,}, R={R}, pl={pl}, pn={pn}, sigma={sigma}")
print(f"Regime: CONDITIONAL (Failure[i] > 0)")
print(f"Sweep: pe = {pe_values}, pr = 0.1 to 0.9; ternary: pe = {pe_ternary} at pr = {pr_ternary}")
print(f"Stationary averaging: final 50% (t = {T_half} to {T}); parallel jobs: {N_JOBS}")
print()

# =============================================================================
# SIMULATION FUNCTION (vectorised; same model and step order as the loop version)
# =============================================================================
def run_simulation(pe, pr, seed):
    """
    One realisation under conditional propagation.
    Returns stationary-state averages (final 50%) of the population means of
    memorised functional capacity c_m, failure rate f, protection level f_p,
    protection cost f_p*C and strategy diversity (std of f_p across agents),
    plus the population trajectory averaged in the geometric windows win_edges
    and in fixed windows of BLOCK steps (columns: c_m, f, f_p, f_p*C, std f_p).
    """
    rng = np.random.default_rng(seed)
    Capital = np.ones(n) * cin
    Capital_m = np.ones(n) * memory
    Strategy_0 = np.zeros(n)
    Strategy_1 = np.zeros(n)
    Failure = np.zeros(n, dtype=bool)

    series = np.zeros((T, 5))   # population means per step: cm, f, fp, fp*C, std of fp across agents

    for t in range(T):
        # Step 1: failure potential origination
        failure_potential = rng.random(n) < pn
        # Step 2: propagation (CONDITIONAL: from failed nodes only)
        active = Failure[src]
        if active.any():
            idx = np.nonzero(active)[0]
            failure_potential[dst[idx[rng.random(len(idx)) <= pl]]] = True
        # Step 3: recovery of last step's failures (failtime = 1)
        Failure[:] = False
        # Step 4: failure realisation with the current protection level
        fp = np.clip(Strategy_0 + Strategy_1 * Centrality, 0.0, 1.0 - fm)
        protection_probability = pmax / (1.0 + cp / np.maximum(fp * Capital, 1e-12))
        R1 = failure_potential & (rng.random(n) > protection_probability)
        Failure[R1] = True
        Capital[R1] = 0.0
        # Step 5: functional capacity update
        cost = fp * Capital
        Capital = cin + (1.0 - fm - fp) * Capital
        Capital_m = memory * Capital + (1.0 - memory) * Capital_m
        series[t, 0] = Capital_m.mean(); series[t, 1] = Failure.mean()
        series[t, 2] = fp.mean(); series[t, 3] = cost.mean(); series[t, 4] = fp.std()
        # Steps 6 and 7: imitation of Strategy_0 and Strategy_1 (Fermi rule on Capital_m)
        for strat in (Strategy_0, Strategy_1):
            im = np.nonzero(rng.random(n) <= pr)[0]
            if len(im):
                rr = rng.integers(0, n, size=len(im)); same = rr == im
                while same.any():
                    rr[same] = rng.integers(0, n, size=same.sum()); same = rr == im
                pi = 1.0 / (1.0 + np.exp(np.clip(-s * (Capital_m[rr] - Capital_m[im]), -500, 500)))
                cpy = rng.random(len(im)) <= pi
                strat[im[cpy]] = strat[rr[cpy]]
        # Steps 8 and 9: exploration of Strategy_0 and Strategy_1
        e0 = rng.random(n) <= pe; Strategy_0[e0] += rng.normal(mu, sigma, size=e0.sum())
        e1 = rng.random(n) <= pe; Strategy_1[e1] += rng.normal(mu, sigma, size=e1.sum())

    stat = series[T_half:]
    traj = np.array([series[a:b].mean(axis=0) for a, b in zip(win_edges[:-1], win_edges[1:])])
    blocks = series[:(T // BLOCK) * BLOCK].reshape(T // BLOCK, BLOCK, 5).mean(axis=1)
    return {
        'cm_mean': stat[:, 0].mean(), 'cm_std': stat[:, 0].std(),
        'f_mean': stat[:, 1].mean(), 'f_std': stat[:, 1].std(),
        'fp_mean': stat[:, 2].mean(), 'pc_mean': stat[:, 3].mean(), 'div_mean': stat[:, 4].mean(),
        'traj': traj, 'blocks': blocks,
    }

# =============================================================================
# RUN ALL CONDITIONS (pr sweep at pe = 0.1, 0.9; then pe = 0.3, 0.5, 0.7 at pr = 0.1)
# =============================================================================
conditions = [(pe, round(pr, 1)) for pe in pe_values for pr in pr_values] + \
             [(pe, pr_ternary) for pe in pe_ternary if pe not in pe_values]
tcrit = stats.t.ppf(0.975, R - 1)
def hw(x): return tcrit * np.std(x, ddof=1) / np.sqrt(len(x))

total_start = time.time()
all_results = {}
for ci, (pe, pr) in enumerate(conditions):
    t0 = time.time()
    runs = Parallel(n_jobs=N_JOBS)(delayed(run_simulation)(pe, pr, SEED0 + 100 * ci + r) for r in range(R))
    all_results[(pe, pr)] = {
        'cm_runs': np.array([x['cm_mean'] for x in runs]),
        'f_runs': np.array([x['f_mean'] for x in runs]),
        'fp_runs': np.array([x['fp_mean'] for x in runs]),
        'pc_runs': np.array([x['pc_mean'] for x in runs]),
        'div_runs': np.array([x['div_mean'] for x in runs]),
        'traj': np.array([x['traj'] for x in runs]),          # (R, windows, 5)
        'blocks': np.array([x['blocks'] for x in runs]) if pr == pr_ternary else None,   # (R, T/BLOCK, 5)
    }
    d = all_results[(pe, pr)]
    print(f"pe={pe:.1f} pr={pr:.1f}: cm = {d['cm_runs'].mean():.4f} (±{hw(d['cm_runs']):.4f}), "
          f"f = {d['f_runs'].mean():.4f} (±{hw(d['f_runs']):.4f}), fp = {d['fp_runs'].mean():.4f}, "
          f"R={R}  [{(time.time()-t0)/60:.1f} min, total {(time.time()-total_start)/60:.1f} min]", flush=True)

# =============================================================================
# EXTRACT AND SAVE (pr sweep; original keys unchanged, 95% half-widths and runs added)
# =============================================================================
def col(pe, key, fn): return np.array([fn(all_results[(pe, round(pr, 1))][key]) for pr in pr_values])
save = dict(pr_values=pr_values, n=n, m=m, T=T, R=R, s=s)
for pe, tag in [(0.1, 'pe01'), (0.9, 'pe09')]:
    for q, key in [('cm', 'cm_runs'), ('f', 'f_runs'), ('fp', 'fp_runs')]:
        save[f'{q}_{tag}'] = col(pe, key, np.mean)
        save[f'{q}_{tag}_std'] = col(pe, key, np.std)      # std across realisations (as before)
        save[f'{q}_{tag}_ci'] = col(pe, key, hw)           # 95% t half-width across realisations
        save[f'{q}_{tag}_runs'] = np.array([all_results[(pe, round(pr, 1))][key] for pr in pr_values])
np.savez('fig2_data.npz', **save)
print("\nData saved to: fig2_data.npz")

# ternary row: five pe levels at pr = 0.1
tern = dict(pe_levels=np.array(pe_ternary), pr=pr_ternary, n=n, m=m, T=T, R=R,
            win_edges=win_edges, win_mid=win_mid, block=BLOCK)
for q, key in [('cm', 'cm_runs'), ('f', 'f_runs'), ('fp', 'fp_runs'), ('pc', 'pc_runs'), ('div', 'div_runs')]:
    tern[q] = np.array([all_results[(pe, pr_ternary)][key].mean() for pe in pe_ternary])
    tern[f'{q}_runs'] = np.array([all_results[(pe, pr_ternary)][key] for pe in pe_ternary])
tern['traj'] = np.array([all_results[(pe, pr_ternary)]['traj'] for pe in pe_ternary])      # (5, R, windows, 5)
tern['blocks'] = np.array([all_results[(pe, pr_ternary)]['blocks'] for pe in pe_ternary])  # (5, R, T/BLOCK, 5)
np.savez('fig3_ternary_data.npz', **tern)
print("Data saved to: fig3_ternary_data.npz")
print(f"\nTotal simulation time: {(time.time()-total_start)/60:.1f} min")
