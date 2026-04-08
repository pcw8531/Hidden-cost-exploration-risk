"""
Figure 5 simulation: Agent-level data for fold bifurcation analysis.

Computes the normalized hub-peripheral protection gap (strategy diversity)
at four (pe, pr) conditions to locate the observed states on the
bifurcation diagram.

Parameters: BA(100,10), seed=42, s=10, T=200,000, conditional regime.
Estimated runtime: ~10 min on local machine (4 conditions x ~2.5 min each).

Output: fig5_data.npz
"""

import numpy as np
import networkx as nx
import time

# Network
G = nx.barabasi_albert_graph(100, 10, seed=42)
cent_dict = nx.eigenvector_centrality(G, max_iter=1000)
cent_raw = np.array([cent_dict[i] for i in range(100)])
cent = (cent_raw - cent_raw.min()) / (cent_raw.max() - cent_raw.min())
n = 100
adj_list = [np.array(list(G.neighbors(i)), dtype=np.int32) for i in range(n)]
q75, q25 = np.percentile(cent, 75), np.percentile(cent, 25)
hub = cent >= q75
peri = cent <= q25


def run_agent(pe, pr, T=200000):
    """Run one simulation, return stationary-state agent-level averages."""
    capital = np.ones(n)
    capital_m = np.ones(n)
    s0 = np.zeros(n)
    s1 = np.zeros(n)
    failure = np.zeros(n, dtype=bool)
    T_half = T // 2

    acc_fp = np.zeros(n)
    acc_fail = np.zeros(n)
    acc_cap = np.zeros(n)

    for t in range(T):
        # Failure potential origination
        fp_pot = np.random.random(n) < 0.1

        # Conditional propagation
        if failure.any():
            for i in np.where(failure)[0]:
                nb = adj_list[i]
                if len(nb) > 0:
                    fp_pot[nb[np.random.random(len(nb)) < 0.3]] = True

        # Failure realization
        failure[:] = False
        fp_eff = np.clip(s0 + s1 * cent, 0, 0.9)
        pp = 1.0 / (1.0 + 0.05 / np.maximum(fp_eff * capital, 1e-12))
        new_fails = fp_pot & (np.random.random(n) > pp)
        failure[new_fails] = True
        capital[new_fails] = 0.0

        # Capital update
        fp_eff = np.clip(s0 + s1 * cent, 0, 0.9)
        capital = 1 + (1 - 0.1 - fp_eff) * capital
        capital_m = 0.99 * capital + 0.01 * capital_m

        # Accumulate stationary-state values
        if t >= T_half:
            acc_fp += fp_eff
            acc_fail += failure.astype(float)
            acc_cap += capital_m

        # Imitation (vectorized)
        imitators = np.random.random(n) < pr
        if imitators.any():
            idx = np.where(imitators)[0]
            rm = np.random.randint(0, n, size=len(idx))
            same = rm == idx
            while same.any():
                rm[same] = np.random.randint(0, n, size=same.sum())
                same = rm == idx
            dcm = capital_m[rm] - capital_m[idx]
            pi = 1.0 / (1.0 + np.exp(np.clip(-10 * dcm, -500, 500)))
            c0 = np.random.random(len(idx)) < pi
            s0[idx[c0]] = s0[rm[c0]]
            c1 = np.random.random(len(idx)) < pi
            s1[idx[c1]] = s1[rm[c1]]

        # Exploration
        e0 = np.random.random(n) < pe
        s0[e0] += np.random.normal(0, 0.001, size=e0.sum())
        e1 = np.random.random(n) < pe
        s1[e1] += np.random.normal(0, 0.001, size=e1.sum())

    steps = T - T_half
    return acc_fp / steps, acc_fail / steps, acc_cap / steps


# Run four conditions
print("Figure 5: Agent-level simulation for bifurcation analysis")
print("=" * 60)

results = {}
for pr in [0.1, 0.9]:
    for pe in [0.1, 0.9]:
        t0 = time.time()
        fp, fail, cap = run_agent(pe=pe, pr=pr)
        div = fp[hub].mean() - fp[peri].mean()
        results[(pr, pe)] = {
            'diversity': div, 'fail': fail.mean(), 'cm': cap.mean()
        }
        print(f"  pr={pr}, pe={pe}: div={div:.4f}, fail={fail.mean():.4f}, "
              f"cm={cap.mean():.3f} ({time.time()-t0:.0f}s)")

# Save
np.savez('fig5_data.npz',
         pr01_pe01_div=results[(0.1, 0.1)]['diversity'],
         pr01_pe01_fail=results[(0.1, 0.1)]['fail'],
         pr01_pe01_cm=results[(0.1, 0.1)]['cm'],
         pr01_pe09_div=results[(0.1, 0.9)]['diversity'],
         pr01_pe09_fail=results[(0.1, 0.9)]['fail'],
         pr01_pe09_cm=results[(0.1, 0.9)]['cm'],
         pr09_pe01_div=results[(0.9, 0.1)]['diversity'],
         pr09_pe01_fail=results[(0.9, 0.1)]['fail'],
         pr09_pe01_cm=results[(0.9, 0.1)]['cm'],
         pr09_pe09_div=results[(0.9, 0.9)]['diversity'],
         pr09_pe09_fail=results[(0.9, 0.9)]['fail'],
         pr09_pe09_cm=results[(0.9, 0.9)]['cm'])

print("\nSaved: fig5_data.npz")
