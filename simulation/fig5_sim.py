"""
Agent-level observation points for the bifurcation analysis.

Produces the data behind Figure 6 of the manuscript, which was Figure 5 of the
submitted version. The revision adds a schematic as Figure 1, so the submitted
Figures 1 to 5 appear as Figures 2 to 6.

Runs the agent-level model at four (pr, pe) combinations and records, for each,
the hub-peripheral protection gap, the failure rate and the memorised
functional capacity, averaged over agents and over the stationary second half
of the run. Figure 6 uses the two conditions at pr = 0.1; the reduced cubic
model drawn there is fitted to reproduce the transition between them, not to
the individual points. The gap is normalised for the figure, not here.

Hub agents are the top quartile of eigenvector centrality and peripheral agents
the bottom quartile.

Parameters: BA(100,10), network seed=42, s=10, T=200,000, one realisation,
            conditional regime. Stationary averages over the final 50%.

Runtime: a few minutes per condition.

Output: fig5_data.npz
"""

import numpy as np
import networkx as nx
import time

# Parameters (Supplementary Table 2)
n = 100
m = 10
T = 200000

pn = 0.1          # failure origination probability
pl = 0.3          # failure propagation probability
pmax = 1.0        # maximum protection probability
cp = 0.05         # protection half-saturation
fm = 0.1          # maintenance fraction
memory = 0.99     # memory parameter
s = 10            # selection intensity
sigma = 0.001     # exploration noise SD

SEED = 42         # dynamics seed, so the single realisation is reproducible

# Network
G = nx.barabasi_albert_graph(n, m, seed=42)
cent_dict = nx.eigenvector_centrality(G, max_iter=1000)
cent_raw = np.array([cent_dict[i] for i in range(n)])
cent = (cent_raw - cent_raw.min()) / (cent_raw.max() - cent_raw.min())
adj_list = [np.array(list(G.neighbors(i)), dtype=np.int32) for i in range(n)]
q75, q25 = np.percentile(cent, 75), np.percentile(cent, 25)
hub = cent >= q75
peri = cent <= q25


def run_agent(pe, pr, T=T, seed=SEED):
    """Run one simulation, return stationary-state agent-level averages."""
    np.random.seed(seed)

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
        fp_pot = np.random.random(n) < pn

        # Conditional propagation
        if failure.any():
            for i in np.where(failure)[0]:
                nb = adj_list[i]
                if len(nb) > 0:
                    fp_pot[nb[np.random.random(len(nb)) < pl]] = True

        # Failure realization, with immediate recovery of the previous step
        failure[:] = False
        fp_eff = np.clip(s0 + s1 * cent, 0, 1 - fm)
        pp = pmax / (1.0 + cp / np.maximum(fp_eff * capital, 1e-12))
        new_fails = fp_pot & (np.random.random(n) > pp)
        failure[new_fails] = True
        capital[new_fails] = 0.0

        # Capital update
        fp_eff = np.clip(s0 + s1 * cent, 0, 1 - fm)
        capital = 1 + (1 - fm - fp_eff) * capital
        capital_m = memory * capital + (1 - memory) * capital_m

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
            pi = 1.0 / (1.0 + np.exp(np.clip(-s * dcm, -500, 500)))
            c0 = np.random.random(len(idx)) < pi
            s0[idx[c0]] = s0[rm[c0]]
            c1 = np.random.random(len(idx)) < pi
            s1[idx[c1]] = s1[rm[c1]]

        # Exploration
        e0 = np.random.random(n) < pe
        s0[e0] += np.random.normal(0, sigma, size=e0.sum())
        e1 = np.random.random(n) < pe
        s1[e1] += np.random.normal(0, sigma, size=e1.sum())

    steps = T - T_half
    return acc_fp / steps, acc_fail / steps, acc_cap / steps


# Run four conditions
print("Figure 6: agent-level observation points for the bifurcation analysis")
print(f"BA({n}, {m}) seed 42, T={T:,}, one realisation, conditional regime")
print("=" * 70)

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
         T=T, seed=SEED, network_seed=42,
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
