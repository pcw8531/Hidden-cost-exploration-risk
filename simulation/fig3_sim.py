"""
Exploration under risk in scale-free networks.

Produces the data behind Figure 4 of the manuscript, which was Figure 3 of the
submitted version. The revision adds a schematic as Figure 1, so the submitted
Figures 1 to 5 appear as Figures 2 to 6.

Sweeps imitation probability pr = 0.1 to 0.9 at two exploration levels
(pe = 0.1 and pe = 0.9) under conditional propagation on a BA scale-free
network. Every realisation is written out, not only its mean, so that the 95%
intervals and the bootstrap interval on the exploration advantage reported in
the manuscript can be recomputed from the file.

Parameters: BA(100,10), network seed=42, s=10, T=1,000,000, R=10,
            conditional regime.
The ten realisations differ in the dynamics only; the network is fixed.
Stationary values are averages over the final 50% of each run.

Runtime: this is the full-length protocol of the revision, so a single core
takes days. Run the conditions in parallel, or use core/model_hpc.py with the
SLURM array.

Output: fig3_data.npz
"""

import networkx as nx
import numpy as np
import time

# Parameters
n = 100
m = 10
T = 1000000
T_half = T // 2

pmax = 1.0
cp = 0.05
pn = 0.1
pl = 0.3
cin = 1.0
fm = 0.1
memory = 0.99
s = 10
sigma = 0.001
rec1 = 1.0
failtime = 1

pr_values = np.round(np.arange(0.1, 1.0, 0.1), 1)
pe_values = [0.1, 0.9]
R = 10
SEED0 = 40000

# Network
G = nx.barabasi_albert_graph(n, m, seed=42)
eig_centrality = nx.eigenvector_centrality_numpy(G)
Centrality = np.array(list(eig_centrality.values()))
neighbors_list = {i: list(G.neighbors(i)) for i in range(n)}

print(f"Network: BA({n}, {m}), seed=42")
print(f"Centrality range: {Centrality.min():.4f} to {Centrality.max():.4f}")
print(f"Parameters: s={s}, T={T}, pl={pl}, pn={pn}, sigma={sigma}")
print(f"Regime: CONDITIONAL (Failure[i] > 0)")
print(f"Sweep: pe = {pe_values}, pr = 0.1 to 0.9, R = {R}")
print(f"Stationary averaging: final 50% (t = {T_half} to {T})")
print()


def run_simulation(pe, pr, seed):
    """Run one realization under conditional propagation."""
    np.random.seed(seed)

    Capital = np.ones(n) * cin
    Capital_m = np.ones(n) * memory
    Strategy_0 = np.zeros(n)
    Strategy_1 = np.zeros(n)
    Failure = np.zeros(n)
    failure_potential = np.zeros(n)
    fp = np.zeros(n)
    failtimear = np.zeros(n)
    failidx = []

    # Stationary accumulators, so memory does not scale with T
    acc_cm = acc_f = 0.0
    nacc = 0

    for t in range(T):
        # Failure potential origination
        failure_potential[np.random.random(n) <= pn] = 1

        # Failure potential propagation (CONDITIONAL: Failure[i] > 0)
        for i in range(n):
            if Failure[i] > 0:
                for j in neighbors_list[i]:
                    if np.random.random() <= pl:
                        failure_potential[j] = 1

        # Failure realization
        if isinstance(failidx, np.ndarray):
            Failure[failidx] = 0

        protection_probability = np.zeros(n)
        index = (failure_potential > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            protection_probability[index] = pmax / (1 + cp / (fp[index] * Capital[index]))
        R1 = (np.random.random(n) <= 1 - protection_probability) & index
        Failure[R1] = 1
        Capital[R1] = 0

        # Recovery
        index = np.random.random(n) < rec1
        failure_potential[index] = 0

        failtimear[Failure == 0] += 1
        failidx = (failtimear % failtime) == 0

        # Functional capacity update
        fp = Strategy_0 + Strategy_1 * Centrality
        fp[fp < 0] = 0
        fp[fp > (1 - fm)] = 1 - fm

        Capital = 1 + (1 - fm - fp) * Capital
        Capital_m = memory * Capital + (1 - memory) * Capital_m

        # Imitation of Strategy_0
        for i in range(n):
            if np.random.random() <= pr:
                ff = i
                while True:
                    rr = np.random.choice(n)
                    if ff != rr:
                        break
                pi = 1 / (1 + np.exp(-s * (Capital_m[rr] - Capital_m[ff])))
                if np.random.random() <= pi:
                    Strategy_0[ff] = Strategy_0[rr]

        # Imitation of Strategy_1
        for i in range(n):
            if np.random.random() <= pr:
                ff = i
                while True:
                    rr = np.random.choice(n)
                    if ff != rr:
                        break
                pi = 1 / (1 + np.exp(-s * (Capital_m[rr] - Capital_m[ff])))
                if np.random.random() <= pi:
                    Strategy_1[ff] = Strategy_1[rr]

        # Exploration
        for i in range(n):
            if np.random.random() <= pe:
                Strategy_0[i] += np.random.normal(0, sigma)
        for i in range(n):
            if np.random.random() <= pe:
                Strategy_1[i] += np.random.normal(0, sigma)

        if t >= T_half:
            acc_cm += np.average(Capital_m)
            acc_f += np.average(Failure)
            nacc += 1

    return acc_cm / nacc, acc_f / nacc


# Run all conditions
conditions = [(pe, round(float(pr), 1)) for pe in pe_values for pr in pr_values]
out = {}

for ci, (pe, pr) in enumerate(conditions):
    cm_runs, f_runs = [], []
    t0 = time.time()
    for r in range(R):
        cm, f = run_simulation(pe, pr, SEED0 + 1000 * ci + r)
        cm_runs.append(cm)
        f_runs.append(f)
    out[f'cm_pe{pe}_pr{pr}'] = np.array(cm_runs)
    out[f'f_pe{pe}_pr{pr}'] = np.array(f_runs)
    print(f"pe={pe}, pr={pr:.1f}: cm={np.mean(cm_runs):.4f} (+/-{np.std(cm_runs, ddof=1):.4f}), "
          f"f={np.mean(f_runs):.4f} (+/-{np.std(f_runs, ddof=1):.4f})  "
          f"[{time.time() - t0:.0f} s]")

# Save. Per-realisation arrays of length R, one pair per condition.
np.savez('fig3_data.npz',
         T=T, R=R, seed0=SEED0, network_seed=42,
         pr_values=pr_values, pe_values=np.array(pe_values),
         centrality=Centrality,
         **out)

print("\nSaved: fig3_data.npz")
