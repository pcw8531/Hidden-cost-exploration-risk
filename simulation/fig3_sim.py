"""
Figure 3 simulation: Exploration under risk in scale-free networks.

Sweeps imitation probability pr = 0.1 to 0.9 at two exploration levels
(pe = 0.1 and pe = 0.9) under conditional propagation on BA scale-free
networks. Outputs stationary-state averages of memorized functional
capacity and failure rate for each (pe, pr) combination.

Parameters: BA(100,10), seed=42, s=10, T=100,000, R=3, conditional regime.
Estimated runtime: ~1 hour on local machine (18 conditions x ~3 min each).

Output: fig3_data.npz
"""

import networkx as nx
import numpy as np
import time

# Parameters
n = 100
m = 10
T = 100000
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

pr_values = np.arange(0.1, 1.0, 0.1)
pe_values = [0.1, 0.9]
R = 3

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


def run_simulation(pe, pr):
    """Run one realization under conditional propagation."""
    Capital = np.ones(n) * cin
    Capital_m = np.ones(n) * memory
    Strategy_0 = np.zeros(n)
    Strategy_1 = np.zeros(n)
    Failure = np.zeros(n)
    failure_potential = np.zeros(n)
    fp = np.zeros(n)
    failtimear = np.zeros(n)
    failidx = []

    result_cm = np.zeros(T)
    result_f = np.zeros(T)

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

        result_cm[t] = np.average(Capital_m)
        result_f[t] = np.average(Failure)

    return {
        'cm_mean': np.mean(result_cm[T_half:]),
        'cm_std': np.std(result_cm[T_half:]),
        'f_mean': np.mean(result_f[T_half:]),
        'f_std': np.std(result_f[T_half:]),
    }


# Run all conditions
all_results = {}

for pe in pe_values:
    print(f"pe = {pe}")
    for pr in pr_values:
        cm_runs, f_runs = [], []
        for r in range(R):
            res = run_simulation(pe, pr)
            cm_runs.append(res['cm_mean'])
            f_runs.append(res['f_mean'])

        all_results[(pe, round(pr, 1))] = {
            'cm_mean': np.mean(cm_runs), 'cm_std': np.std(cm_runs),
            'f_mean': np.mean(f_runs), 'f_std': np.std(f_runs),
        }
        print(f"  pr={pr:.1f}: cm={np.mean(cm_runs):.4f} (+/-{np.std(cm_runs):.4f}), "
              f"f={np.mean(f_runs):.4f} (+/-{np.std(f_runs):.4f})")

# Save
cm_pe01 = np.array([all_results[(0.1, round(pr, 1))]['cm_mean'] for pr in pr_values])
cm_pe09 = np.array([all_results[(0.9, round(pr, 1))]['cm_mean'] for pr in pr_values])
cm_pe01_std = np.array([all_results[(0.1, round(pr, 1))]['cm_std'] for pr in pr_values])
cm_pe09_std = np.array([all_results[(0.9, round(pr, 1))]['cm_std'] for pr in pr_values])
f_pe01 = np.array([all_results[(0.1, round(pr, 1))]['f_mean'] for pr in pr_values])
f_pe09 = np.array([all_results[(0.9, round(pr, 1))]['f_mean'] for pr in pr_values])
f_pe01_std = np.array([all_results[(0.1, round(pr, 1))]['f_std'] for pr in pr_values])
f_pe09_std = np.array([all_results[(0.9, round(pr, 1))]['f_std'] for pr in pr_values])

np.savez('fig3_data.npz',
         pr_values=pr_values,
         cm_pe01=cm_pe01, cm_pe09=cm_pe09,
         cm_pe01_std=cm_pe01_std, cm_pe09_std=cm_pe09_std,
         f_pe01=f_pe01, f_pe09=f_pe09,
         f_pe01_std=f_pe01_std, f_pe09_std=f_pe09_std)

print("\nSaved: fig3_data.npz")
