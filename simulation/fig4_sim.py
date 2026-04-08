"""
Figure 4 simulation: Structure-dependent scaling of the hidden cost.

Runs the same pr sweep (0.1, 0.3, 0.5, 0.7, 0.9) at pe = 0.1 and pe = 0.9
across four network topologies: Regular, Erdos-Renyi, Watts-Strogatz, and
Barabasi-Albert. Computes the exploration advantage delta_cm for each topology.

Parameters: n=100, m=10 (or k=20), seed=42, s=10, T=20,000, R=2,
            conditional regime.
Estimated runtime: ~30 min on local machine (40 conditions x ~45 sec each).

Output: fig4_data.npz
"""

import networkx as nx
import numpy as np
import time

# Parameters
n = 100
T = 20000
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

pr_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
pe_values = [0.1, 0.9]
R = 2


def make_network(topo, n=100, seed=42):
    """Generate network and return graph, centrality, neighbor list."""
    if topo == 'scalefree':
        G = nx.barabasi_albert_graph(n, 10, seed=seed)
    elif topo == 'regular':
        G = nx.random_regular_graph(20, n, seed=seed)
    elif topo == 'random':
        G = nx.erdos_renyi_graph(n, 20 / (n - 1), seed=seed)
    elif topo == 'smallworld':
        G = nx.watts_strogatz_graph(n, 20, 0.3, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topo}")

    eig = nx.eigenvector_centrality_numpy(G)
    cent = np.array([eig[i] for i in range(n)])
    nbrs = {i: list(G.neighbors(i)) for i in range(n)}
    return G, cent, nbrs


def run_simulation(pe, pr, Centrality, neighbors_list):
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
        failure_potential[np.random.random(n) <= pn] = 1

        for i in range(n):
            if Failure[i] > 0:
                for j in neighbors_list[i]:
                    if np.random.random() <= pl:
                        failure_potential[j] = 1

        if isinstance(failidx, np.ndarray):
            Failure[failidx] = 0

        protection_probability = np.zeros(n)
        index = (failure_potential > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            protection_probability[index] = pmax / (1 + cp / (fp[index] * Capital[index]))
        R1 = (np.random.random(n) <= 1 - protection_probability) & index
        Failure[R1] = 1
        Capital[R1] = 0

        index = np.random.random(n) < rec1
        failure_potential[index] = 0

        failtimear[Failure == 0] += 1
        failidx = (failtimear % failtime) == 0

        fp = Strategy_0 + Strategy_1 * Centrality
        fp[fp < 0] = 0
        fp[fp > (1 - fm)] = 1 - fm

        Capital = 1 + (1 - fm - fp) * Capital
        Capital_m = memory * Capital + (1 - memory) * Capital_m

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

        for i in range(n):
            if np.random.random() <= pe:
                Strategy_0[i] += np.random.normal(0, sigma)
        for i in range(n):
            if np.random.random() <= pe:
                Strategy_1[i] += np.random.normal(0, sigma)

        result_cm[t] = np.average(Capital_m)
        result_f[t] = np.average(Failure)

    return np.mean(result_cm[T_half:]), np.mean(result_f[T_half:])


# Run all topologies
TOPOS = ['regular', 'random', 'smallworld', 'scalefree']
all_data = {}

for topo in TOPOS:
    G, Centrality, neighbors_list = make_network(topo, n=n, seed=42)

    # Degree heterogeneity
    degrees = np.array([G.degree(i) for i in range(n)])
    cv_k = degrees.std() / degrees.mean() if degrees.mean() > 0 else 0.0

    print(f"\n{topo.upper()} (CV(k) = {cv_k:.3f})")
    all_data[topo] = {'cv_k': cv_k}

    for pe in pe_values:
        cm_list, f_list = [], []
        for r in range(R):
            cm, f = run_simulation(pe, pr=0.1, Centrality=Centrality,
                                   neighbors_list=neighbors_list)
        # Full pr sweep
        for pr in pr_values:
            cm_runs, f_runs = [], []
            for r in range(R):
                cm, f = run_simulation(pe, pr, Centrality, neighbors_list)
                cm_runs.append(cm)
                f_runs.append(f)
            key = f"pe{pe}_pr{pr}"
            all_data[topo][key] = {
                'cm': np.mean(cm_runs), 'f': np.mean(f_runs)
            }
            print(f"  pe={pe}, pr={pr}: cm={np.mean(cm_runs):.3f}, f={np.mean(f_runs):.3f}")

# Save
save_dict = {}
for topo in TOPOS:
    save_dict[f'{topo}_cv_k'] = all_data[topo]['cv_k']
    for pe in pe_values:
        for pr in pr_values:
            key = f"pe{pe}_pr{pr}"
            save_dict[f'{topo}_{key}_cm'] = all_data[topo][key]['cm']
            save_dict[f'{topo}_{key}_f'] = all_data[topo][key]['f']

np.savez('fig4_data.npz', **save_dict)
print("\nSaved: fig4_data.npz")
