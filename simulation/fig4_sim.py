"""
Structure-dependent scaling of the exploration advantage.

Produces the data behind Figure 5 of the manuscript, which was Figure 4 of the
submitted version, and behind Supplementary Figure 4 and Supplementary Table 3.
The revision adds a schematic as Figure 1, so the submitted Figures 1 to 5
appear as Figures 2 to 6.

Runs the imitation sweep pr = 0.1 to 0.9 at pe = 0.1 and pe = 0.9 across four
topologies (regular, Erdos-Renyi, Watts-Strogatz, Barabasi-Albert), and adds
the intermediate exploration levels pe = 0.3, 0.5 and 0.7 at pr = 0.1 for the
strategy phase space. Six stationary quantities are written out per
realisation, so the 95% intervals, the bootstrap interval on the exploration
advantage, and the Supplementary Table 3 rows can all be recomputed from the
file.

Parameters: n=100, mean degree 20, network seed=42, s=10, T=1,000,000, R=10,
            conditional regime.
One fixed network per topology; the ten realisations differ in the dynamics
only. Stationary values are averages over the final 50% of each run.

Runtime: this is the full-length protocol of the revision, so a single core
takes days. Run the conditions in parallel, or use core/model_hpc.py with the
SLURM array.

Output: fig4_data.npz
"""

import networkx as nx
import numpy as np
import time

# Parameters
n = 100
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
pe_extra = [0.3, 0.5, 0.7]          # phase space, at pr = 0.1 only
R = 10
SEED0 = 40000

TOPOS = ['regular', 'random', 'smallworld', 'scalefree']


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


def run_simulation(pe, pr, Centrality, neighbors_list, seed):
    """Run one realization under conditional propagation.

    Returns the six stationary quantities, in order:
      cm   memorized functional capacity, population mean
      f    failure rate, population mean
      fp   effective protection, population mean
      pc   protection cost, population mean of fp * capital
      div  strategy diversity, standard deviation of fp across agents
      pp   protection probability, population mean
    """
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

    acc = np.zeros(6)
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

        cost = fp * Capital
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
            acc += (Capital_m.mean(), Failure.mean(), fp.mean(),
                    cost.mean(), fp.std(), protection_probability.mean())
            nacc += 1

    return acc / nacc


# Build the condition list once, so the seed of every run is reproducible
conditions = ([(t, pe, round(float(pr), 1)) for t in TOPOS
               for pe in pe_values for pr in pr_values]
              + [(t, pe, 0.1) for t in TOPOS for pe in pe_extra])

networks, cv = {}, {}
for topo in TOPOS:
    G, Centrality, neighbors_list = make_network(topo, n=n, seed=42)
    networks[topo] = (Centrality, neighbors_list)
    k = np.array([G.degree(i) for i in range(n)], float)
    cv[topo] = k.std() / k.mean() if k.mean() > 0 else 0.0
    print(f"{topo:11s} mean degree {k.mean():.1f}, CV(k) = {cv[topo]:.4f}")
print()

KEYS = ('cm', 'f', 'fp', 'pc', 'div', 'pp')
out = {}

for ci, (topo, pe, pr) in enumerate(conditions):
    Centrality, neighbors_list = networks[topo]
    runs = np.zeros((R, 6))
    t0 = time.time()
    for r in range(R):
        runs[r] = run_simulation(pe, pr, Centrality, neighbors_list,
                                 SEED0 + 1000 * ci + r)
    for j, key in enumerate(KEYS):
        out[f'{topo}_pe{pe}_pr{pr}_{key}'] = runs[:, j]
    print(f"[{ci + 1}/{len(conditions)}] {topo:11s} pe={pe} pr={pr}: "
          f"cm={runs[:, 0].mean():.4f}+/-{runs[:, 0].std(ddof=1):.4f}  "
          f"f={runs[:, 1].mean():.4f}+/-{runs[:, 1].std(ddof=1):.4f}  "
          f"[{time.time() - t0:.0f} s]")

np.savez('fig4_data.npz',
         T=T, R=R, seed0=SEED0, network_seed=42,
         pr_values=pr_values, topos=np.array(TOPOS),
         cv=np.array([cv[t] for t in TOPOS]),
         **out)

print("\nSaved: fig4_data.npz")
