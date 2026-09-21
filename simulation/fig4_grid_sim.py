"""
Exploration advantage on the imitation-exploration plane, per topology.

Produces the data behind the centre panel of Figure 5 of the manuscript, which
was Figure 4 of the submitted version. The revision adds a schematic as
Figure 1, so the submitted Figures 1 to 5 appear as Figures 2 to 6.

Each topology is measured on its own grid of nine imitation probabilities by
five exploration probabilities, against its own baseline at the lowest
exploration level, so the whole of each panel is measured and nothing is
interpolated between topologies. The companion script fig4_sim.py produces the
ternary panels of the same figure.

Parameters: n=100, mean degree 20, network seed=42, s=10, T=1,000,000, R=10,
            conditional regime.
One fixed network per topology; the ten realisations differ in the dynamics
only. Stationary values are averages over the final 50% of each run.

Runtime: 9 x 5 x 10 = 450 runs per topology, 1800 in all. This is far beyond a
single uninterrupted session in pure Python, so the sweep checkpoints after
every run and resumes where it stopped. Give it a time budget in seconds and
call it repeatedly, or run the four topologies in parallel:

    python3 fig4_grid_sim.py regular 3600
    python3 fig4_grid_sim.py smallworld 3600
    python3 fig4_grid_sim.py random 3600
    python3 fig4_grid_sim.py scalefree 3600

State is held in fig4_grid_state_<topology>.npz and removed when the topology
completes.

Output: fig4_grid_<topology>.npz, one per topology
"""

import os
import sys
import time

import networkx as nx
import numpy as np

# Parameters (Supplementary Table 2)
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

PR = np.round(np.arange(0.1, 0.91, 0.1), 2)     # nine imitation probabilities
PE = np.array([0.1, 0.3, 0.5, 0.7, 0.9])        # five exploration probabilities
R = 10
SEED0 = 70000

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

    Returns the stationary memorized functional capacity and failure rate.
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


def sweep(topo, budget_s):
    """Fill the grid for one topology, resuming from any saved state."""
    G, Centrality, neighbors_list = make_network(topo, n=n, seed=42)
    k = np.array([G.degree(i) for i in range(n)], float)
    cv = k.std() / k.mean() if k.mean() > 0 else 0.0

    state = f'fig4_grid_state_{topo}.npz'
    if os.path.exists(state):
        z = np.load(state)
        cm, f, done = z['cm'], z['f'], z['done']
        print(f"resuming {topo}: {int(done.sum())} of {done.size} runs already finished")
    else:
        cm = np.zeros((len(PE), len(PR), R))
        f = np.zeros((len(PE), len(PR), R))
        done = np.zeros((len(PE), len(PR), R), dtype=bool)
        print(f"starting {topo}: CV(k) = {cv:.4f}, {done.size} runs")

    t_start = time.time()
    for i, pe in enumerate(PE):
        for j, pr in enumerate(PR):
            for r in range(R):
                if done[i, j, r]:
                    continue
                if time.time() - t_start > budget_s:
                    np.savez(state, cm=cm, f=f, done=done)
                    print(f"budget reached, saved {state} "
                          f"({int(done.sum())} of {done.size} done)")
                    return False
                seed = SEED0 + 100000 * i + 1000 * j + r
                cm[i, j, r], f[i, j, r] = run_simulation(
                    pe, pr, Centrality, neighbors_list, seed)
                done[i, j, r] = True
                if done.sum() % 10 == 0:
                    np.savez(state, cm=cm, f=f, done=done)
                    print(f"  {int(done.sum())}/{done.size}  "
                          f"pe={pe} pr={pr} r={r}  cm={cm[i, j, r]:.4f}")

    np.savez(f'fig4_grid_{topo}.npz',
             pr=PR, pe=PE, cm=cm, f=f, cv=cv, T=T, R=R,
             seed0=SEED0, network_seed=42)
    if os.path.exists(state):
        os.remove(state)
    print(f"\nSaved: fig4_grid_{topo}.npz")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in TOPOS:
        print(f"usage: python3 {os.path.basename(__file__)} "
              f"<{'|'.join(TOPOS)}> [budget_seconds]")
        sys.exit(1)
    topo = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    finished = sweep(topo, budget)
    sys.exit(0 if finished else 2)
