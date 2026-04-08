"""
Core simulation model for local execution.

This is the original simulation code implementing the network-agent dynamics
model with conditional failure propagation. The critical regime distinction
is the single condition in the propagation loop:

    if Failure[i] > 0:     # CONDITIONAL: propagation only from failed nodes

This code produces the local-scale results (n=100, T=100,000) used for
prototyping and figure iteration. The HPC version (model_hpc.py) scales
this to n=500, T=10,000,000 with SLURM parallel processing.

Parameters: BA(100,10), T=100,000, conditional propagation.
Output: Population-averaged time series (8 variables x T).

Based on the original model specification in:
  Park, C. Role of recovery in evolving protection against systemic risk:
  a mechanical perspective in network-agent dynamics. Complexity (2021).
"""

import networkx as nx
import numpy as np

# =============================================================================
# NETWORK GENERATION
# =============================================================================
n = 100
m = 10

G = nx.barabasi_albert_graph(n, m, seed=None)
pos = nx.spring_layout(G)
A = np.array(nx.adjacency_matrix(G).todense())
eig_centrality = nx.eigenvector_centrality_numpy(G)

# =============================================================================
# PARAMETERS
# =============================================================================
Capital = np.zeros(n)
Capital_m = np.zeros(n)
Strategy_0 = np.zeros(n)
Strategy_1 = np.zeros(n)
Failure = np.zeros(n)

failure_potential = np.zeros(n)
protection_probability = np.zeros(n)
fp = np.zeros(n)
Centrality = np.zeros(n)

capital = 1
memory = 0.99

m0, s0 = 0.0, 0.0
Strategy_0 = np.random.normal(m0, s0, size=[n])

m1, s1 = 0.0, 0.0
Strategy_1 = np.random.normal(m1, s1, size=[n])

Centrality = np.array(list(eig_centrality.values()))

fm = 0.1                   # maintenance
s = 1                       # selection intensity
pr = 0.1                    # imitation probability

# exploration dynamics
pe = 0.1                    # exploration probability
mu = 0.0                    # mean for normal increment
sigma = 0.001               # standard deviation for normal increment

# failure dynamics
pn = 0.1                    # failure origination probability
pl = 0.3                    # failure propagation probability

pmax = 1                    # max protection probability
cp = 0.05                   # half-saturation constant

# recovery
rec1 = 1.0                  # always reset failure potential
rec2 = 0.0                  # never reset failure directly

# recovery time delay
failtime = 1
failtimear = np.zeros(n)
failidx = []

# time
timePeriod = 100000

# results storage
result = np.zeros([8, timePeriod])
result_m = np.zeros([8, timePeriod])
result_sd = np.zeros([8, timePeriod])

investment = np.zeros([n, timePeriod])
centrality = np.zeros([n, timePeriod])

# =============================================================================
# INITIAL CONDITION
# =============================================================================
Capital[:] = capital
Capital_m[:] = memory
Strategy_0[:] = Strategy_0
Strategy_1[:] = Strategy_1
Failure[:] = 0

# =============================================================================
# SIMULATION
# =============================================================================
for t in range(0, timePeriod):

    # ==================================================================
    # FAILURE DYNAMICS
    # ==================================================================
    failure_potential[np.random.random(n) <= pn] = 1

    for i in range(n):

        # ---------------------------------------------------------------
        # CONDITIONAL PROPAGATION: spreads only from actually failed nodes
        # This is the single-condition mechanical distinction from the
        # unrestricted regime (Failure[i] >= 0) reported in Figure 1.
        # ---------------------------------------------------------------
        if Failure[i] > 0:

            neighbors = nx.all_neighbors(G, i)
            for j in neighbors:
                if np.random.random() <= pl:
                    failure_potential[j] = 1

    # Failure realization
    Failure[failidx] = 0
    protection_probability = np.zeros(n)
    index = (failure_potential > 0)
    with np.errstate(divide='ignore'):
        protection_probability[index] = (pmax / (1 + cp / (fp[index] * Capital[index])))
    R1 = ((np.random.random(n) <= 1 - protection_probability) & index)
    Failure[R1] = 1
    Capital[R1] = 0

    # Recovery
    index = np.random.random(n) < rec1
    failure_potential[index] = 0
    index = np.random.random(n) < rec2
    Failure[index] = 0

    failtimear[Failure[i] == 0] += 1
    failidx = ((failtimear % failtime) == 0)

    # Capital update
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
            pi = 1 / (1 + (np.exp(-s * (Capital_m[rr] - Capital_m[ff]))))
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
            pi = 1 / (1 + (np.exp(-s * (Capital_m[rr] - Capital_m[ff]))))
            if np.random.random() <= pi:
                Strategy_1[ff] = Strategy_1[rr]

    # Exploration
    for i in range(n):
        if np.random.random() <= pe:
            norInc = np.random.normal(mu, sigma, size=None)
            Strategy_0[i] = Strategy_0[i] + norInc
    for i in range(n):
        if np.random.random() <= pe:
            norInc = np.random.normal(mu, sigma, size=None)
            Strategy_1[i] = Strategy_1[i] + norInc

    # Save results
    result[0, t] = np.average(Capital[:])
    result[1, t] = np.average(Capital_m[:])
    result[2, t] = np.average(Strategy_0[:])
    result[3, t] = np.average(Strategy_1[:])
    result[4, t] = np.average(Failure[:])
    result[5, t] = np.average(protection_probability[:])
    result[6, t] = np.average(fp[:])
    result[7, t] = np.average(Centrality[:])

    result_m[0, t] = np.median(Capital[:])
    result_m[1, t] = np.median(Capital_m[:])
    result_m[2, t] = np.median(Strategy_0[:])
    result_m[3, t] = np.median(Strategy_1[:])
    result_m[4, t] = np.median(Failure[:])
    result_m[5, t] = np.median(protection_probability[:])
    result_m[6, t] = np.median(fp[:])
    result_m[7, t] = np.median(Centrality[:])

    result_sd[0, t] = np.std(Capital[:])
    result_sd[1, t] = np.std(Capital_m[:])
    result_sd[2, t] = np.std(Strategy_0[:])
    result_sd[3, t] = np.std(Strategy_1[:])
    result_sd[4, t] = np.std(Failure[:])
    result_sd[5, t] = np.std(protection_probability[:])
    result_sd[6, t] = np.std(fp[:])
    result_sd[7, t] = np.std(Centrality[:])

    investment[:, t] = fp[:]
    centrality[:, t] = Centrality[:]

print("Simulation complete.")
print(f"Memorized functional capacity (final avg): {np.mean(result[1, timePeriod//2:]):.4f}")
print(f"Failure rate (final avg): {np.mean(result[4, timePeriod//2:]):.4f}")
