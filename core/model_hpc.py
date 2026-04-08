"""
Core simulation model for HPC execution (SLURM sbatch).

This is the foundational simulation code from which all results in the
manuscript are derived. The single-condition distinction between conditional
and unrestricted failure propagation regimes is controlled by one line:

    if Failure[i] >= 0:    # UNRESTRICTED: failure potential spreads from all nodes
    if Failure[i] > 0:     # CONDITIONAL: failure potential spreads only from failed nodes

This mechanical difference produces the regime comparison reported in Figure 1
and underlies all subsequent analyses. The conditional regime (Failure[i] > 0)
is the empirically relevant case used throughout the manuscript.

Parameters: BA(500,10), T=10,000,000, SLURM job array for pr sweep.
Usage: sbatch --array=0-8 submit.sh
       Each array index maps to pr = [0.1, 0.2, ..., 0.9]
Output: result_r{index} text file with 8 x T population-averaged time series.

Based on the original model specification in:
  Park, C. Role of recovery in evolving protection against systemic risk:
  a mechanical perspective in network-agent dynamics. Complexity (2021).
"""

import networkx as nx
import numpy as np
import sys

# =============================================================================
# PARALLEL PROCESSING: pr indexed by SLURM job array
# =============================================================================
jobindex = sys.argv[1]
prarray = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pr = prarray[int(jobindex)]

# =============================================================================
# NETWORK GENERATION
# =============================================================================
n = 500
m = 10

G = nx.barabasi_albert_graph(n, m, seed=None)
M = nx.adjacency_matrix(G)

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
s = 10                      # selection intensity

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
timePeriod = 10000000

# results storage
result_r = np.zeros([8, timePeriod])

realizaiton = 1

# =============================================================================
# SIMULATION
# =============================================================================
for real in range(realizaiton):

    Capital[:] = capital
    Capital_m[:] = memory
    Strategy_0[:] = Strategy_0
    Strategy_1[:] = Strategy_1
    Failure[:] = 0

    for t in range(0, timePeriod):

        # ==================================================================
        # FAILURE DYNAMICS
        # ==================================================================
        failure_potential[np.random.random(n) <= pn] = 1

        for i in range(n):

            # ---------------------------------------------------------------
            # CRITICAL REGIME DISTINCTION
            # Toggle between the two lines below to switch propagation regime.
            # ---------------------------------------------------------------
            if Failure[i] >= 0:                    # UNRESTRICTED: spreads from all nodes
            #if Failure[i] > 0:                    # CONDITIONAL: spreads only when failed
            # ---------------------------------------------------------------

                neighbors = nx.all_neighbors(G, i)
                for j in neighbors:
                    if np.random.random() <= pl:
                        failure_potential[j] = 1

        # Failure realization
        Failure[failidx] = 0
        protection_probability = np.zeros(n)
        index = (failure_potential > 0)
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

        # Save population averages
        result_r[0, t] = np.average(Capital[:])
        result_r[1, t] = np.average(Capital_m[:])
        result_r[2, t] = np.average(Strategy_0[:])
        result_r[3, t] = np.average(Strategy_1[:])
        result_r[4, t] = np.average(Failure[:])
        result_r[5, t] = np.average(protection_probability[:])
        result_r[6, t] = np.average(fp[:])
        result_r[7, t] = np.average(Centrality[:])

result_r /= float(realizaiton)
np.savetxt('result_r{}'.format(prarray.index(pr)), result_r)
