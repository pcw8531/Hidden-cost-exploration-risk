# Exploration Under Risk in Hub-Dependent Networks

Simulation code and data for:

**Exploration under risk sustains functional capacity in hub-dependent networks when failure propagation is contained**

Chulwook Park (Seoul National University, OIST, IIASA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

Every measured value in the manuscript and the electronic supplementary material is reproducible from this repository. The core model implements network-agent dynamics under two failure propagation regimes separated by a single mechanical condition.

The model shared with the companion papers, refs [1] and [5] in the manuscript, is at https://github.com/pcw8531/sports-network-risk-propagation

## The regime distinction

The analysis rests on one line in the propagation loop.

```python
# CONDITIONAL: propagation only from nodes that have actually failed
if Failure[i] > 0:

# UNRESTRICTED: propagation from all nodes regardless of state
if Failure[i] >= 0:
```

Under the conditional rule, propagation pressure scales with the local failure environment, so the strategy composition of the population can influence the outcome. Under the unrestricted rule the propagation term saturates and differences in protection between agents produce only small differences in failure. In the scale-free case 89 per cent of agents stay active under the conditional rule against 80 per cent failing under the unrestricted rule, a 4.3-fold difference in mean functional capacity.

Both are implemented in `core/model_hpc.py`, with the two lines marked, and in `core/model_local.py`.

## Requirements

Python 3.9+, NumPy, NetworkX, SciPy, Matplotlib.

```
pip install numpy networkx scipy matplotlib
```

## Structure

```
Hidden-cost-exploration-risk/
├── core/
│   ├── model_local.py                 # local execution, conditional propagation
│   ├── model_hpc.py                   # HPC execution, both regimes, SLURM job array
│   └── submit.sh                      # array indices 0-8 map to pr = 0.1 ... 0.9
├── simulation/
│   ├── fig2_regime_topology.py
│   ├── fig3_penalty.py
│   ├── fig4_exploration.py
│   ├── fig5_topology_grid.py
│   ├── fig6_bifurcation.py
│   └── si_figures.py                  # Supplementary Figures 5, 6, 7
└── data/                              # .npz output, one file per analysis
```

## Data

Unless a row says otherwise, each run is T = 1,000,000 steps with R = 10 independent realisations on one fixed network per topology generated with seed 42, averaged over the stationary final half.

| File | Used in | Run |
|------|---------|-----|
| `fig2_regime_comparison.npz` | Figure 2 | connectance sweep, four topologies, both regimes, T = 100, new network per realisation |
| `fig3_scatter.npz` | Figure 3, top | BA(100, 10), pr = 0.1, pe = 0.1 and 0.9, agent-level stationary values |
| `fig3_traces.npz` | Figure 3, bottom | BA(100, 10), T = 100,000, one realisation, five agents across the centrality range |
| `fig4_exploration.npz` | Figure 4 | BA(100, 10), nine imitation probabilities at two exploration levels |
| `fig5_topology.npz` | Figure 5 ternary panels, Supplementary Figure 4, Table S3 | four topologies, nine pr at pe = 0.1 and 0.9, plus pe = 0.3, 0.5, 0.7 at pr = 0.1 |
| `fig5_grid_*.npz` | Figure 5 centre panel | four topologies on a grid of nine pr by five pe, 1800 runs |
| `fig6_bifurcation.npz` | Figure 6 | BA(100, 10), T = 200,000, one realisation |
| `si_fig1_meanfield.npz` | Supplementary Figure 1 | regular network, fixed protection, T = 10,000 |
| `si_fig3_agent_level.npz` | Supplementary Figure 3 | BA(100, 10), pr = 0.1, both exploration levels |
| `si_fig5_regime_topology.npz` | Supplementary Figure 5, Note 4 | both regimes across four topologies at the stationary state, reduced connectance grid |
| `si_fig6_targeted.npz` | Supplementary Figure 6 | degree-targeted failure origination, BA(100, 10), pr = 0.1 |
| `si_fig7_observed_network.npz` | Supplementary Figure 7 | observed positional passing network, eleven positions, 21 links |

Supplementary Figure 2 is produced directly by `core/model_hpc.py` at BA(500, 10), T = 10,000,000, one realisation.

## Parameters

Supplementary Table 2 of the manuscript gives the full cross-study comparison.

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Primary network | BA scale-free | n = 100, m = 10 |
| Comparison topologies | regular, ER, WS, BA | mean degree 20, WS rewiring 0.3 |
| Failure origination | pn | 0.1 |
| Failure propagation | pl | 0.3 |
| Max protection probability | pp,max | 1.0 |
| Protection half-saturation | cp,1/2 | 0.05 |
| Initial functional capacity | c_in | 1.0 |
| Maintenance fraction | fm | 0.1 |
| Selection intensity | s | 10 |
| Exploration noise SD | sigma_e | 0.001 |
| Memory parameter | alpha | 0.99 |
| Imitation probability | pr | 0.1 to 0.9 |
| Exploration probability | pe | 0.1 and 0.9, with 0.3, 0.5, 0.7 in the phase space |
| Recovery delay | rt | 1, immediate |
| Time steps, realisations | T, R | 1,000,000 and 10 for the main results |
| Propagation regime | conditional | `Failure[i] > 0` |

## Citation

```
Park, C. Exploration under risk sustains functional capacity in hub-dependent
networks when failure propagation is contained. J. R. Soc. Interface (2026).
```

## Related work

Third in a series sharing the same core model specification.

1. Park, C. Network topology and recovery delay thresholds determine cascading failure vulnerability in sports systems. *Sci. Rep.* **16**, 10852 (2026).
2. Park, C. Network centrality drives optimal protection investment against systemic risk propagation in complex systems. *Sci. Rep.* **16**, 4595 (2026).

Code: https://github.com/pcw8531/sports-network-risk-propagation

## License

MIT License. See [LICENSE](LICENSE) for details.
