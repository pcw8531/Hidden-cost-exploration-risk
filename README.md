# Exploration Under Risk in Hub-Dependent Networks

Simulation code and data for:

**Exploration under risk sustains functional capacity in hub-dependent networks when failure propagation is contained**

Chulwook Park (Seoul National University, OIST, IIASA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19467397.svg)](https://doi.org/10.5281/zenodo.19467397)

## Overview

This repository holds the core model, the figure-specific simulation scripts, and the simulation output listed below. The core model implements network-agent dynamics under two failure propagation regimes separated by a single mechanical condition.

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

Python 3.9+, NumPy, NetworkX, SciPy, Matplotlib, joblib.

```
pip install numpy networkx scipy matplotlib joblib
```

## Structure

```
Hidden-cost-exploration-risk/
├── README.md
├── LICENSE
├── requirements.txt
├── core/
│   ├── model_local.py                 # local execution, conditional propagation
│   ├── model_hpc.py                   # HPC execution, both regimes, SLURM job array
│   └── submit.sh                      # array indices 0-8 map to pr = 0.1 ... 0.9
├── simulation/
│   ├── fig3_sim.py                    # imitation sweep at two exploration levels, BA(100, 10)
│   ├── fig4_sim.py                    # four topologies, ternary panels, Supplementary Table S3
│   ├── fig4_grid_sim.py               # four topologies on the nine by five grid, resumable
│   └── fig5_sim.py                    # bifurcation observation points
└── data/                              # .npz output, one file per analysis
```

Script and data file names follow the submitted figure numbers. The revision adds a schematic as Figure 1, so submitted Figures 3, 4 and 5 are Figures 4, 5 and 6 in the published version.

## Data

Each run is T = 1,000,000 steps with R = 10 independent realisations, averaged over the stationary final half.

| File | Used in | Run |
|------|---------|-----|
| `fig2_data.npz` | Figure 4, upper row | BA(100, 10), nine imitation probabilities at two exploration levels |
| `fig3_ternary_data.npz` | Figure 4, centre and ternary row | BA(100, 10), pr = 0.1, five exploration levels, with the transient |
| `fig4_data.npz` | Figure 5 ternary panels, Supplementary Figure 4, Table S3 | four topologies, nine pr at pe = 0.1 and 0.9, plus pe = 0.3, 0.5, 0.7 at pr = 0.1 |
| `fig4_grid_*.npz` | Figure 5 centre panel | four topologies on a grid of nine pr by five pe |
| `fig1_data.npz` | Supplementary Figure 5, Note 4 | both regimes across four topologies, reduced degree grid, new network per realisation |
| `fig3_unified_scatter_targeted.npz` | Supplementary Figure 6 | degree-targeted failure origination, BA(100, 10), pr = 0.1 |
| `sfig7_data.npz` | Supplementary Figure 7 | observed positional passing network, eleven positions, 21 links |

The remaining figures are reproduced by running the scripts in `simulation/` and `core/`. Supplementary Figure 2 comes directly from `core/model_hpc.py` at BA(500, 10), T = 10,000,000, one realisation.

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
