# Reproduction code for *Demonstration Experiments*

Minimal code to reproduce the simulation figures and table in the paper.
Each Section-5 output traces back to a single script in this folder.

## Layout

```
reproduction/
├── algorithms/
│   ├── baselines.py            # UCB, Thompson, Equal, Oracle
│   └── sn_ucb.py               # SN-UCB (Algorithm 1)
├── statistics/
│   └── test_statistics.py      # pooled, max-linear, max-log, naive, oracle tests
├── studies/
│   ├── type_i_error_heatmap.py # Type-I error grid (Table 1)
│   ├── spike_model_power.py    # Single-spike power curves (Figure 3)
│   └── multiscale_power.py     # Multi-scale power curves (Figure 2)
├── experiments/
│   └── setting_uct_application.py  # UCT application (Figure 4 + appendix)
├── make_paper_figures.py       # combines per-study CSVs into 3-panel paper figures
├── make_type1_table.py         # renders Type-I error LaTeX table from CSV
├── run_all.py                  # runs all four studies + both combiners
└── outputs/
    ├── csvs/                   # raw per-cell results
    ├── figures/                # PNGs, including the three paper figures
    └── tables/                 # auxiliary LaTeX tables
```

## Paper output → script map

| Paper output (Overleaf path) | Section | Script | CSV produced |
|---|---|---|---|
| `content/figures/type1_error_table.tex` | §5.2 | [studies/type_i_error_heatmap.py](studies/type_i_error_heatmap.py) → [make_type1_table.py](make_type1_table.py) | `type_i_error_results.csv` |
| `content/figures/multiscale_power_combined.png` | §5.3 | [studies/multiscale_power.py](studies/multiscale_power.py) → `make_paper_figures.py` | `multiscale_power_results.csv` |
| `content/figures/spike_power_combined.png` | §5.4 | [studies/spike_model_power.py](studies/spike_model_power.py) → `make_paper_figures.py` | `spike_model_power_results.csv` |
| `content/figures/uct_power_curves.png`, `uct_allocation_paths.png` | §5.5, App. C | [experiments/setting_uct_application.py](experiments/setting_uct_application.py) | `uct_power_results.csv` |

## Setup

Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `numba`, `pandas`, `matplotlib`,
`seaborn`, `tqdm`. Any virtual environment with those packages will work.

## Reproducing the paper figures

Run everything with the orchestrator:

```bash
python run_all.py            # full paper run, ~3–4 hours
python run_all.py --quick    # ~1 minute pipeline check (smaller grids, 100 reps)
```

`run_all.py` invokes each study followed by the two combiners. Each study
uses `seed=42` by default, matching what was used for the submitted PDF.

Or run individual stages:

```bash
# Type-I table (k ∈ {5,10,20,50}, T ∈ {200,500,1000,2000}, 1000 reps; 1–2 h)
python studies/type_i_error_heatmap.py

# Spike-model power curves (k=10, T=200, 11 deltas, 1000 reps each; 30–60 min)
python studies/spike_model_power.py

# Multi-scale power curves (k=10, T=200, 11 deltas, 1000 reps each; 30–60 min)
python studies/multiscale_power.py

# UCT application (10 horizons, 1000 reps; 30–60 min)
python experiments/setting_uct_application.py

# Combine per-study CSVs into the three 3-panel paper figures
python make_paper_figures.py

# Render the Type-I error LaTeX table from the CSV
python make_type1_table.py
```


