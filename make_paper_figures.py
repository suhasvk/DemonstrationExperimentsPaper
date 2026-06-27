"""
Generate publication-ready 3-panel power curve figures for all three studies.

Each figure has three panels (Pooled, Max linear, Max log) with a shared
cloglog y-axis and consistent styling.  Fonts are 14pt throughout.

Reads from existing CSV results; does not re-run simulations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Shared cloglog helpers
# ---------------------------------------------------------------------------
_EPS = 1e-7

def clog(p):
    return -np.log(np.clip(1.0 - np.asarray(p, dtype=float), _EPS, 1.0 - _EPS))

TICK_POWERS = np.array([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
TICK_POS    = clog(TICK_POWERS)
TICK_LABELS = [".25", ".50", ".75", ".90", ".95", ".99"]
Y_LO = clog([0.10])[0]
Y_HI = np.sqrt(clog([0.99])[0] * clog([0.999])[0])

TEST_STATS_ORDERED = ["pooled", "max-linear", "max-log"]
PANEL_TITLES = {"pooled": "Pooled", "max-linear": "Max (linear)", "max-log": "Max (log)"}


def make_power_figure(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    alg_style: dict,
    x_ticks: list,
    output_path: Path,
    use_cloglog: bool = True,
):
    """
    Create a 3-panel power figure.

    Args:
        df:         DataFrame with columns [x_col, 'algorithm', 'test_statistic', 'power', 'se']
        x_col:      column name for x-axis (e.g. 'T' or 'delta')
        x_label:    x-axis label string
        title:      figure suptitle
        alg_style:  dict mapping algorithm name -> dict(label, color, marker, lw)
        x_ticks:    list of x-axis tick values
        output_path: where to save the PNG
        use_cloglog: if True, use cloglog y-axis; if False, use linear [0,1] axis
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, ts in zip(axes, TEST_STATS_ORDERED):
        df_ts = df[df["test_statistic"] == ts]
        for alg, style in alg_style.items():
            df_alg = df_ts[df_ts["algorithm"] == alg].sort_values(x_col)
            if df_alg.empty:
                continue
            x_vals = df_alg[x_col].values
            power  = df_alg["power"].values
            se     = df_alg["se"].values
            if use_cloglog:
                y_mid     = clog(power)
                y_lo_band = clog(np.clip(power - se, _EPS, 1 - _EPS))
                y_hi_band = clog(np.clip(power + se, _EPS, 1 - _EPS))
            else:
                y_mid     = power
                y_lo_band = np.clip(power - se, 0, 1)
                y_hi_band = np.clip(power + se, 0, 1)
            ax.plot(x_vals, y_mid, **style)
            ax.fill_between(x_vals, y_lo_band, y_hi_band,
                            alpha=0.15, color=style["color"])

        ax.set_title(PANEL_TITLES[ts], fontsize=16)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")

        if use_cloglog:
            ax.axhline(clog([0.05])[0], color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_ylim(Y_LO, Y_HI)
            ax.set_yticks(TICK_POS)
            ax.set_yticklabels(TICK_LABELS, fontsize=14)
        else:
            ax.axhline(0.05, color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Power", fontsize=14)
    fig.supxlabel(x_label, fontsize=14, y=0.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(alg_style), fontsize=14, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------
STYLE_5ALG = {
    "oracle":   dict(label="Oracle",   color="black",   marker="D", lw=2.5),
    "sn_ucb":   dict(label="SN-UCB",   color="#d62728", marker="s", lw=2.5),
    "ucb":      dict(label="UCB",      color="#1f77b4", marker="^", lw=2.5),
    "ucb_v":    dict(label="UCB-V",    color="#ff7f0e", marker="P", lw=2.5),
    "thompson": dict(label="Thompson", color="#2ca02c", marker="v", lw=2.5),
    "equal":    dict(label="Uniform",  color="#7f7f7f", marker="o", lw=2.5),
}

STYLE_UCT = {
    "sn_ucb":   dict(label="SN-UCB",   color="#d62728", marker="s", lw=2.5),
    "ucb":      dict(label="UCB",      color="#1f77b4", marker="^", lw=2.5),
    "ucb_v":    dict(label="UCB-V",    color="#ff7f0e", marker="P", lw=2.5),
    "thompson": dict(label="Thompson", color="#2ca02c", marker="v", lw=2.5),
    "equal":    dict(label="Uniform",  color="#7f7f7f", marker="o", lw=2.5),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    package_root = Path(__file__).parent
    csvs_dir    = package_root / "outputs" / "csvs"
    figures_dir = package_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Spike model ---
    spike_df = pd.read_csv(csvs_dir / "spike_model_power_results.csv")
    make_power_figure(
        df=spike_df,
        x_col="delta",
        x_label=r"Effect size $\delta$",
        title="Single-Spike Power with 10 arms",
        alg_style=STYLE_5ALG,
        x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        output_path=figures_dir / "spike_power_combined.png",
        use_cloglog=False,
    )

    # --- Multi-scale ---
    ms_df = pd.read_csv(csvs_dir / "multiscale_power_results.csv")
    make_power_figure(
        df=ms_df,
        x_col="delta",
        x_label=r"Effect size $\delta$",
        title="Multi-Scale Power with 10 arms",
        alg_style=STYLE_5ALG,
        x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        output_path=figures_dir / "multiscale_power_combined.png",
        use_cloglog=False,
    )

    # --- UCT application ---
    uct_df = pd.read_csv(csvs_dir / "uct_power_results.csv")
    make_power_figure(
        df=uct_df,
        x_col="T",
        x_label=r"Horizon $T$",
        title="UCT Application: Power vs. Horizon with 14 arms",
        alg_style=STYLE_UCT,
        x_ticks=[100, 200, 300, 400, 500],
        output_path=figures_dir / "uct_power_curves.png",
        use_cloglog=False,
    )
