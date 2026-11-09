"""Visualize fold/model performance from summary_without.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent
CSV = HERE / "summary_without.csv"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

OUTPUT_ORDER = ["t1", "t2", "ratio", "t1t2", "t1t2ratio"]
PAIRS = ["pair1", "pair2"]

sns.set_theme(style="whitegrid", context="talk")

df = pd.read_csv(CSV)
df["fold"] = df["fold"].astype(int)

model_order = (
    df.groupby("model")["mae"].mean().sort_values().index.tolist()
)


# ----- 1. Heatmap: model × fold per (output_type, pair) for a metric -----
def heatmap_grid(metric: str, cmap: str, fmt: str, title: str, fname: str,
                 vmin=None, vmax=None, center=None):
    fig, axes = plt.subplots(
        len(OUTPUT_ORDER), len(PAIRS),
        figsize=(2 + 4 * len(PAIRS), 2 + 2.6 * len(OUTPUT_ORDER)),
        squeeze=False,
    )
    for i, ot in enumerate(OUTPUT_ORDER):
        for j, pair in enumerate(PAIRS):
            ax = axes[i][j]
            sub = df[(df.output_type == ot) & (df.pair == pair)]
            pivot = (
                sub.pivot_table(index="model", columns="fold", values=metric)
                .reindex(model_order)
            )
            sns.heatmap(
                pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt,
                annot_kws={"size": 9}, cbar=(j == len(PAIRS) - 1),
                vmin=vmin, vmax=vmax, center=center,
                linewidths=0.4, linecolor="white",
            )
            ax.set_title(f"{ot} · {pair}", fontsize=13, pad=6)
            ax.set_xlabel("fold" if i == len(OUTPUT_ORDER) - 1 else "")
            ax.set_ylabel("model" if j == 0 else "")
            ax.tick_params(axis="y", labelsize=10, rotation=0)
            ax.tick_params(axis="x", labelsize=10)
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path = OUT / fname
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ----- 2. Bar chart: mean ± std across folds, model on x, hue=output_type ----
def bar_mean_per_model(metric: str, ylabel: str, title: str, fname: str,
                       lower_is_better: bool):
    agg = (
        df.groupby(["model", "output_type"])[metric]
        .agg(["mean", "std"]).reset_index()
    )
    agg["output_type"] = pd.Categorical(
        agg["output_type"], categories=OUTPUT_ORDER, ordered=True
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    palette = sns.color_palette("Set2", n_colors=len(OUTPUT_ORDER))
    sns.barplot(
        data=agg, x="model", y="mean", hue="output_type",
        order=model_order, hue_order=OUTPUT_ORDER,
        palette=palette, ax=ax, errorbar=None,
    )
    # add error bars manually
    n_hue = len(OUTPUT_ORDER)
    width = 0.8 / n_hue
    for i, ot in enumerate(OUTPUT_ORDER):
        offsets = np.arange(len(model_order)) - 0.4 + width / 2 + i * width
        sub = agg[agg.output_type == ot].set_index("model").reindex(model_order)
        ax.errorbar(
            offsets, sub["mean"].values, yerr=sub["std"].values,
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2, alpha=0.6,
        )
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    arrow = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.text(0.99, 0.97, arrow, ha="right", va="top", transform=ax.transAxes,
            fontsize=11, color="dimgray",
            bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3"))
    ax.legend(title="output_type", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=True)
    fig.tight_layout()
    out_path = OUT / fname
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ----- 3. Line plot: per-fold trajectory, hue=model, facet by output_type ----
def line_per_fold(metric: str, ylabel: str, title: str, fname: str):
    g = sns.relplot(
        data=df, x="fold", y=metric, hue="model", col="output_type",
        row="pair", kind="line", marker="o", height=3.4, aspect=1.4,
        facet_kws=dict(sharey=False),
        col_order=OUTPUT_ORDER, row_order=PAIRS,
        hue_order=model_order, palette="tab20",
        estimator=None, units=None,
    )
    g.set_titles("{row_name} · {col_name}")
    g.set_axis_labels("fold", ylabel)
    g.fig.suptitle(title, fontsize=16, y=1.02)
    for ax in g.axes.flat:
        ax.set_xticks(sorted(df.fold.unique()))
    g.fig.tight_layout()
    out_path = OUT / fname
    g.fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  saved -> {out_path}")


print(f"models (best→worst by mean MAE): {model_order}")
print("→ writing heatmaps")
heatmap_grid(
    "mae", "rocket_r", ".2f",
    "MAE  (lower is better)  ·  model × fold",
    "heatmap_mae.png",
    vmin=0, vmax=df.mae.quantile(0.95),
)
heatmap_grid(
    "pearsonr", "RdBu_r", ".2f",
    "Pearson r  (higher is better)  ·  model × fold",
    "heatmap_pearsonr.png",
    vmin=-0.5, vmax=0.8, center=0.0,
)

print("→ writing bar charts")
bar_mean_per_model(
    "mae", "MAE  (mean ± std across folds & pairs)",
    "Mean MAE per model · grouped by output_type",
    "bar_mae_by_model.png",
    lower_is_better=True,
)
bar_mean_per_model(
    "pearsonr", "Pearson r  (mean ± std across folds & pairs)",
    "Mean Pearson r per model · grouped by output_type",
    "bar_pearsonr_by_model.png",
    lower_is_better=False,
)

print("→ writing fold-trajectory line plots")
line_per_fold(
    "mae", "MAE",
    "Per-fold MAE trajectory · faceted by pair × output_type",
    "lines_mae_per_fold.png",
)
line_per_fold(
    "pearsonr", "Pearson r",
    "Per-fold Pearson r trajectory · faceted by pair × output_type",
    "lines_pearsonr_per_fold.png",
)
print("done.")
