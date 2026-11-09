"""Visualize CV robustness across 5 folds × 2 seeds (10 runs per config)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

DIR = Path("$PATH/RI_anatomical_prior/cv_robustness")
FIG = DIR / "figures"
FIG.mkdir(exist_ok=True)

# Internal data keys, ordered as we want them to appear in figures
OUTS = ["t1", "t2", "t1t2", "ratio", "t1t2ratio"]
DISPLAY = {
    "t1":         "T1w",
    "t2":         "T2w",
    "t1t2":       "T1w+T2w",
    "ratio":      "RI",
    "t1t2ratio":  "T1w+T2w+RI",
}
OUTS_LABEL = [DISPLAY[o] for o in OUTS]
OT_PALETTE = dict(zip(OUTS, sns.color_palette("Set2", n_colors=5)))
sns.set_theme(style="whitegrid", context="talk")

# Read while preserving "None" string (don't auto-convert to NaN)
runs = pd.read_csv(DIR / "all_runs.csv", keep_default_na=False, na_values=[""])
runs = runs[runs.add_option == "None"].copy()
runs["output_type"] = pd.Categorical(runs["output_type"], categories=OUTS, ordered=True)
summary = pd.read_csv(DIR / "summary_per_model_outputtype.csv")
summary["output_type"] = pd.Categorical(summary["output_type"], categories=OUTS, ordered=True)
decomp = pd.read_csv(DIR / "variance_decomposition.csv")
decomp["output_type"] = pd.Categorical(decomp["output_type"], categories=OUTS, ordered=True)

# Sort models by mean MAE for consistent ordering
model_order = runs.groupby("model")["mae"].mean().sort_values().index.tolist()


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def fig1_overview():
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.16
    x = np.arange(len(model_order))
    for i, ot in enumerate(OUTS):
        sub = summary[summary.output_type == ot].set_index("model").reindex(model_order)
        offsets = x + (i - 2) * width
        ax.bar(offsets, sub["mae_mean"], width=width, yerr=sub["mae_std"],
               label=DISPLAY[ot], color=OT_PALETTE[ot], edgecolor="black", linewidth=0.6,
               capsize=2.5, error_kw=dict(lw=0.8, alpha=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=35, ha="right")
    ax.set_ylabel("MAE (months)  ·  mean ± std across 10 runs")
    ax.set_title("Per-model held-out MAE  ·  5 fold × 2 seed (10 evaluations each)",
                 fontsize=15, pad=10)
    ax.legend(title="modality", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "01_per_model_mae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2 — Coefficient-of-variation heatmap (the headline robustness chart)
# ----------------------------------------------------------------------
def fig2_cv_heatmap():
    pivot = (summary.pivot_table(index="model", columns="output_type",
                                 values="mae_cv_pct", observed=True)
             .reindex(model_order)[OUTS])
    pivot.columns = OUTS_LABEL
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".1f",
                vmin=5, vmax=40, center=20,
                annot_kws={"size": 11}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "CV(MAE) %  ·  lower = more stable"})
    ax.set_title("Coefficient of variation of MAE across 10 runs\n"
                 "(green = stable across folds & seeds)",
                 fontsize=13, pad=12)
    ax.set_xlabel("modality"); ax.set_ylabel("model  (best → worst by mean MAE)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "02_cv_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 — Variance decomposition: fold std vs seed std scatter
# ----------------------------------------------------------------------
def fig3_variance_decomp():
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for ot in OUTS:
        sub = decomp[decomp.output_type == ot]
        ax.scatter(sub["fold_std_within_seed"], sub["seed_std_within_fold"],
                   s=110, color=OT_PALETTE[ot], edgecolor="black",
                   alpha=0.85, label=DISPLAY[ot])
    lim = max(decomp["fold_std_within_seed"].max(), decomp["seed_std_within_fold"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("std of MAE across folds  (within a seed)")
    ax.set_ylabel("std of MAE across seeds  (within a fold)")
    leg = ax.legend(title="modality", loc="upper left", fontsize=15, markerscale=0.8)
    leg.get_title().set_fontsize(17)
    fig.tight_layout()
    fig.savefig(FIG / "03_variance_decomposition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4 — Per-fold MAE for the BEST configs (both seeds overlaid)
# ----------------------------------------------------------------------
def fig4_per_fold_trajectory():
    best_per_ot = (summary.sort_values("mae_mean")
                   .groupby("output_type", observed=True).head(1))
    fig, ax = plt.subplots(figsize=(11, 6))
    for ot in OUTS:
        brow = best_per_ot[best_per_ot.output_type == ot]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        m = brow["model"]
        sub = runs[(runs.model == m) & (runs.output_type == ot)]
        for seed, marker in [("seed_default", "o"), ("seed_1234", "s")]:
            ss = sub[sub.seed == seed].sort_values("fold")
            ax.plot(ss.fold, ss.mae, marker=marker, color=OT_PALETTE[ot],
                    lw=2, markersize=10, alpha=0.85, mec="black",
                    label=f"{DISPLAY[ot]} ({m}) · {seed}")
    ax.set_xticks(range(5))
    ax.set_xlabel("fold"); ax.set_ylabel("MAE (months)")
    ax.set_title("Per-fold MAE for best (model, output_type) — both seeds overlaid\n"
                 "(parallel lines & small spread = stable across folds)",
                 fontsize=13, pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "04_per_fold_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def fig5_complexity_vs_variance():
    params_M = {
        "basiccnn": 1.2, "cbr_tiny": 0.7, "agenet": 0.8, "chen_net": 0.7,
        "sfcn": 3.0, "efficientnet": 4.6,
        "resnet18": 33.2, "resnet34": 63.5, "resnet50": 46.0,
        "densenet": 11.2, "unet": 19.0,
    }
    summary["params_M"] = summary["model"].map(params_M)
    sub = summary.dropna(subset=["params_M"])
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(9, 6.5))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#b0b0b0')
    for ot in OUTS:
        s = sub[sub.output_type == ot]
        ax.scatter(s["params_M"], s["mae_cv_pct"], s=140, color=OT_PALETTE[ot],
                   edgecolor="black", alpha=0.85, label=DISPLAY[ot])
    ax.set_xscale("log")
    ax.set_xlabel("backbone parameters (M)  ·  log scale")
    ax.set_ylabel("CV(MAE) %")
    ymax = max(110, sub["mae_cv_pct"].max() * 1.1)
    ax.set_ylim(0, ymax)
    # Major ticks at 20 intervals (with labels): solid grid
    ax.set_yticks(np.arange(0, ymax + 1, 20))
    ax.grid(True, which='major', axis='both', linestyle='-', color='#b0b0b0', linewidth=0.8)
    # Minor ticks at 10 intervals (no labels): light dashed grid
    ax.set_yticks(np.arange(10, ymax + 1, 20), minor=True)
    ax.grid(True, which='minor', axis='y', linestyle='--', color='#cccccc', alpha=0.5, linewidth=0.5)
    pr, pp = pearsonr(np.log(sub["params_M"]), sub["mae_cv_pct"])
    sr, sp = spearmanr(sub["params_M"], sub["mae_cv_pct"])
    ax.text(0.98, 0.98,
            f"Pearson r (log params, CV) = {pr:+.3f}  (p={pp:.3g})\n"
            f"Spearman ρ                 = {sr:+.3f}  (p={sp:.3g})",
            transform=ax.transAxes, va="top", ha="right", fontsize=15,
            bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.4"))
    leg = ax.legend(title="modality", loc="upper right", fontsize=13, markerscale=0.8,
                    bbox_to_anchor=(0.99, 0.84))
    leg.get_title().set_fontsize(15)
    fig.tight_layout()
    fig.savefig(FIG / "05_complexity_vs_cv.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 6 — Distribution of MAE across all runs per output_type
# ----------------------------------------------------------------------
def fig6_overall_mae_dist():
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=runs, x="output_type", y="mae", order=OUTS,
                palette=[OT_PALETTE[o] for o in OUTS],
                hue="output_type", legend=False, ax=ax,
                width=0.55, linewidth=1.2, fliersize=0)
    sns.stripplot(data=runs, x="output_type", y="mae", order=OUTS,
                  ax=ax, color="black", size=3, alpha=0.45, jitter=0.2)
    medians = runs.groupby("output_type", observed=True)["mae"].median().reindex(OUTS)
    for i, m in enumerate(medians):
        ax.text(i, m, f" {m:.2f}", color="white", fontweight="bold",
                ha="left", va="center", fontsize=11,
                bbox=dict(facecolor="black", edgecolor="none", pad=2))
    ax.set_xticks(range(len(OUTS)))
    ax.set_xticklabels(OUTS_LABEL)
    ax.set_title(f"MAE distribution across all {len(runs)} held-out runs", fontsize=14, pad=10)
    ax.set_ylabel("MAE (months)"); ax.set_xlabel("modality")
    fig.tight_layout()
    fig.savefig(FIG / "06_mae_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


fig1_overview()
fig2_cv_heatmap()
fig3_variance_decomp()
fig4_per_fold_trajectory()
fig5_complexity_vs_variance()
fig6_overall_mae_dist()
print(f"saved 6 figures -> {FIG}")
