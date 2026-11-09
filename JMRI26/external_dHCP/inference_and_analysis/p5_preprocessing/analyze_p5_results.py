"""
Comprehensive analysis of p5-preprocessed dHCP inference results.
5 modalities: T1w, T2w, T1w+T2w, RI(p5), T1w+T2w+RI(p5)
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path("$PATH/DHCP/JMRI_inference")
OUT = BASE / "p5_preprocessing"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# ── Load all results ──
configs = {
    "T1w":              ("t1",        "pair{p}"),
    "T2w":              ("t2",        "pair{p}"),
    "T1w+T2w":          ("t1t2",      "pair{p}"),
    "RI (p5)":          ("ratio",     "pair{p}_improved"),
    "T1w+T2w+RI (p5)":  ("t1t2ratio", "pair{p}_improved"),
}

MODALITY_ORDER = ["T1w", "T2w", "T1w+T2w", "RI (p5)", "T1w+T2w+RI (p5)"]

rows = []
for label, (ot, pair_tmpl) in configs.items():
    for p in ["1", "2"]:
        pair_dir = pair_tmpl.format(p=p)
        pattern = f"{ot}/{pair_dir}/*_metrics.json"
        for f in sorted(glob.glob(str(BASE / pattern))):
            if "_p10" in f or "_nonzero" in f:
                continue
            fp = Path(f)
            name = fp.stem.replace("_metrics", "")
            parts = name.rsplit("_fold", 1)
            model, fold = parts[0], int(parts[1]) if len(parts) > 1 else 0
            with open(f) as fh:
                m = json.load(fh)
            rows.append({
                "modality": label, "output_type": ot,
                "pair": f"pair{p}", "model": model, "fold": fold,
                "mae": m["mae"], "rmse": m["rmse"],
                "pearsonr": m["pearsonr"], "spearmanr": m["spearmanr"],
                "r2": m["r2"], "n_subjects": m["n_subjects"],
            })

df = pd.DataFrame(rows)
df = df[~df["model"].isin(["resnet101", "resnet152"])]
df["modality"] = pd.Categorical(df["modality"], categories=MODALITY_ORDER, ordered=True)

MODELS = sorted(df["model"].unique())
FOLDS = sorted(df["fold"].unique())

# ── Save full per-model-fold CSV ──
df.to_csv(OUT / "full_results_per_model_fold.csv", index=False, float_format="%.4f")
print(f"Saved full_results_per_model_fold.csv ({len(df)} rows)")

# ── Per-model-fold summary (averaged across pairs) ──
agg = df.groupby(["modality", "model", "fold"]).agg({
    "mae": "mean", "rmse": "mean", "pearsonr": "mean",
    "spearmanr": "mean", "r2": "mean", "n_subjects": "sum",
}).reset_index()
agg.to_csv(OUT / "per_model_fold_avg.csv", index=False, float_format="%.4f")
print(f"Saved per_model_fold_avg.csv ({len(agg)} rows)")

# ── Per-model summary (averaged across pairs & folds) ──
model_avg = df.groupby(["modality", "model"]).agg({
    "mae": "mean", "rmse": "mean", "pearsonr": "mean",
    "spearmanr": "mean",
}).reset_index()
model_avg.to_csv(OUT / "per_model_avg.csv", index=False, float_format="%.4f")

# ── Per-modality summary (grand ensemble) ──
mod_avg = model_avg.groupby("modality").agg({
    "mae": "mean", "rmse": "mean", "pearsonr": "mean", "spearmanr": "mean",
}).reset_index()
mod_avg["rank_mae"] = mod_avg["mae"].rank().astype(int)
mod_avg["rank_pearsonr"] = mod_avg["pearsonr"].rank(ascending=False).astype(int)
mod_avg.to_csv(OUT / "per_modality_summary.csv", index=False, float_format="%.4f")
print(f"Saved per_modality_summary.csv")
print(mod_avg.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════
palette = sns.color_palette("Set2", n_colors=len(MODALITY_ORDER))
mod_colors = dict(zip(MODALITY_ORDER, palette))

# ── Figure 1: Heatmap model × fold per modality (MAE) ──
fig, axes = plt.subplots(1, 5, figsize=(30, 7), sharey=True)
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    sub = agg[agg["modality"] == mod]
    pivot = sub.pivot_table(index="model", columns="fold", values="mae").reindex(MODELS)
    sns.heatmap(pivot, ax=ax, cmap="rocket_r", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
                vmin=0, vmax=agg["mae"].quantile(0.9),
                cbar=(idx == len(MODALITY_ORDER) - 1))
    short = mod.replace(" (p5)", "*")
    ax.set_title(short, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Model" if idx == 0 else "")
    ax.tick_params(axis="y", labelsize=9, rotation=0)

fig.suptitle("MAE: Model × Fold per Modality (lower is better)", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "01_heatmap_mae_model_fold.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 01_heatmap_mae_model_fold.png")

# ── Figure 2: Heatmap model × fold per modality (Pearson r) ──
fig, axes = plt.subplots(1, 5, figsize=(30, 7), sharey=True)
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    sub = agg[agg["modality"] == mod]
    pivot = sub.pivot_table(index="model", columns="fold", values="pearsonr").reindex(MODELS)
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
                vmin=-0.3, vmax=0.7, center=0.0,
                cbar=(idx == len(MODALITY_ORDER) - 1))
    short = mod.replace(" (p5)", "*")
    ax.set_title(short, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Model" if idx == 0 else "")
    ax.tick_params(axis="y", labelsize=9, rotation=0)

fig.suptitle("Pearson r: Model × Fold per Modality (higher is better)", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "02_heatmap_pearsonr_model_fold.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 02_heatmap_pearsonr_model_fold.png")

# ── Figure 3: Bar chart — mean MAE per model, grouped by modality ──
fig, ax = plt.subplots(figsize=(16, 7))
model_avg_plot = model_avg.copy()
model_avg_plot["modality"] = pd.Categorical(model_avg_plot["modality"], categories=MODALITY_ORDER, ordered=True)
sns.barplot(data=model_avg_plot, x="model", y="mae", hue="modality",
            order=MODELS, hue_order=MODALITY_ORDER, palette=palette, ax=ax)
ax.set_title("Mean MAE per Model (averaged across folds & pairs)", fontsize=15, fontweight="bold")
ax.set_ylabel("MAE (weeks)")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=35)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="Modality", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
ax.text(0.99, 0.97, "↓ lower is better", ha="right", va="top", transform=ax.transAxes,
        fontsize=11, color="dimgray",
        bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3"))
fig.tight_layout()
fig.savefig(FIG / "03_bar_mae_per_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 03_bar_mae_per_model.png")

# ── Figure 4: Bar chart — mean Pearson r per model ──
fig, ax = plt.subplots(figsize=(16, 7))
sns.barplot(data=model_avg_plot, x="model", y="pearsonr", hue="modality",
            order=MODELS, hue_order=MODALITY_ORDER, palette=palette, ax=ax)
ax.set_title("Mean Pearson r per Model (averaged across folds & pairs)", fontsize=15, fontweight="bold")
ax.set_ylabel("Pearson r")
ax.set_xlabel("Model")
ax.tick_params(axis="x", rotation=35)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="Modality", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.text(0.99, 0.97, "↑ higher is better", ha="right", va="top", transform=ax.transAxes,
        fontsize=11, color="dimgray",
        bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=0.3"))
fig.tight_layout()
fig.savefig(FIG / "04_bar_pearsonr_per_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 04_bar_pearsonr_per_model.png")

# ── Figure 5: Ranking summary — modality comparison ──
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
metrics = [("mae", "MAE (weeks)", True), ("rmse", "RMSE (weeks)", True),
           ("pearsonr", "Pearson r", False), ("spearmanr", "Spearman rho", False)]

for idx, (metric, ylabel, lower_better) in enumerate(metrics):
    ax = axes[idx]
    vals = []
    for mod in MODALITY_ORDER:
        sub = model_avg[model_avg["modality"] == mod][metric]
        vals.append(sub.values)

    bp = ax.boxplot(vals, labels=[m.replace(" (p5)", "\n(p5)") for m in MODALITY_ORDER],
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual model points
    for i, mod in enumerate(MODALITY_ORDER):
        sub = model_avg[model_avg["modality"] == mod][metric].values
        jitter = np.random.normal(0, 0.05, len(sub))
        ax.scatter(np.full(len(sub), i + 1) + jitter, sub, color="black", s=20, alpha=0.5, zorder=5)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    arrow = "↓ lower is better" if lower_better else "↑ higher is better"
    ax.text(0.98, 0.02, arrow, ha="right", va="bottom", transform=ax.transAxes,
            fontsize=8, color="dimgray")

fig.suptitle("Modality Comparison: Distribution across 11 models", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "05_modality_boxplot_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 05_modality_boxplot_comparison.png")

# ── Figure 6: Fold stability — line plot per model ──
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes_flat = axes.flat

for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes_flat[idx]
    sub = agg[agg["modality"] == mod]
    for model in MODELS:
        msub = sub[sub["model"] == model].sort_values("fold")
        ax.plot(msub["fold"], msub["mae"], "o-", markersize=4, label=model, alpha=0.7)

    ax.set_title(mod.replace(" (p5)", " (p5)"), fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_xticks(FOLDS)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2, loc="upper right")

# Remove extra subplot
axes_flat[5].set_visible(False)

fig.suptitle("Fold Stability: MAE per Model across Folds", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "06_fold_stability_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 06_fold_stability_mae.png")

# ── Figure 7: Pair comparison (pair1 vs pair2) ──
pair_avg = df.groupby(["modality", "pair"]).agg({"mae": "mean", "pearsonr": "mean", "spearmanr": "mean"}).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, (metric, title, lower) in enumerate([
    ("mae", "MAE", True), ("pearsonr", "Pearson r", False), ("spearmanr", "Spearman rho", False)
]):
    ax = axes[idx]
    pivot = pair_avg.pivot(index="modality", columns="pair", values=metric).reindex(MODALITY_ORDER)
    x = np.arange(len(MODALITY_ORDER))
    w = 0.35
    ax.bar(x - w/2, pivot["pair1"], w, label="pair1 (n=44)", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, pivot["pair2"], w, label="pair2 (n=236)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (p5)", "\n(p5)") for m in MODALITY_ORDER], fontsize=9)
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Pair1 vs Pair2 Comparison", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "07_pair_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 07_pair_comparison.png")

# ── Figure 8: Grand summary radar/table ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

table_data = []
for _, r in mod_avg.iterrows():
    table_data.append([
        r["modality"],
        f"{r['mae']:.3f}",
        f"{r['rmse']:.3f}",
        f"{r['pearsonr']:.4f}",
        f"{r['spearmanr']:.4f}",
        f"#{int(r['rank_mae'])}",
        f"#{int(r['rank_pearsonr'])}",
    ])

table = ax.table(
    cellText=table_data,
    colLabels=["Modality", "MAE", "RMSE", "Pearson r", "Spearman r", "Rank(MAE)", "Rank(r)"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color best cells
for i, row in enumerate(table_data):
    # MAE: best = lowest
    if row[5] == "#1":
        table[i + 1, 1].set_facecolor("#c8e6c9")
        table[i + 1, 5].set_facecolor("#c8e6c9")
    # Pearson: best = highest
    if row[6] == "#1":
        table[i + 1, 3].set_facecolor("#bbdefb")
        table[i + 1, 6].set_facecolor("#bbdefb")

ax.set_title("Grand Ensemble Summary (11 models × 5 folds, mean across pairs)",
             fontsize=14, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(FIG / "08_summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 08_summary_table.png")

print(f"\nAll outputs saved to: {OUT}")
print(f"Figures saved to: {FIG}")
