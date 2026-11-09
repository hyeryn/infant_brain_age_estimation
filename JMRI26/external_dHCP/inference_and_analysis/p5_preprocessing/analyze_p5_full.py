"""
Full p5-preprocessing analysis — matching *nice_preprocessing format.

1. Original (raw) performance per modality
2. Bias correction analysis
3. Per-subject GT vs Pred scatter
4. Per-subject error distribution
5. QC-filtered subsets for RI(p5), T1w+T2w+RI(p5)
6. All figures + CSVs
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

BASE = Path("$PATH/DHCP/JMRI_inference")
OUT = BASE / "p5_preprocessing"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
FIG_BC = OUT / "figures_biascorr"
FIG_BC.mkdir(exist_ok=True)

QC_CSV = BASE / "nice_preprocessing" / "subject_qc_list.csv"

sns.set_theme(style="whitegrid", context="talk")

MODALITY_ORDER = ["T1w", "T2w", "T1w+T2w", "RI (p5)", "T1w+T2w+RI (p5)"]
CONFIGS = {
    "T1w":              ("t1",        "pair{p}"),
    "T2w":              ("t2",        "pair{p}"),
    "T1w+T2w":          ("t1t2",      "pair{p}"),
    "RI (p5)":          ("ratio",     "pair{p}_improved"),
    "T1w+T2w+RI (p5)":  ("t1t2ratio", "pair{p}_improved"),
}
MODELS = [
    "agenet", "basiccnn", "cbr_tiny", "chen_net", "densenet",
    "efficientnet", "resnet18", "resnet34", "resnet50", "sfcn", "unet",
]
FOLDS = list(range(5))

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
print("Loading predictions...")

def load_all_predictions(modality_label):
    ot, pair_tmpl = CONFIGS[modality_label]
    pair_results = []
    for p in ["1", "2"]:
        pair_dir = pair_tmpl.format(p=p)
        model_dfs = []
        for model in MODELS:
            fold_preds = []
            for fold in FOLDS:
                csv_path = BASE / ot / pair_dir / f"{model}_fold{fold}_predictions.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                df = df.rename(columns={"age_pred": f"pred_f{fold}"})
                fold_preds.append(df[["subject_id", "age_gt", f"pred_f{fold}"]])
            if not fold_preds:
                continue
            merged = fold_preds[0]
            for fp in fold_preds[1:]:
                merged = merged.merge(fp.drop(columns=["age_gt"]), on="subject_id", how="outer")
            pred_cols = [c for c in merged.columns if c.startswith("pred_f")]
            merged[f"ensemble_{model}"] = merged[pred_cols].mean(axis=1)
            model_dfs.append(merged[["subject_id", "age_gt", f"ensemble_{model}"]])

        if not model_dfs:
            continue
        pair_df = model_dfs[0]
        for md in model_dfs[1:]:
            pair_df = pair_df.merge(md.drop(columns=["age_gt"], errors="ignore"),
                                    on="subject_id", how="outer")
        pair_df["pair"] = f"pair{p}"
        pair_results.append(pair_df)

    if not pair_results:
        return pd.DataFrame()
    result = pd.concat(pair_results, ignore_index=True)
    # Grand ensemble
    ens_cols = [c for c in result.columns if c.startswith("ensemble_")]
    result["grand_ensemble"] = result[ens_cols].mean(axis=1)
    return result


pred_data = {}
for mod in MODALITY_ORDER:
    pred_data[mod] = load_all_predictions(mod)
    n = len(pred_data[mod])
    print(f"  {mod}: {n} subjects")

# ── Load QC data ──
qc_df = pd.read_csv(QC_CSV)
qc_per_subj = qc_df.groupby("subject_id")["ratio_max"].min().reset_index()

# ══════════════════════════════════════════════════════════════
# Compute metrics
# ══════════════════════════════════════════════════════════════
def compute_metrics(gt, pred):
    n = len(gt)
    error = pred - gt
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)
    ss_res = np.sum(error**2)
    ss_tot = np.sum((gt - gt.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pr = sp_stats.pearsonr(gt, pred)[0] if n >= 3 else np.nan
    sr = sp_stats.spearmanr(gt, pred)[0] if n >= 3 else np.nan
    # Bias corrected
    corr_pred = pred - bias
    corr_mae = np.mean(np.abs(corr_pred - gt))
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearsonr": pr,
            "spearmanr": sr, "bias": bias, "corr_mae": corr_mae, "n": n}


# ── Per-modality summary (grand ensemble) ──
print("\n" + "="*70)
print("PER-MODALITY SUMMARY (grand ensemble, all subjects)")
print("="*70)

mod_summary_rows = []
for mod in MODALITY_ORDER:
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"])
    gt = df["age_gt"].values
    pred = df["grand_ensemble"].values
    m = compute_metrics(gt, pred)
    m["modality"] = mod
    mod_summary_rows.append(m)
    print(f"  {mod:>20s}: MAE={m['mae']:.4f} cMAE={m['corr_mae']:.4f} "
          f"r={m['pearsonr']:.4f} rho={m['spearmanr']:.4f} bias={m['bias']:.4f} n={m['n']}")

mod_summary = pd.DataFrame(mod_summary_rows)
mod_summary["rank"] = mod_summary["mae"].rank().astype(int)
mod_summary.to_csv(OUT / "per_outputtype_avg.csv", index=False, float_format="%.4f")

# ── Bias-corrected summary ──
biascorr_rows = []
for mod in MODALITY_ORDER:
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"])
    gt = df["age_gt"].values
    pred = df["grand_ensemble"].values
    m = compute_metrics(gt, pred)
    ot = CONFIGS[mod][0]
    biascorr_rows.append({
        "modality": mod, "output_type": ot,
        "raw_mae": m["mae"], "corr_mae": m["corr_mae"],
        "raw_bias": m["bias"], "pearsonr": m["pearsonr"], "spearmanr": m["spearmanr"],
    })

biascorr_df = pd.DataFrame(biascorr_rows)
biascorr_df.to_csv(OUT / "per_outputtype_biascorr.csv", index=False, float_format="%.4f")
print("\nSaved per_outputtype_biascorr.csv")

# ── Per-model summary ──
model_rows = []
for mod in MODALITY_ORDER:
    df = pred_data[mod]
    for model in MODELS:
        col = f"ensemble_{model}"
        if col not in df.columns:
            continue
        valid = df[["age_gt", col]].dropna()
        if len(valid) < 3:
            continue
        m = compute_metrics(valid["age_gt"].values, valid[col].values)
        m["modality"] = mod
        m["model"] = model
        model_rows.append(m)

model_summary = pd.DataFrame(model_rows)
model_summary = model_summary.reset_index(drop=True)
model_summary.to_csv(OUT / "per_model_avg_full.csv", index=False, float_format="%.4f")

# ══════════════════════════════════════════════════════════════
# FIGURES — Original
# ══════════════════════════════════════════════════════════════
palette = sns.color_palette("Set2", n_colors=5)
mod_colors = dict(zip(MODALITY_ORDER, palette))

# ── Fig 09: Ranking bar (MAE + bias-corrected MAE) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
order = mod_summary.sort_values("mae")["modality"].tolist()
colors = [mod_colors[m] for m in order]
bars = ax.barh(range(len(order)), [mod_summary[mod_summary["modality"]==m]["mae"].values[0] for m in order],
               color=colors, edgecolor="gray", linewidth=0.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=11)
ax.set_xlabel("MAE (weeks)")
ax.set_title("Raw MAE Ranking", fontsize=14, fontweight="bold")
for i, m in enumerate(order):
    val = mod_summary[mod_summary["modality"]==m]["mae"].values[0]
    ax.text(val + 0.03, i, f"{val:.3f}", va="center", fontsize=10)
ax.invert_yaxis()

ax = axes[1]
order_bc = biascorr_df.sort_values("corr_mae")["modality"].tolist()
colors_bc = [mod_colors[m] for m in order_bc]
bars = ax.barh(range(len(order_bc)),
               [biascorr_df[biascorr_df["modality"]==m]["corr_mae"].values[0] for m in order_bc],
               color=colors_bc, edgecolor="gray", linewidth=0.5)
ax.set_yticks(range(len(order_bc)))
ax.set_yticklabels(order_bc, fontsize=11)
ax.set_xlabel("Bias-corrected MAE (weeks)")
ax.set_title("Bias-corrected MAE Ranking", fontsize=14, fontweight="bold")
for i, m in enumerate(order_bc):
    val = biascorr_df[biascorr_df["modality"]==m]["corr_mae"].values[0]
    ax.text(val + 0.03, i, f"{val:.3f}", va="center", fontsize=10)
ax.invert_yaxis()

fig.suptitle("Modality Ranking: Raw vs Bias-corrected MAE", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "09_ranking_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 09_ranking_bar.png")

# ── Fig 10: GT vs Pred scatter (grand ensemble) ──
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"])
    gt = df["age_gt"].values
    pred = df["grand_ensemble"].values
    m = compute_metrics(gt, pred)

    ax.scatter(gt, pred, s=15, alpha=0.5, color=mod_colors[mod], edgecolors="none")
    lims = [min(gt.min(), pred.min()) - 0.5, max(gt.max(), pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GT age (weeks)")
    ax.set_ylabel("Pred age (weeks)" if idx == 0 else "")
    ax.set_title(mod.replace(" (p5)", "*"), fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    stats_txt = f"MAE={m['mae']:.2f}\nr={m['pearsonr']:.3f}\nrho={m['spearmanr']:.3f}\nbias={m['bias']:.2f}"
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.9, boxstyle="round,pad=0.3"))

fig.suptitle("GT vs Predicted Age (Grand Ensemble)", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "10_gt_vs_pred_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 10_gt_vs_pred_scatter.png")

# ── Fig 11: Per-subject error distribution ──
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"])
    errors = df["grand_ensemble"].values - df["age_gt"].values
    ax.hist(errors, bins=40, color=mod_colors[mod], alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(np.mean(errors), color="red", linestyle="-", linewidth=1.5, label=f"bias={np.mean(errors):.2f}")
    ax.set_xlabel("Pred - GT (weeks)")
    ax.set_ylabel("Count" if idx == 0 else "")
    ax.set_title(mod.replace(" (p5)", "*"), fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

fig.suptitle("Per-subject Error Distribution (Grand Ensemble)", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "11_per_subject_error.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 11_per_subject_error.png")

# ── Fig 12: Subject age distribution ──
fig, ax = plt.subplots(figsize=(10, 5))
df0 = pred_data[MODALITY_ORDER[0]]
ax.hist(df0["age_gt"].dropna(), bins=30, color="steelblue", alpha=0.7, edgecolor="white")
ax.set_xlabel("Postnatal age (weeks)")
ax.set_ylabel("Count")
ax.set_title("dHCP Subject Age Distribution (all pairs)", fontsize=14, fontweight="bold")
ax.axvline(df0["age_gt"].mean(), color="red", linestyle="--", label=f"mean={df0['age_gt'].mean():.2f}")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "12_subject_age_dist.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 12_subject_age_dist.png")

# ══════════════════════════════════════════════════════════════
# FIGURES — Bias Correction
# ══════════════════════════════════════════════════════════════

# ── Fig BC01: Raw vs Corrected MAE ──
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(MODALITY_ORDER))
w = 0.35
raw_vals = [biascorr_df[biascorr_df["modality"]==m]["raw_mae"].values[0] for m in MODALITY_ORDER]
corr_vals = [biascorr_df[biascorr_df["modality"]==m]["corr_mae"].values[0] for m in MODALITY_ORDER]

bars1 = ax.bar(x - w/2, raw_vals, w, label="Raw MAE", color="steelblue", alpha=0.8)
bars2 = ax.bar(x + w/2, corr_vals, w, label="Bias-corrected MAE", color="coral", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([m.replace(" (p5)", "\n(p5)") for m in MODALITY_ORDER], fontsize=10)
ax.set_ylabel("MAE (weeks)")
ax.set_title("Raw vs Bias-corrected MAE per Modality", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars1, raw_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.2f}", ha="center", fontsize=9)
for bar, val in zip(bars2, corr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.2f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(FIG_BC / "01_raw_vs_corrected_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved biascorr/01_raw_vs_corrected_mae.png")

# ── Fig BC02: GT vs Pred (bias-corrected) ──
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"])
    gt = df["age_gt"].values
    pred = df["grand_ensemble"].values
    bias = np.mean(pred - gt)
    pred_corr = pred - bias
    m = compute_metrics(gt, pred_corr)

    ax.scatter(gt, pred_corr, s=15, alpha=0.5, color=mod_colors[mod], edgecolors="none")
    lims = [min(gt.min(), pred_corr.min()) - 0.5, max(gt.max(), pred_corr.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GT age (weeks)")
    ax.set_ylabel("Corrected pred (weeks)" if idx == 0 else "")
    ax.set_title(mod.replace(" (p5)", "*"), fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    stats_txt = f"cMAE={m['mae']:.2f}\nr={m['pearsonr']:.3f}\nrho={m['spearmanr']:.3f}"
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.9, boxstyle="round,pad=0.3"))

fig.suptitle("GT vs Bias-corrected Predicted Age (Grand Ensemble)", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG_BC / "02_gt_vs_pred_corrected.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved biascorr/02_gt_vs_pred_corrected.png")

# ══════════════════════════════════════════════════════════════
# Using ratio_max from QC data + per-subject error
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("QC-FILTERED ANALYSIS")
print("="*70)

# For each modality, compute per-subject grand ensemble error
subj_results = {}
for mod in MODALITY_ORDER:
    df = pred_data[mod].dropna(subset=["age_gt", "grand_ensemble"]).copy()
    df["error"] = df["grand_ensemble"] - df["age_gt"]
    df["abs_error"] = np.abs(df["error"])
    df = df.merge(qc_per_subj, on="subject_id", how="left")
    subj_results[mod] = df

# QC thresholds to test
QC_THRESHOLDS = [3.0, 4.0, 999]  # 999 = no filter

qc_rows = []
for mod in MODALITY_ORDER:
    df = subj_results[mod]
    for thr in QC_THRESHOLDS:
        if thr < 100:
            sub = df[df["ratio_max"] < thr]
            thr_label = f"<{thr:.0f}"
        else:
            sub = df
            thr_label = "all"
        if len(sub) < 5:
            continue
        gt = sub["age_gt"].values
        pred = sub["grand_ensemble"].values
        m = compute_metrics(gt, pred)
        qc_rows.append({
            "modality": mod, "qc_threshold": thr_label,
            "n": m["n"], "mae": m["mae"], "corr_mae": m["corr_mae"],
            "pearsonr": m["pearsonr"], "spearmanr": m["spearmanr"], "bias": m["bias"],
        })

qc_df_result = pd.DataFrame(qc_rows)
qc_df_result.to_csv(OUT / "qc_filtered_comparison.csv", index=False, float_format="%.4f")

print("\nQC-filtered results:")
print(f"{'Modality':>20s} {'QC':>5s} {'N':>5s} {'MAE':>8s} {'cMAE':>8s} {'r':>8s} {'rho':>8s} {'bias':>8s}")
print("-"*75)
for _, r in qc_df_result.iterrows():
    print(f"{r['modality']:>20s} {r['qc_threshold']:>5s} {int(r['n']):>5d} "
          f"{r['mae']:>8.3f} {r['corr_mae']:>8.3f} {r['pearsonr']:>8.4f} "
          f"{r['spearmanr']:>8.4f} {r['bias']:>8.3f}")

# ── Fig 13: QC-filtered comparison bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
metrics_plot = [("mae", "MAE (weeks)"), ("corr_mae", "Bias-corr MAE"), ("pearsonr", "Pearson r")]

for idx, (metric, ylabel) in enumerate(metrics_plot):
    ax = axes[idx]
    pivot = qc_df_result.pivot(index="modality", columns="qc_threshold", values=metric)
    pivot = pivot.reindex(MODALITY_ORDER)
    cols_order = ["all", "<4", "<3"]
    cols_exist = [c for c in cols_order if c in pivot.columns]
    pivot = pivot[cols_exist]

    x = np.arange(len(MODALITY_ORDER))
    w = 0.25
    for j, col in enumerate(cols_exist):
        offset = (j - len(cols_exist)/2 + 0.5) * w
        vals = pivot[col].values
        label = f"QC {col}" if col != "all" else "No filter"
        ax.bar(x + offset, vals, w, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (p5)", "\n(p5)") for m in MODALITY_ORDER], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Effect of QC Filtering on Performance", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "13_qc_filtered_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 13_qc_filtered_comparison.png")

# ── Fig 14: Per-subject error vs QC ratio_max ──
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    df = subj_results[mod].dropna(subset=["ratio_max"])
    sc = ax.scatter(df["ratio_max"], df["abs_error"], s=15, alpha=0.5,
                    c=mod_colors[mod], edgecolors="none")
    ax.axvline(3.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="QC=3.0")
    ax.axvline(4.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="QC=4.0")
    ax.set_xlabel("Ratio max (QC)")
    ax.set_ylabel("|Error| (weeks)" if idx == 0 else "")
    ax.set_title(mod.replace(" (p5)", "*"), fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 16)

fig.suptitle("Per-subject |Error| vs Ratio Max (QC metric)", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "14_error_vs_qc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 14_error_vs_qc.png")

# ── Fig 15: Model heatmap (per-model MAE, bias-corrected) ──
fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=True)
for idx, mod in enumerate(MODALITY_ORDER):
    ax = axes[idx]
    sub = model_summary[model_summary["modality"] == mod]
    data = sub.set_index("model")[["mae", "corr_mae", "pearsonr", "spearmanr"]].reindex(MODELS)
    sns.heatmap(data, ax=ax, cmap="rocket_r", annot=True, fmt=".2f",
                annot_kws={"size": 9}, linewidths=0.4, linecolor="white",
                cbar=(idx == len(MODALITY_ORDER)-1))
    ax.set_title(mod.replace(" (p5)", "*"), fontsize=12, fontweight="bold")
    ax.set_ylabel("" if idx > 0 else "Model")
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=9)

fig.suptitle("Per-model Metrics (MAE, cMAE, Pearson r, Spearman rho)", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "15_model_metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 15_model_metrics_heatmap.png")

# ── Selection info ──
info = f"""p5-Preprocessed dHCP External Validation
============================================================

Ratio Preprocessing: T2w > p5 masking before T1w/T2w ratio computation.
  - For each subject, T2w brain voxels below the 5th percentile are excluded
  - Removes near-zero T2w boundary voxels that cause ratio explosion
  - Remaining ratio values clipped at 98th percentile
  - Applied to: RI (p5), T1w+T2w+RI (p5) modalities

Total subjects: {len(pred_data[MODALITY_ORDER[0]])} (pair1 + pair2)
Models: {len(MODELS)} ({', '.join(MODELS)})
Folds: ALL (fold-mean ensemble)

Grand Ensemble Results:
"""
for _, r in mod_summary.iterrows():
    info += f"  {r['modality']:>20s}: MAE={r['mae']:.4f} cMAE={r['corr_mae']:.4f} r={r['pearsonr']:.4f} rank={int(r['rank'])}\n"

with open(OUT / "selection_info.txt", "w") as f:
    f.write(info)
print("\nSaved selection_info.txt")

print(f"\nAll outputs saved to: {OUT}")
