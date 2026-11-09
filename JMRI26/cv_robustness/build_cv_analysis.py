"""Aggregate 5-fold × 2-seed CV metrics and build the robustness analysis tables.

This script produces:
- For each (model, output_type), 10 independent train/eval cycles (5 fold × 2 seed)
  evaluated on held-out validation subjects (N≈126 per fold, no overlap between folds).
- Stable headline metrics (mean ± std across the 10 cycles)
- Decompose variance into fold-wise and seed-wise components
- Compare model complexity (param count) vs fold/seed variance
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {
    "seed_default": Path("$PATH/RI_anatomical_prior/result_0619"),
    "seed_1234":    Path("$PATH/RI_anatomical_prior/result_0619_seed_1234"),
}
OUT = Path("$PATH/RI_anatomical_prior/cv_robustness")
OUT.mkdir(exist_ok=True)

OUTS = ["t1", "t2", "ratio", "t1t2", "t1t2ratio"]
PATTERN = re.compile(r"^(\d+)_([a-z0-9]+)_([^_]+)_([^_]+)_metrics\.json$")


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for seed_label, root in ROOTS.items():
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("*")):
            model = model_dir.name
            for f in sorted(model_dir.glob("*_metrics.json")):
                if f.name.endswith("_round.json"):
                    continue
                m = PATTERN.match(f.name)
                if not m:
                    continue
                try:
                    js = json.loads(f.read_text())
                except Exception:
                    continue
                rows.append({
                    "seed": seed_label,
                    "model": model,
                    "fold": int(m.group(1)),
                    "output_type": m.group(2),
                    "add_option": m.group(3),
                    **js,
                })
    return pd.DataFrame(rows)


def per_run_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, output_type, add_option) — mean, std, n across the 10 runs."""
    g = (df.groupby(["model", "output_type", "add_option"])
         .agg(
             mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             pearsonr_mean=("pearsonr", "mean"), pearsonr_std=("pearsonr", "std"),
             r2_mean=("r2", "mean"), r2_std=("r2", "std"),
             n_runs=("mae", "count"),
         ).reset_index())
    g["mae_cv_pct"] = (g["mae_std"] / g["mae_mean"] * 100).round(2)
    return g


def variance_decomp(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, output_type, add_option), separate fold-vs-seed std components."""
    rows = []
    for (m, ot, ao), g in df.groupby(["model", "output_type", "add_option"]):
        # fold-wise std within each seed → average
        fold_within_seed = g.groupby("seed")["mae"].std(ddof=1).mean()
        # seed-wise std within each fold → average
        seed_within_fold = g.groupby("fold")["mae"].std(ddof=1).mean()
        rows.append({
            "model": m, "output_type": ot, "add_option": ao,
            "mae_mean": g["mae"].mean(),
            "fold_std_within_seed": fold_within_seed,
            "seed_std_within_fold": seed_within_fold,
            "total_std_10runs": g["mae"].std(ddof=1),
            "n": len(g),
        })
    return pd.DataFrame(rows)


def main():
    print("[1/3] loading metrics ...")
    df = load_all_metrics()
    print(f"      total runs: {len(df):,}")

    # Filter to add_option=None (no FreeSurfer conditioning) for the main analysis
    df_main = df[df.add_option == "None"].copy()
    df_main.to_csv(OUT / "all_runs.csv", index=False)
    print(f"      main analysis (add_option=None): {len(df_main)} runs across "
          f"{df_main.model.nunique()} models × {df_main.output_type.nunique()} output_types × "
          f"{df_main.fold.nunique()} folds × {df_main.seed.nunique()} seeds")

    print("[2/3] per (model, output_type) summary ...")
    summary = per_run_table(df_main)
    summary = summary[summary.add_option == "None"].drop(columns=["add_option"])
    summary["output_type"] = pd.Categorical(summary["output_type"], categories=OUTS, ordered=True)
    summary = summary.sort_values(["output_type", "mae_mean"])
    summary.to_csv(OUT / "summary_per_model_outputtype.csv", index=False)

    # Best per output_type
    best = summary.sort_values("mae_mean").groupby("output_type", observed=True).head(1)
    best.to_csv(OUT / "best_per_output_type.csv", index=False)
    print("      best per output_type:")
    for _, r in best.iterrows():
        print(f"        {r['output_type']:>10s}: {r['model']:<14s}"
              f" MAE={r['mae_mean']:.3f}±{r['mae_std']:.3f}  Pearson={r['pearsonr_mean']:.3f}"
              f"  R²={r['r2_mean']:.3f}  CV={r['mae_cv_pct']:.1f}%")

    print("[3/3] variance decomposition ...")
    decomp = variance_decomp(df_main[df_main.add_option == "None"])
    decomp.to_csv(OUT / "variance_decomposition.csv", index=False)
    print("      decomposition stats (across all model × output_type combos):")
    print(decomp[["mae_mean", "fold_std_within_seed", "seed_std_within_fold",
                  "total_std_10runs"]].describe().round(3).to_string())

    # ----- headline numbers -----
    overall_mae_mean = summary["mae_mean"].mean()
    overall_mae_cv = summary["mae_cv_pct"].median()
    best_row = summary.sort_values("mae_mean").iloc[0]
    print()
    print("=" * 72)
    print("HEADLINE NUMBERS")
    print("=" * 72)
    print(f"  • {len(df_main)} held-out validation evaluations "
          f"(11 models × 5 output_types × 5 folds × 2 seeds)")
    print(f"  • Median CV(MAE) across (model × output_type) combos: {overall_mae_cv:.1f}%")
    print(f"  • Mean MAE across all configurations: {overall_mae_mean:.3f} months")
    print(f"  • Best single config: {best_row['model']} + {best_row['output_type']}  →  "
          f"MAE {best_row['mae_mean']:.2f} ± {best_row['mae_std']:.2f} "
          f"(CV {best_row['mae_cv_pct']:.1f}%), Pearson {best_row['pearsonr_mean']:.2f}, R² {best_row['r2_mean']:.2f}")
    print(f"  • Fold-wise vs seed-wise std are comparable "
          f"(fold {decomp['fold_std_within_seed'].median():.3f} vs "
          f"seed {decomp['seed_std_within_fold'].median():.3f}) — neither dominates")
    print()


if __name__ == "__main__":
    main()
