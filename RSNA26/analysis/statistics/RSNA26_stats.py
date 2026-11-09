"""Descriptive and inferential statistics,
restricted to two architectures: ResNet18 and ChenNet.

Methodology mirrors **effect_size_and_stats.ipynb**:
    - unit of analysis = per-seed metric value (read from *_metrics.json)
    - 2 rounds of experiments: result_0619 (seeds 0-4) and
      result_0619_seed_1234 (seeds 0-4)  →  10 values per (model, modality)
    - independent-sample t-test, Mann-Whitney U, one-way ANOVA, Cohen's d
      are computed on those 10-value groups.

Outputs CSVs in the current directory:
    rsna26_descriptive.csv     mean±SD per model×modality (MAE, RMSE, r, R², ρ)
    rsna26_anova.csv           one-way ANOVA + Kruskal-Wallis per model
    rsna26_pairwise_vs_RI.csv  RI as reference — Cohen's d + p values
    rsna26_pairwise_all.csv    all pairwise modality comparisons per model

Run:
    python "RSNA26.py"
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOTS = [
    Path("$PATH/RI_anatomical_prior/result_0619"),
    Path("$PATH/RI_anatomical_prior/result_0619_seed_1234"),
]
OUT_DIR = Path(__file__).resolve().parent
SEEDS   = [0, 1, 2, 3, 4]
MODELS  = ["resnet18", "chen_net"]                  # model dir names
MODEL_LABEL = {"resnet18": "ResNet18", "chen_net": "ChenNet"}

MODALITIES = {
    "T1w":         "t1",
    "T2w":         "t2",
    "T1w+T2w":     "t1t2",
    "RI":          "ratio",
    "T1w+T2w+RI":  "t1t2ratio",
}
MOD_ORDER = ["T1w", "T2w", "T1w+T2w", "RI", "T1w+T2w+RI"]
METRICS   = ["mae", "rmse", "r2", "pearsonr", "spearmanr"]
METRIC_LABEL = {"mae": "MAE", "rmse": "RMSE", "r2": "R2",
                "pearsonr": "r", "spearmanr": "rho"}

# ---------------------------------------------------------------------------
# Load per-seed metrics
# ---------------------------------------------------------------------------
def load_records() -> pd.DataFrame:
    rows = []
    for root in DATA_ROOTS:
        round_name = "r1" if "seed_1234" not in root.name else "r2"
        for mdir in MODELS:
            for mod_name, tag in MODALITIES.items():
                for s in SEEDS:
                    jp = root / mdir / f"{s}_{tag}_None_None_metrics.json"
                    if not jp.exists():
                        continue
                    with open(jp) as f:
                        d = json.load(f)
                    rec = {"model": mdir, "model_label": MODEL_LABEL[mdir],
                           "round": round_name, "seed": s, "modality": mod_name}
                    for m in METRICS:
                        rec[m] = d.get(m, np.nan)
                    rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def interpret_d(d):
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def pfmt(p):
    if np.isnan(p): return "nan"
    if p < 1e-4:  return "<0.0001"
    if p < 1e-3:  return f"{p:.1e}"
    return f"{p:.4f}"


def print_header(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """mean±SD per (model, modality) for every metric, using n=10 seeds."""
    rows = []
    for mdir in MODELS:
        for mod in MOD_ORDER:
            sub = df[(df.model == mdir) & (df.modality == mod)]
            row = {"Model": MODEL_LABEL[mdir], "Modality": mod, "N": len(sub)}
            for m in METRICS:
                row[f"{METRIC_LABEL[m]}_mean"] = sub[m].mean()
                row[f"{METRIC_LABEL[m]}_std"]  = sub[m].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def omnibus(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA + Kruskal-Wallis across 5 modalities, per model, per metric.

    (Also runs pooled over both models as a sanity check.)"""
    rows = []
    for mdir in list(MODELS) + ["(pooled)"]:
        for m in METRICS:
            groups = []
            for mod in MOD_ORDER:
                if mdir == "(pooled)":
                    g = df[df.modality == mod][m].dropna().values
                else:
                    g = df[(df.model == mdir) & (df.modality == mod)][m].dropna().values
                groups.append(g)
            f_stat, f_p = stats.f_oneway(*groups)
            kw = stats.kruskal(*groups)
            rows.append({
                "Model":   MODEL_LABEL.get(mdir, mdir),
                "Metric":  METRIC_LABEL[m],
                "k":       len(groups),
                "N_per_grp": "/".join(str(len(g)) for g in groups),
                "F":       f_stat, "p_ANOVA":   f_p,
                "H":       kw.statistic, "p_Kruskal": kw.pvalue,
            })
    return pd.DataFrame(rows)


def pairwise_vs_RI(df: pd.DataFrame) -> pd.DataFrame:
    """RI-as-reference pairwise comparisons per model, per metric."""
    rows = []
    for mdir in MODELS:
        ref_data = {m: df[(df.model == mdir) & (df.modality == "RI")][m].dropna().values
                    for m in METRICS}
        for mod in MOD_ORDER:
            if mod == "RI": continue
            sub = df[(df.model == mdir) & (df.modality == mod)]
            for m in METRICS:
                a = ref_data[m]
                b = sub[m].dropna().values
                if len(a) < 2 or len(b) < 2: continue
                t = stats.ttest_ind(a, b, equal_var=False)   # Welch
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                d = cohens_d(a, b)
                rows.append({
                    "Model":  MODEL_LABEL[mdir],
                    "Metric": METRIC_LABEL[m],
                    "Comparison": f"RI vs {mod}",
                    "RI_mean":     a.mean(), "RI_std":     a.std(ddof=1),
                    "other_mean":  b.mean(), "other_std":  b.std(ddof=1),
                    "Welch_t": t.statistic, "p_ttest":       t.pvalue,
                    "MWU":     u.statistic, "p_mannwhitney": u.pvalue,
                    "Cohen_d": d, "Interpretation": interpret_d(d),
                })
    return pd.DataFrame(rows)


def pairwise_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mdir in MODELS:
        for m in METRICS:
            for ma, mb in combinations(MOD_ORDER, 2):
                a = df[(df.model == mdir) & (df.modality == ma)][m].dropna().values
                b = df[(df.model == mdir) & (df.modality == mb)][m].dropna().values
                if len(a) < 2 or len(b) < 2: continue
                t = stats.ttest_ind(a, b, equal_var=False)
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                d = cohens_d(a, b)
                rows.append({
                    "Model":  MODEL_LABEL[mdir], "Metric": METRIC_LABEL[m],
                    "Comparison": f"{ma} vs {mb}",
                    "A_mean": a.mean(), "A_std": a.std(ddof=1),
                    "B_mean": b.mean(), "B_std": b.std(ddof=1),
                    "Welch_t": t.statistic, "p_ttest":       t.pvalue,
                    "MWU":     u.statistic, "p_mannwhitney": u.pvalue,
                    "Cohen_d": d, "Interpretation": interpret_d(d),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw = load_records()
    print_header(f"Loaded {len(raw)} records "
                 f"({raw['model'].nunique()} models × {raw['round'].nunique()} rounds "
                 f"× {raw['seed'].nunique()} seeds × {raw['modality'].nunique()} modalities)")
    print(raw.groupby(["model_label", "modality"]).size().unstack(fill_value=0)[MOD_ORDER])

    desc = descriptive(raw)
    anov = omnibus(raw)
    pr_R = pairwise_vs_RI(raw)
    pr_A = pairwise_all(raw)

    # ---- persist (numeric, machine-readable) -------------------------------
    desc.to_csv(OUT_DIR / "rsna26_descriptive.csv",      index=False)
    anov.to_csv(OUT_DIR / "rsna26_anova.csv",            index=False)
    pr_R.to_csv(OUT_DIR / "rsna26_pairwise_vs_RI.csv",   index=False)
    pr_A.to_csv(OUT_DIR / "rsna26_pairwise_all.csv",     index=False)

    # ---- display -----------------------------------------------------------
    print_header("1. Descriptive statistics (n=10 seeds per cell)  — MAE and r")
    tbl = desc.copy()
    tbl["MAE"] = tbl.apply(lambda r: f"{r.MAE_mean:.3f} ± {r.MAE_std:.3f}", axis=1)
    tbl["r"]   = tbl.apply(lambda r: f"{r.r_mean:.3f} ± {r.r_std:.3f}",   axis=1)
    tbl["RMSE"] = tbl.apply(lambda r: f"{r.RMSE_mean:.3f} ± {r.RMSE_std:.3f}", axis=1)
    tbl["R²"]  = tbl.apply(lambda r: f"{r.R2_mean:.3f} ± {r.R2_std:.3f}", axis=1)
    tbl["ρ"]   = tbl.apply(lambda r: f"{r.rho_mean:.3f} ± {r.rho_std:.3f}", axis=1)
    print(tbl[["Model", "Modality", "N", "MAE", "RMSE", "R²", "r", "ρ"]].to_string(index=False))

    print_header("2. Omnibus tests across 5 modalities (per model, MAE)")
    av = anov[anov.Metric == "MAE"].copy()
    av["p_ANOVA"]   = av["p_ANOVA"].map(pfmt)
    av["p_Kruskal"] = av["p_Kruskal"].map(pfmt)
    print(av[["Model", "k", "N_per_grp", "F", "p_ANOVA", "H", "p_Kruskal"]].to_string(index=False))

    print_header("3. Pairwise (RI as reference, MAE)")
    pr = pr_R[pr_R.Metric == "MAE"].copy()
    pr["Cohen_d"] = pr.Cohen_d.map(lambda x: f"{x:.3f}")
    pr["p_ttest"]       = pr["p_ttest"].map(pfmt)
    pr["p_mannwhitney"] = pr["p_mannwhitney"].map(pfmt)
    print(pr[["Model", "Comparison", "RI_mean", "other_mean",
              "Cohen_d", "Interpretation", "p_ttest", "p_mannwhitney"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nAll CSVs saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
