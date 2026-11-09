"""
Re-run dHCP inference with norm_ignore_zero=True (nonzero normalization).

Key change: NormalizeIntensity(nonzero=True) — computes mean/std from brain
voxels only (excluding background zeros), reducing domain gap between
Infancy (training) and dHCP (external validation).

Output directories: {output_type}/pair{1,2}_nonzero/
"""
import sys
import os
from pathlib import Path

# Add RI_anatomical_prior to path for model/transform imports
RI_ROOT = Path("$PATH/RI_anatomical_prior")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

import json
import time
import yaml
import warnings
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from classifier.lit_classifier import LitClassifier
from classifier.utils_loading import _add_missing_hparams
from classifier.transformations import get_transforms
from libs.dp_utils import num2vect_torch

warnings.filterwarnings("ignore")
os.environ["lightning_quiet"] = "1"

# ============================================================
# Config
# ============================================================
DHCP_ROOT = Path("$PATH/DHCP")
LOG_ROOT = RI_ROOT / "log" / "basic_0619"
OUTPUT_ROOT = DHCP_ROOT / "JMRI_inference"

PAIRS = ["pair1", "pair2"]
OUTPUT_TYPES = ["t1", "t2", "ratio", "t1t2", "t1t2ratio"]

IMG_FILE_MAP = {
    "t1_restore_regist2_15.nii.gz": ("t1w_reg2atlas", "_T1w_reg2atlas.nii.gz"),
    "t2_restore_regist2_15.nii.gz": ("t2w_reg2atlas", "_T2w_reg2atlas.nii.gz"),
    "ratio_regist_15_modified.nii.gz": ("ratio", "_ratio_clipped_98.nii.gz"),
}

# Output suffix for nonzero normalization
PAIR_SUFFIX = "_nonzero"


# ============================================================
# DHCP Dataset (identical to original)
# ============================================================
class DHCPDataset(Dataset):
    def __init__(self, pair_path, subject_ids, labels, img_files, transform,
                 dim=3, zoom=[1.0, 1.0, 1.0], loss="mse"):
        self.pair_path = Path(pair_path)
        self.subject_ids = subject_ids
        self.labels = labels
        self.img_files = img_files
        self.transform = transform
        self.dim = dim
        self.zoom = zoom
        self.loss = loss

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        sid = self.subject_ids[index]
        y = self.labels[index]

        x_all = []
        for img_file in self.img_files:
            dhcp_subdir, suffix = IMG_FILE_MAP[img_file]
            fpath = self.pair_path / dhcp_subdir / f"{sid}{suffix}"
            x = nib.load(str(fpath)).get_fdata()

            if np.any(np.array(self.zoom) != 1.0):
                from scipy import ndimage
                x = ndimage.zoom(x, self.zoom, order=0)

            x = x[None, ...]  # add channel dim: (1, D, H, W)
            x_all.append(x)

        x = np.array(x_all)
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]])

        x = self.transform(x) if self.transform else x
        x = x.float()

        if self.loss.startswith("mse") or self.loss.startswith("mae"):
            label = np.array(y).astype(np.float32)
        elif self.loss == "kld":
            label = np.array(y).astype(np.float32)
        else:
            label = np.array(y).astype(np.float32)

        dummy = np.float32(0.0)
        return x, label, dummy, dummy, dummy


# ============================================================
# Inference Function — with norm_ignore_zero override
# ============================================================
def run_single_inference(ckpt_path, pair_path, device="cuda:0"):
    """Run inference for a single checkpoint on a single DHCP pair.
    Override: norm_ignore_zero = True for nonzero normalization.
    """
    model = LitClassifier.load_from_checkpoint(str(ckpt_path), map_location="cpu",
                                                weights_only=False)
    model = _add_missing_hparams(model)
    model.eval().to(device)

    hp = model.hparams

    # ── KEY CHANGE: override norm_ignore_zero to True ──
    hp.norm_ignore_zero = True

    _, tfs_val = get_transforms(hp)
    img_files = hp.img_files
    img_files = [f.replace(".npy", ".nii.gz") for f in img_files]

    # Load labels
    pair_path = Path(pair_path)
    labels_df = pd.read_csv(pair_path / "labels.csv")
    subject_ids = labels_df["subject_id"].tolist()
    labels = labels_df["postnatal_months"].values.astype(np.float32)

    # Filter subjects that have all required files
    valid_mask = []
    for sid in subject_ids:
        ok = True
        for img_file in img_files:
            dhcp_subdir, suffix = IMG_FILE_MAP[img_file]
            if not (pair_path / dhcp_subdir / f"{sid}{suffix}").exists():
                ok = False
                break
        valid_mask.append(ok)
    valid_mask = np.array(valid_mask)
    subject_ids = [s for s, v in zip(subject_ids, valid_mask) if v]
    labels = labels[valid_mask]

    if len(subject_ids) == 0:
        return None

    dataset = DHCPDataset(
        pair_path=pair_path,
        subject_ids=subject_ids,
        labels=labels,
        img_files=img_files,
        transform=tfs_val,
        dim=hp.dim,
        zoom=hp.zoom,
        loss=hp.loss,
    )
    loader = DataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True)

    gts = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            x, y, wm, icv, ratio = batch
            x = x.to(device)
            y_hat = model(x)

            if hp.loss.startswith("mse") or hp.loss.startswith("mae"):
                y_hat = y_hat[:, 0]
            elif hp.loss == "kld":
                # Decode KLD output
                _, bc = num2vect_torch(y.to(device), [-1, 36], 1, 1)
                y_hat = (torch.exp(y_hat).float()) @ (bc.float())

            gts.append(y.cpu().numpy())
            preds.append(y_hat.cpu().numpy())

    gts = np.concatenate(gts).flatten()
    preds = np.concatenate(preds).flatten()

    # Compute metrics
    mae = float(mean_absolute_error(gts, preds))
    rmse = float(np.sqrt(mean_squared_error(gts, preds)))
    pr = float(pearsonr(gts, preds)[0]) if len(gts) > 2 else 0.0
    sr = float(spearmanr(gts, preds)[0]) if len(gts) > 2 else 0.0
    r2 = float(r2_score(gts, preds)) if len(gts) > 2 else 0.0

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearsonr": round(pr, 4),
        "spearmanr": round(sr, 4),
        "r2": round(r2, 4),
        "n_subjects": len(gts),
    }

    predictions = pd.DataFrame({
        "subject_id": subject_ids,
        "age_gt": gts.round(4),
        "age_pred": preds.round(4),
        "error": np.abs(gts - preds).round(4),
    })

    return metrics, predictions


# ============================================================
# Discover all checkpoints for an output type
# ============================================================
def discover_checkpoints(output_type):
    """Scan all versions for an output type, return list of (ckpt_path, model_name, fold, add_option)."""
    log_dir = LOG_ROOT / output_type
    if not log_dir.exists():
        return []

    results = []
    for version_dir in sorted(log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1])):
        hparams_file = version_dir / "hparams.yaml"
        if not hparams_file.exists():
            continue

        try:
            with open(hparams_file) as f:
                hp = yaml.unsafe_load(f)
        except Exception as e:
            print(f"  WARN: failed to parse {hparams_file}: {e}")
            continue

        model_name = hp.get("model", "unknown")
        fold = hp.get("fold", -1)
        add_option = hp.get("add_option", None)

        # Find checkpoint
        ckpt_dir = version_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        ckpt_files = sorted(ckpt_dir.glob("epoch=*.ckpt"))
        if not ckpt_files:
            continue

        # Use last checkpoint if multiple
        ckpt_path = ckpt_files[-1]

        results.append({
            "ckpt_path": ckpt_path,
            "model": model_name,
            "fold": fold,
            "add_option": str(add_option),
            "version": version_dir.name,
        })

    return results


# ============================================================
# Main
# ============================================================
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    total_start = time.time()

    print("=" * 60)
    print("  NONZERO NORMALIZATION INFERENCE")
    print("  Change: NormalizeIntensity(nonzero=True)")
    print("  Output: {output_type}/pair{1,2}_nonzero/")
    print("=" * 60)

    all_results = []

    for output_type in OUTPUT_TYPES:
        print(f"\n{'#'*60}")
        print(f"  Output Type: {output_type}")
        print(f"{'#'*60}")

        checkpoints = discover_checkpoints(output_type)
        print(f"  Found {len(checkpoints)} versions")

        for ckpt_info in checkpoints:
            model_name = ckpt_info["model"]
            fold = ckpt_info["fold"]
            add_option = ckpt_info["add_option"]
            ckpt_path = ckpt_info["ckpt_path"]
            version = ckpt_info["version"]

            if add_option not in ["None", "none", "null"]:
                continue

            for pair in PAIRS:
                pair_path = DHCP_ROOT / pair
                if not pair_path.exists():
                    continue

                # ── KEY CHANGE: output to pair_nonzero dirs ──
                out_dir = OUTPUT_ROOT / output_type / f"{pair}{PAIR_SUFFIX}"
                out_dir.mkdir(parents=True, exist_ok=True)

                metrics_file = out_dir / f"{model_name}_fold{fold}_metrics.json"
                preds_file = out_dir / f"{model_name}_fold{fold}_predictions.csv"

                # Skip if already done
                if metrics_file.exists():
                    continue

                try:
                    result = run_single_inference(ckpt_path, pair_path, device)
                    if result is None:
                        print(f"  SKIP {model_name} fold{fold} {pair}: no valid subjects")
                        continue

                    metrics, predictions = result

                    # Save
                    with open(metrics_file, "w") as f:
                        json.dump(metrics, f, indent=2)
                    predictions.to_csv(preds_file, index=False)

                    print(f"  DONE {output_type}/{pair}{PAIR_SUFFIX} {model_name} fold{fold} | MAE={metrics['mae']:.3f} R2={metrics['r2']:.3f}")

                    all_results.append({
                        "output_type": output_type,
                        "pair": f"{pair}{PAIR_SUFFIX}",
                        "model": model_name,
                        "fold": fold,
                        "version": version,
                        **metrics,
                    })

                except Exception as e:
                    print(f"  ERROR {model_name} fold{fold} {pair}: {e}")

                finally:
                    torch.cuda.empty_cache()

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(OUTPUT_ROOT / "summary_nonzero.csv", index=False)

        # Print best per output_type per pair
        print(f"\n{'='*60}")
        print(f"  Summary: Best Model per Output Type (nonzero norm)")
        print(f"{'='*60}")
        for ot in OUTPUT_TYPES:
            for pair in PAIRS:
                pair_nz = f"{pair}{PAIR_SUFFIX}"
                subset = summary_df[(summary_df["output_type"] == ot) & (summary_df["pair"] == pair_nz)]
                if subset.empty:
                    continue
                # Average across folds per model
                avg = subset.groupby("model")["mae"].mean().sort_values()
                if len(avg) > 0:
                    best_model = avg.index[0]
                    best_mae = avg.iloc[0]
                    print(f"  {ot:12s} {pair_nz:20s} | Best: {best_model:<20s} MAE={best_mae:.3f}")

    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n  Total time: {h:02d}h {m:02d}m {s:02d}s")


if __name__ == "__main__":
    main()
