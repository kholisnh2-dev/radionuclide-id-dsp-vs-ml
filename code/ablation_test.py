"""Ablation test: blank the bins around reference lines and re-test the models.

All bins within two grid cells of any of the 179 reference lines annotated in
the GSD pages (256 of 512 bins) are set to zero in each original test spectrum;
the trained models are applied again without retraining.

Input : data/augmented/rebinned_real.npz, data/processed/reference_peaks_gsd.csv, models/
Output: results/ml/ablation_results.csv, results/ml/ablation_summary.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUG_DIR = ROOT / "data" / "augmented"
MODEL_DIR = ROOT / "models"
RES_DIR = ROOT / "results" / "ml"
PROC_DIR = ROOT / "data" / "processed"

TOTAL_COUNT_NORM = 1e5


def featurize(X):
    X = np.clip(X, 0, None)
    sums = X.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    return np.log1p(X / sums * TOTAL_COUNT_NORM)


def main():
    real = np.load(AUG_DIR / "rebinned_real.npz")
    grid_centers = real["grid_centers"]
    ref_peaks = pd.read_csv(PROC_DIR / "reference_peaks_gsd.csv")
    isotopes = sorted([k for k in real.files if k != "grid_centers"])

    bin_width = grid_centers[1] - grid_centers[0]
    mask_bins = set()
    for e in ref_peaks["energy_kev"].values:
        center_idx = int(round((e - grid_centers[0]) / bin_width))
        for off in range(-2, 3):
            idx = center_idx + off
            if 0 <= idx < len(grid_centers):
                mask_bins.add(idx)
    mask_bins = sorted(mask_bins)
    print(f"Blanking {len(mask_bins)}/{len(grid_centers)} bins "
          f"(within +/-2 grid-cells of any of the {len(ref_peaks)} known GSD reference peaks)")

    rf = joblib.load(MODEL_DIR / "rf_model.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    import tensorflow as tf
    cnn = tf.keras.models.load_model(MODEL_DIR / "cnn_model.keras")

    rf_correct, cnn_correct = 0, 0
    rows = []
    for isotope in isotopes:
        x = real[isotope].astype(float).copy()
        x[mask_bins] = 0.0
        x_feat = featurize(x[None, :])

        rf_pred = le.inverse_transform(rf.predict(x_feat))[0]
        cnn_proba = cnn.predict(x_feat[..., None], verbose=0)[0]
        cnn_pred = le.inverse_transform([np.argmax(cnn_proba)])[0]

        rf_ok = rf_pred == isotope
        cnn_ok = cnn_pred == isotope
        rf_correct += rf_ok
        cnn_correct += cnn_ok
        rows.append({"isotope": isotope, "rf_pred_ablated": rf_pred, "rf_correct_ablated": rf_ok,
                      "cnn_pred_ablated": cnn_pred, "cnn_correct_ablated": cnn_ok})

    n = len(isotopes)
    rf_acc = rf_correct / n
    cnn_acc = cnn_correct / n
    print(f"Random Forest, known-peak bins blanked: {rf_acc:.1%} ({rf_correct}/{n})")
    print(f"CNN, known-peak bins blanked:            {cnn_acc:.1%} ({cnn_correct}/{n})")

    pd.DataFrame(rows).to_csv(RES_DIR / "ablation_results.csv", index=False)
    with open(RES_DIR / "ablation_summary.json", "w") as f:
        json.dump({"n_bins_masked": len(mask_bins), "n_bins_total": int(len(grid_centers)),
                    "rf_ablated_accuracy": rf_acc, "cnn_ablated_accuracy": cnn_acc}, f, indent=2)
    return rf_acc, cnn_acc


if __name__ == "__main__":
    main()
