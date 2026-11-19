"""Control tests for the ablation test (run after ablation_test.py).

The ablation test blanks the 256 bins that lie within two grid cells of the
179 GSD reference lines. Because this removes half of the 512 input bins, two
controls are run with the same trained models (no retraining):

  A. peak regions blanked      : the original ablation (256 bins around the lines);
  B. only peak regions kept    : the other 256 bins (outside the peak regions) blanked;
  C. random bins blanked       : 256 bins chosen at random, repeated with 20 seeds (0-19).

If the models rely on peak information, accuracy should collapse in A, stay
high in B, and stay well above A in C.

Input : data/augmented/rebinned_real.npz, data/processed/reference_peaks_gsd.csv, models/
Output: results/ml/ablation_control.csv, results/ml/ablation_control.json
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
N_RANDOM = 20


def featurize(X):
    X = np.clip(X, 0, None)
    sums = X.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    return np.log1p(X / sums * TOTAL_COUNT_NORM)


def peak_bins(grid_centers, energies):
    width = grid_centers[1] - grid_centers[0]
    bins = set()
    for e in energies:
        c = int(round((e - grid_centers[0]) / width))
        bins.update(i for i in range(c - 2, c + 3) if 0 <= i < len(grid_centers))
    return np.array(sorted(bins))


def main():
    real = np.load(AUG_DIR / "rebinned_real.npz")
    grid = real["grid_centers"]
    isotopes = sorted(k for k in real.files if k != "grid_centers")
    X = np.stack([real[k].astype(float) for k in isotopes])
    ref = pd.read_csv(PROC_DIR / "reference_peaks_gsd.csv")
    peaks = peak_bins(grid, ref["energy_kev"].values)
    others = np.setdiff1d(np.arange(len(grid)), peaks)

    rf = joblib.load(MODEL_DIR / "rf_model.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    import tensorflow as tf
    cnn = tf.keras.models.load_model(MODEL_DIR / "cnn_model.keras")
    y = np.array(isotopes)

    def accuracy(mask):
        Xm = X.copy()
        Xm[:, mask] = 0.0
        F = featurize(Xm)
        rf_pred = le.inverse_transform(rf.predict(F))
        cnn_pred = le.inverse_transform(np.argmax(cnn.predict(F[..., None], verbose=0), axis=1))
        return float(np.mean(rf_pred == y)), float(np.mean(cnn_pred == y))

    rows = []
    rf_a, cnn_a = accuracy(peaks)
    rows.append({"test": "A_peak_regions_blanked", "n_bins_blanked": len(peaks), "seed": "", "rf": rf_a, "cnn": cnn_a})
    rf_b, cnn_b = accuracy(others)
    rows.append({"test": "B_only_peak_regions_kept", "n_bins_blanked": len(others), "seed": "", "rf": rf_b, "cnn": cnn_b})
    for s in range(N_RANDOM):
        mask = np.random.default_rng(s).choice(len(grid), size=len(peaks), replace=False)
        rf_c, cnn_c = accuracy(mask)
        rows.append({"test": "C_random_bins_blanked", "n_bins_blanked": len(peaks), "seed": s, "rf": rf_c, "cnn": cnn_c})
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "ablation_control.csv", index=False)
    rnd = df[df.test == "C_random_bins_blanked"]
    summary = {
        "n_bins_total": int(len(grid)), "n_peak_bins": int(len(peaks)), "n_random_seeds": N_RANDOM,
        "A_peak_regions_blanked": {"rf": rf_a, "cnn": cnn_a},
        "B_only_peak_regions_kept": {"rf": rf_b, "cnn": cnn_b},
        "C_random_bins_blanked": {"rf_mean": float(rnd.rf.mean()), "rf_std": float(rnd.rf.std(ddof=1)),
                                  "rf_min": float(rnd.rf.min()), "rf_max": float(rnd.rf.max()),
                                  "cnn_mean": float(rnd.cnn.mean()), "cnn_std": float(rnd.cnn.std(ddof=1)),
                                  "cnn_min": float(rnd.cnn.min()), "cnn_max": float(rnd.cnn.max())},
    }
    (RES_DIR / "ablation_control.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
