"""Train the Random Forest and the 1D CNN and test them on the original spectra.

Input features: spectrum normalised to 1e5 total counts, then x = ln(1 + counts).
Split of the augmented set: 80 % training (8640) / 20 % validation (2160), stratified.
Random Forest: 300 trees (scikit-learn).
CNN: Conv1D(16, 9) - MaxPool(4) - Conv1D(32, 7) - MaxPool(4) - Conv1D(64, 5) -
     GlobalAveragePooling - Dropout(0.3) - Dense(64) - Dropout(0.2) - Dense(36, softmax);
     Adam (learning rate 1e-3), 30 epochs, batch size 64.
Test: the 36 original spectra (never augmented), at native resolution and after
broadening to the NaI(Tl)-like resolution.

Input : data/augmented/*.npz
Output: models/rf_model.joblib, models/cnn_model.keras, models/label_encoder.joblib
        results/ml/heldout_original_results.csv, results/ml/ml_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
AUG_DIR = ROOT / "data" / "augmented"
MODEL_DIR = ROOT / "models"
RES_DIR = ROOT / "results" / "ml"
for _d in (MODEL_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TOTAL_COUNT_NORM = 1e5


def featurize(X: np.ndarray) -> np.ndarray:
    """Normalize each spectrum to a fixed total count (removes acquisition- time/dose confound from absolute scale) then log1p-compress (peaks span orders of magnitude more counts than the continuum)."""
    X = np.clip(X, 0, None)
    sums = X.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    normalized = X / sums * TOTAL_COUNT_NORM
    return np.log1p(normalized)


def build_cnn(n_classes: int, n_bins: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(n_bins, 1)),
        layers.Conv1D(16, 9, padding="same", activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(32, 7, padding="same", activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def degrade_resolution_for_eval(binned_counts, fwhm_fraction, grid_centers):
    NAI_FWHM_AT_662 = 0.07 * 661.66
    NAI_FLOOR_KEV = 3.0
    NAI_B = np.sqrt(max(NAI_FWHM_AT_662**2 - NAI_FLOOR_KEV**2, 0.0) / 661.66)
    edges = np.linspace(0, grid_centers[-1] + (grid_centers[1] - grid_centers[0]) / 2,
                         len(grid_centers) + 1)
    bin_width = edges[1] - edges[0]
    energy = np.clip(grid_centers, 1.0, None)
    target_fwhm = np.sqrt(NAI_FLOOR_KEV**2 + (NAI_B**2) * energy) * fwhm_fraction
    sigma_bins = np.clip((target_fwhm / 2.3548) / bin_width, 0.3, None)
    out = np.zeros_like(binned_counts, dtype=float)
    idx = np.arange(len(binned_counts))
    for i in np.nonzero(binned_counts > 0)[0]:
        s = sigma_bins[i]
        window = slice(max(0, i - int(6 * s)), min(len(binned_counts), i + int(6 * s) + 1))
        weights = np.exp(-0.5 * ((idx[window] - i) / s) ** 2)
        weights /= weights.sum()
        out[window] += binned_counts[i] * weights
    return out


def main():
    aug = np.load(AUG_DIR / "augmented_train.npz", allow_pickle=True)
    real = np.load(AUG_DIR / "rebinned_real.npz")
    grid_centers = aug["grid_centers"]

    X_aug_raw, y_aug = aug["X"], aug["y"]
    isotopes = sorted(set(y_aug.tolist()))
    le = LabelEncoder().fit(isotopes)

    X_aug = featurize(X_aug_raw)
    y_aug_enc = le.transform(y_aug)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug, y_aug_enc, test_size=0.2, random_state=0, stratify=y_aug_enc
    )

    print(f"Training set: {X_tr.shape}, validation (in-distribution) set: {X_val.shape}")
    print(f"Classes: {len(isotopes)}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                 n_jobs=-1, random_state=0)
    rf.fit(X_tr, y_tr)
    rf_indist_acc = rf.score(X_val, y_val)
    print(f"\n[Random Forest] in-distribution val accuracy: {rf_indist_acc:.1%}")

    # --- CNN ---
    cnn = build_cnn(n_classes=len(isotopes), n_bins=X_tr.shape[1])
    history = cnn.fit(
        X_tr[..., None], y_tr,
        validation_data=(X_val[..., None], y_val),
        epochs=30, batch_size=64, verbose=0,
    )
    cnn_indist_acc = history.history["val_accuracy"][-1]
    print(f"[CNN] in-distribution val accuracy (final epoch): {cnn_indist_acc:.1%}")
    print(f"[CNN] training curve (val_accuracy every 5 epochs): "
          f"{[round(v, 3) for v in history.history['val_accuracy'][::5]]}")

    # --- Held-out-original test (PRIMARY metric) ---
    results = []
    for isotope in isotopes:
        x_real = real[isotope]
        x_real_feat = featurize(x_real[None, :])
        x_degraded = degrade_resolution_for_eval(np.clip(x_real, 0, None), 1.0, grid_centers)
        x_degraded_feat = featurize(x_degraded[None, :])

        true_idx = le.transform([isotope])[0]

        rf_pred_native = le.inverse_transform(rf.predict(x_real_feat))[0]
        rf_pred_degraded = le.inverse_transform(rf.predict(x_degraded_feat))[0]
        rf_proba_native = rf.predict_proba(x_real_feat)[0][true_idx]

        cnn_proba_native = cnn.predict(x_real_feat[..., None], verbose=0)[0]
        cnn_pred_native = le.inverse_transform([np.argmax(cnn_proba_native)])[0]
        cnn_proba_degraded = cnn.predict(x_degraded_feat[..., None], verbose=0)[0]
        cnn_pred_degraded = le.inverse_transform([np.argmax(cnn_proba_degraded)])[0]

        results.append({
            "isotope": isotope,
            "rf_pred_native_hpge": rf_pred_native,
            "rf_correct_native": rf_pred_native == isotope,
            "rf_pred_degraded_nai": rf_pred_degraded,
            "rf_correct_degraded": rf_pred_degraded == isotope,
            "rf_true_class_proba_native": float(rf_proba_native),
            "cnn_pred_native_hpge": cnn_pred_native,
            "cnn_correct_native": cnn_pred_native == isotope,
            "cnn_pred_degraded_nai": cnn_pred_degraded,
            "cnn_correct_degraded": cnn_pred_degraded == isotope,
            "cnn_true_class_proba_native": float(cnn_proba_native[true_idx]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RES_DIR / "heldout_original_results.csv", index=False)

    summary = {
        "n_classes": len(isotopes),
        "n_augmented_train": int(len(X_tr)),
        "n_augmented_indist_val": int(len(X_val)),
        "rf_indist_val_accuracy": rf_indist_acc,
        "cnn_indist_val_accuracy": float(cnn_indist_acc),
        "rf_heldout_original_accuracy_native_hpge": float(results_df["rf_correct_native"].mean()),
        "rf_heldout_original_accuracy_degraded_nai": float(results_df["rf_correct_degraded"].mean()),
        "cnn_heldout_original_accuracy_native_hpge": float(results_df["cnn_correct_native"].mean()),
        "cnn_heldout_original_accuracy_degraded_nai": float(results_df["cnn_correct_degraded"].mean()),
    }
    with open(RES_DIR / "ml_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("HELD-OUT-ORIGINAL TEST (primary metric - each isotope's true, "
          "never-augmented real spectrum)")
    print("=" * 70)
    print(f"Random Forest: native(HPGe)={summary['rf_heldout_original_accuracy_native_hpge']:.1%}  "
          f"degraded(NaI-like)={summary['rf_heldout_original_accuracy_degraded_nai']:.1%}")
    print(f"CNN:           native(HPGe)={summary['cnn_heldout_original_accuracy_native_hpge']:.1%}  "
          f"degraded(NaI-like)={summary['cnn_heldout_original_accuracy_degraded_nai']:.1%}")

    misses_rf = results_df[~results_df["rf_correct_native"]]
    misses_cnn = results_df[~results_df["cnn_correct_native"]]
    print(f"\nRF misses on native HPGe test ({len(misses_rf)}): "
          f"{list(zip(misses_rf['isotope'], misses_rf['rf_pred_native_hpge']))}")
    print(f"CNN misses on native HPGe test ({len(misses_cnn)}): "
          f"{list(zip(misses_cnn['isotope'], misses_cnn['cnn_pred_native_hpge']))}")

    cnn.save(MODEL_DIR / "cnn_model.keras")
    import joblib
    joblib.dump(rf, MODEL_DIR / "rf_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    return results_df, summary


if __name__ == "__main__":
    main()
