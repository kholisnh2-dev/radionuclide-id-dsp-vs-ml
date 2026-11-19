"""Repeat augmentation and training with five random seeds (20260814, 1, 2, 3, 4).

Reports the mean, standard deviation and range of the held-out accuracy of the
Random Forest and the CNN at native and NaI(Tl)-like resolution.

Input : data/processed/spectra.npz, metadata.csv
Output: results/ml/multiseed_results.csv, results/ml/multiseed_results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augmentation as aug_mod  # noqa: E402
import train_models as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# First seed = the original/canonical run already reported everywhere else.
# The other 4 are arbitrary fresh seeds, not selected after seeing results.
SEEDS = [20260814, 1, 2, 3, 4]


def run_one_seed(seed: int, n_variants_per_isotope: int = 300) -> dict:
    t0 = time.time()

    # Monkey-patch the augmentation module's RNG so augment_one() draws from
    # this seed instead of its hardcoded default - reuses its exact logic.
    aug_mod.RNG = np.random.default_rng(seed)

    npz = np.load(DATA_DIR / "spectra.npz")
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    isotopes = sorted(meta["isotope"].tolist())

    real_rebinned = {}
    aug_X, aug_y = [], []
    for isotope in isotopes:
        energy = npz[f"{isotope}__energy_kev"]
        counts = npz[f"{isotope}__counts"]
        bg_key = f"{isotope}__background_counts"
        background = npz[bg_key] if bg_key in npz.files else None
        real_rebinned[isotope] = aug_mod.rebin(energy, counts)
        variants = aug_mod.augment_one(energy, counts, background, n_variants_per_isotope)
        aug_X.append(variants)
        aug_y.extend([isotope] * len(variants))
    aug_X = np.concatenate(aug_X, axis=0)
    aug_y = np.array(aug_y)
    grid_centers = aug_mod.GRID_CENTERS

    le = LabelEncoder().fit(isotopes)
    X_aug = train_mod.featurize(aug_X)
    y_aug_enc = le.transform(aug_y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug, y_aug_enc, test_size=0.2, random_state=seed, stratify=y_aug_enc
    )

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
    rf.fit(X_tr, y_tr)

    import tensorflow as tf
    tf.random.set_seed(seed)
    cnn = train_mod.build_cnn(n_classes=len(isotopes), n_bins=X_tr.shape[1])
    cnn.fit(X_tr[..., None], y_tr, validation_data=(X_val[..., None], y_val),
            epochs=30, batch_size=64, verbose=0)

    rows = []
    for isotope in isotopes:
        x_real = real_rebinned[isotope]
        x_real_feat = train_mod.featurize(x_real[None, :])
        x_degraded = train_mod.degrade_resolution_for_eval(
            np.clip(x_real, 0, None), 1.0, grid_centers)
        x_degraded_feat = train_mod.featurize(x_degraded[None, :])

        rf_pred_native = le.inverse_transform(rf.predict(x_real_feat))[0]
        rf_pred_degraded = le.inverse_transform(rf.predict(x_degraded_feat))[0]
        cnn_proba_native = cnn.predict(x_real_feat[..., None], verbose=0)[0]
        cnn_pred_native = le.inverse_transform([np.argmax(cnn_proba_native)])[0]
        cnn_proba_degraded = cnn.predict(x_degraded_feat[..., None], verbose=0)[0]
        cnn_pred_degraded = le.inverse_transform([np.argmax(cnn_proba_degraded)])[0]

        rows.append({
            "isotope": isotope,
            "rf_correct_native": rf_pred_native == isotope,
            "rf_correct_degraded": rf_pred_degraded == isotope,
            "cnn_correct_native": cnn_pred_native == isotope,
            "cnn_correct_degraded": cnn_pred_degraded == isotope,
        })
    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    result = {
        "seed": seed,
        "rf_native": float(df["rf_correct_native"].mean()),
        "rf_degraded": float(df["rf_correct_degraded"].mean()),
        "cnn_native": float(df["cnn_correct_native"].mean()),
        "cnn_degraded": float(df["cnn_correct_degraded"].mean()),
        "rf_misses_native": df.loc[~df["rf_correct_native"], "isotope"].tolist(),
        "cnn_misses_native": df.loc[~df["cnn_correct_native"], "isotope"].tolist(),
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"seed {seed} done in {elapsed:.1f}s: "
          f"RF native={result['rf_native']:.1%} degraded={result['rf_degraded']:.1%}  "
          f"CNN native={result['cnn_native']:.1%} degraded={result['cnn_degraded']:.1%}")
    return result


def main():
    results = []
    for seed in SEEDS:
        results.append(run_one_seed(seed))

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "multiseed_results.csv", index=False)

    summary = {"seeds": SEEDS, "n_seeds": len(SEEDS)}
    for col in ["rf_native", "rf_degraded", "cnn_native", "cnn_degraded"]:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "values": df[col].tolist(),
        }

    with open(OUT_DIR / "multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"MULTI-SEED SUMMARY (mean +/- std over {len(SEEDS)} seeds: {SEEDS})")
    print("=" * 70)
    for col in ["rf_native", "rf_degraded", "cnn_native", "cnn_degraded"]:
        s = summary[col]
        print(f"{col}: {s['mean']:.1%} +/- {s['std']:.1%}  (range {s['min']:.1%}-{s['max']:.1%})")

    return df, summary


if __name__ == "__main__":
    main()
