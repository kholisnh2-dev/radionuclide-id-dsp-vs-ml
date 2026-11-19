"""Formal comparison of DSP, Random Forest and CNN on the 27-isotope common scope.

- DSP at its own operating point (confidence gate), native and NaI(Tl)-like
  resolution (matching tolerance taken from the NaI(Tl)-like FWHM model);
- RF and CNN thresholds swept (0-1) to obtain coverage-accuracy curves and the
  accuracy at the DSP coverage;
- exact McNemar tests (DSP abstentions counted as wrong);
- Clopper-Pearson 95 % confidence intervals;
- model size and mean inference time per spectrum.

Input : data/processed/, data/augmented/rebinned_real.npz, models/,
        results/dsp/classification_results.csv
Output: results/comparison/*
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
CLS_DIR = ROOT / "results" / "dsp"
AUG_DIR = ROOT / "data" / "augmented"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "results" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION_A, RESOLUTION_B = 0.5651, 0.04100
NAI_FWHM_AT_662 = 0.07 * 661.66
MIN_MATCHED_PEAKS_FOR_CONFIDENT_ID = 2

NAI_FLOOR_KEV = 3.0
NAI_B = np.sqrt(max(NAI_FWHM_AT_662**2 - NAI_FLOOR_KEV**2, 0.0) / 661.66)


def degrade_resolution(energy_kev: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.clip(counts, 0, None)
    bin_width = energy_kev[1] - energy_kev[0]
    energy = np.clip(energy_kev, 1.0, None)
    target_fwhm = np.sqrt(NAI_FLOOR_KEV**2 + (NAI_B**2) * energy)
    sigma_bins = np.clip((target_fwhm / 2.3548) / bin_width, 0.3, None)
    out = np.zeros_like(counts, dtype=float)
    idx = np.arange(len(counts))
    for i in np.nonzero(counts > 0)[0]:
        s = sigma_bins[i]
        window = slice(max(0, i - int(6 * s)), min(len(counts), i + int(6 * s) + 1))
        weights = np.exp(-0.5 * ((idx[window] - i) / s) ** 2)
        weights /= weights.sum()
        out[window] += counts[i] * weights
    return out


def fwhm_kev_native(energy_kev):
    return np.sqrt(RESOLUTION_A**2 + (RESOLUTION_B**2) * energy_kev)


def fwhm_kev_nai(energy_kev):
    energy = np.clip(energy_kev, 1.0, None)
    return np.sqrt(NAI_FLOOR_KEV**2 + (NAI_B**2) * energy)


def detect_peaks(energy_kev, counts):
    counts_clipped = np.clip(counts, 0, None)
    kernel = np.ones(3) / 3
    smoothed = np.convolve(counts_clipped, kernel, mode="same")
    noise_floor = max(np.percentile(smoothed, 75), 1.0)
    min_prominence = max(5 * np.sqrt(noise_floor), 3.0)
    idx, props = find_peaks(smoothed, prominence=min_prominence, distance=3)
    return pd.DataFrame({"energy_kev": energy_kev[idx], "counts": counts[idx],
                          "prominence": props["prominences"]})


def score_candidates(detected_peaks, reference_peaks, fwhm_fn):
    rows = []
    for isotope, group in reference_peaks.groupby("isotope"):
        ref_energies = group["energy_kev"].values
        tol = fwhm_fn(ref_energies)
        matched = sum(
            1 for e_ref, t in zip(ref_energies, tol)
            if len(detected_peaks[np.abs(detected_peaks["energy_kev"] - e_ref) <= t]) > 0
        )
        n_ref = len(ref_energies)
        rows.append({"candidate_isotope": isotope, "n_reference_peaks": n_ref,
                      "n_matched": matched, "match_fraction": matched / n_ref if n_ref else 0.0})
    return pd.DataFrame(rows).sort_values(["match_fraction", "n_matched"], ascending=False).reset_index(drop=True)


def dsp_predict(energy_kev, counts, ref_peaks, fwhm_fn):
    detected = detect_peaks(energy_kev, counts)
    scores = score_candidates(detected, ref_peaks, fwhm_fn)
    top = scores.iloc[0] if len(scores) else None
    predicted_raw = top["candidate_isotope"] if top is not None and top["match_fraction"] > 0 else None
    raw_conf = float(top["match_fraction"]) if top is not None else 0.0
    gated = scores[(scores["n_matched"] >= MIN_MATCHED_PEAKS_FOR_CONFIDENT_ID) & (scores["match_fraction"] >= 0.5)]
    top_g = gated.iloc[0] if len(gated) else None
    predicted_confident = top_g["candidate_isotope"] if top_g is not None else None
    return predicted_raw, raw_conf, predicted_confident


# ---------------------------------------------------------------------------
def main():
    ref_peaks = pd.read_csv(PROC_DIR / "reference_peaks.csv")
    scope_isotopes = sorted(ref_peaks["isotope"].unique().tolist())  # the 26-isotope FAIR SCOPE
    print(f"Fair-comparison scope: {len(scope_isotopes)} isotopes with >=1 GSD reference peak")

    spectra = np.load(PROC_DIR / "spectra.npz")
    dsp_native_results = pd.read_csv(CLS_DIR / "classification_results.csv")
    dsp_native_results = dsp_native_results[dsp_native_results["true_isotope"].isin(scope_isotopes)].copy()

    # --- DSP degraded (recalibrated NaI tolerance) ---------------------------------
    print("Running DSP baseline on resolution-degraded (NaI-recalibrated) spectra...")
    dsp_degraded_rows = []
    dsp_timings_native, dsp_timings_degraded = [], []
    for isotope in scope_isotopes:
        energy = spectra[f"{isotope}__energy_kev"]
        counts = spectra[f"{isotope}__counts"]

        t0 = time.perf_counter()
        dsp_predict(energy, counts, ref_peaks, fwhm_kev_native)
        dsp_timings_native.append(time.perf_counter() - t0)

        degraded_counts = degrade_resolution(energy, counts)
        t0 = time.perf_counter()
        pred_raw_d, conf_d, pred_gated_d = dsp_predict(energy, degraded_counts, ref_peaks, fwhm_kev_nai)
        dsp_timings_degraded.append(time.perf_counter() - t0)

        dsp_degraded_rows.append({
            "isotope": isotope,
            "dsp_degraded_pred_raw": pred_raw_d,
            "dsp_degraded_raw_confidence": conf_d,
            "dsp_degraded_pred_gated": pred_gated_d,
            "dsp_degraded_correct_gated": pred_gated_d == isotope,
        })
    dsp_degraded_df = pd.DataFrame(dsp_degraded_rows)

    meta_all = pd.read_csv(PROC_DIR / "metadata.csv")
    uncovered_isotopes = sorted(set(meta_all["isotope"]) - set(scope_isotopes))
    false_accept_native = 0
    false_accept_degraded = 0
    for isotope in uncovered_isotopes:
        energy = spectra[f"{isotope}__energy_kev"]
        counts = spectra[f"{isotope}__counts"]
        _, _, pred_native = dsp_predict(energy, counts, ref_peaks, fwhm_kev_native)
        degraded_counts = degrade_resolution(energy, counts)
        _, _, pred_degraded = dsp_predict(energy, degraded_counts, ref_peaks, fwhm_kev_nai)
        false_accept_native += pred_native is not None
        false_accept_degraded += pred_degraded is not None
    print(f"False-accept on {len(uncovered_isotopes)} isotopes with NO reference data: "
          f"native={false_accept_native}/{len(uncovered_isotopes)}, "
          f"degraded(NaI-recalibrated)={false_accept_degraded}/{len(uncovered_isotopes)}")

    # --- RF / CNN: recompute full predict_proba (not just true-class proba) -------
    print("Loading trained models to recompute full confidence vectors...")
    rf = joblib.load(MODEL_DIR / "rf_model.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    import tensorflow as tf
    cnn = tf.keras.models.load_model(MODEL_DIR / "cnn_model.keras")

    real = np.load(AUG_DIR / "rebinned_real.npz")
    grid_centers = real["grid_centers"] if "grid_centers" in real.files else None
    if grid_centers is None:
        aug = np.load(AUG_DIR / "augmented_train.npz", allow_pickle=True)
        grid_centers = aug["grid_centers"]

    TOTAL_COUNT_NORM = 1e5

    def featurize(X):
        X = np.clip(X, 0, None)
        sums = X.sum(axis=-1, keepdims=True)
        sums = np.where(sums == 0, 1, sums)
        return np.log1p(X / sums * TOTAL_COUNT_NORM)

    def degrade_rebinned(binned_counts):
        edges = np.linspace(0, grid_centers[-1] + (grid_centers[1] - grid_centers[0]) / 2, len(grid_centers) + 1)
        bin_width = edges[1] - edges[0]
        energy = np.clip(grid_centers, 1.0, None)
        target_fwhm = np.sqrt(NAI_FLOOR_KEV**2 + (NAI_B**2) * energy)
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

    ml_rows = []
    rf_timings, cnn_timings = [], []
    for isotope in scope_isotopes:
        x_native = real[isotope]
        x_degraded = degrade_rebinned(np.clip(x_native, 0, None))
        x_native_feat = featurize(x_native[None, :])
        x_degraded_feat = featurize(x_degraded[None, :])

        t0 = time.perf_counter()
        rf_proba_native = rf.predict_proba(x_native_feat)[0]
        rf_timings.append(time.perf_counter() - t0)
        rf_proba_degraded = rf.predict_proba(x_degraded_feat)[0]
        rf_pred_native = le.inverse_transform([np.argmax(rf_proba_native)])[0]
        rf_pred_degraded = le.inverse_transform([np.argmax(rf_proba_degraded)])[0]

        t0 = time.perf_counter()
        cnn_proba_native = cnn.predict(x_native_feat[..., None], verbose=0)[0]
        cnn_timings.append(time.perf_counter() - t0)
        cnn_proba_degraded = cnn.predict(x_degraded_feat[..., None], verbose=0)[0]
        cnn_pred_native = le.inverse_transform([np.argmax(cnn_proba_native)])[0]
        cnn_pred_degraded = le.inverse_transform([np.argmax(cnn_proba_degraded)])[0]

        ml_rows.append({
            "isotope": isotope,
            "rf_pred_native": rf_pred_native, "rf_conf_native": float(rf_proba_native.max()),
            "rf_correct_native": rf_pred_native == isotope,
            "rf_pred_degraded": rf_pred_degraded, "rf_conf_degraded": float(rf_proba_degraded.max()),
            "rf_correct_degraded": rf_pred_degraded == isotope,
            "cnn_pred_native": cnn_pred_native, "cnn_conf_native": float(cnn_proba_native.max()),
            "cnn_correct_native": cnn_pred_native == isotope,
            "cnn_pred_degraded": cnn_pred_degraded, "cnn_conf_degraded": float(cnn_proba_degraded.max()),
            "cnn_correct_degraded": cnn_pred_degraded == isotope,
        })
    ml_df = pd.DataFrame(ml_rows)

    # ---------------------------------------------------------------------
    # Merge everything on the common 26-isotope scope
    # ---------------------------------------------------------------------
    merged = dsp_native_results[[
        "true_isotope", "predicted_raw", "top_match_fraction_raw", "correct_raw",
        "predicted_confident", "correct_confident",
    ]].rename(columns={"true_isotope": "isotope"})
    merged = merged.merge(dsp_degraded_df, on="isotope").merge(ml_df, on="isotope")
    merged.to_csv(OUT_DIR / "fair_scope_comparison.csv", index=False)

    thresholds = np.linspace(0.0, 1.0, 101)

    def coverage_accuracy_curve(conf_col, correct_col, df):
        cov, acc = [], []
        for t in thresholds:
            mask = df[conf_col] >= t
            cov.append(mask.mean())
            acc.append(df.loc[mask, correct_col].mean() if mask.sum() else np.nan)
        return np.array(cov), np.array(acc)

    curves = {}
    curves["DSP (native, raw score, n>=2 fixed)"] = coverage_accuracy_curve("top_match_fraction_raw", "correct_raw", merged)
    curves["RF (native)"] = coverage_accuracy_curve("rf_conf_native", "rf_correct_native", merged)
    curves["RF (degraded NaI)"] = coverage_accuracy_curve("rf_conf_degraded", "rf_correct_degraded", merged)
    curves["CNN (native)"] = coverage_accuracy_curve("cnn_conf_native", "cnn_correct_native", merged)
    curves["CNN (degraded NaI)"] = coverage_accuracy_curve("cnn_conf_degraded", "cnn_correct_degraded", merged)

    # DSP's REAL operating point (2-part gate, not a swept threshold)
    dsp_gate_coverage = merged["predicted_confident"].notna().mean()
    dsp_gate_accuracy = merged.loc[merged["predicted_confident"].notna(), "correct_confident"].mean()
    print(f"\nDSP's actual gated operating point: coverage={dsp_gate_coverage:.1%}, "
          f"accuracy-when-confident={dsp_gate_accuracy:.1%}")

    # Coverage-matched comparison: find RF/CNN threshold whose coverage is
    # closest to DSP's realized coverage, report accuracy there.
    def accuracy_at_matched_coverage(conf_col, correct_col, target_coverage):
        best_t, best_cov_diff, best_acc, best_cov = None, np.inf, None, None
        for t in thresholds:
            mask = merged[conf_col] >= t
            cov = mask.mean()
            if mask.sum() == 0:
                continue
            diff = abs(cov - target_coverage)
            if diff < best_cov_diff:
                best_cov_diff, best_t, best_cov = diff, t, cov
                best_acc = merged.loc[mask, correct_col].mean()
        return best_t, best_cov, best_acc

    matched = {}
    for label, conf_col, correct_col in [
        ("RF (native)", "rf_conf_native", "rf_correct_native"),
        ("RF (degraded NaI)", "rf_conf_degraded", "rf_correct_degraded"),
        ("CNN (native)", "cnn_conf_native", "cnn_correct_native"),
        ("CNN (degraded NaI)", "cnn_conf_degraded", "cnn_correct_degraded"),
    ]:
        t, cov, acc = accuracy_at_matched_coverage(conf_col, correct_col, dsp_gate_coverage)
        matched[label] = {"threshold": t, "coverage": cov, "accuracy": acc}
        print(f"  Coverage-matched ({label}): threshold={t:.2f} -> coverage={cov:.1%}, accuracy={acc:.1%}")

    robustness = {
        "dsp_native_accuracy_gated": float(dsp_gate_accuracy),
        "dsp_native_coverage_gated": float(dsp_gate_coverage),
        "dsp_degraded_accuracy_gated": float(merged.loc[merged["dsp_degraded_pred_gated"].notna(), "dsp_degraded_correct_gated"].mean()) if merged["dsp_degraded_pred_gated"].notna().any() else float("nan"),
        "dsp_degraded_coverage_gated": float(merged["dsp_degraded_pred_gated"].notna().mean()),
        "rf_native_accuracy_fullscope": float(merged["rf_correct_native"].mean()),
        "rf_degraded_accuracy_fullscope": float(merged["rf_correct_degraded"].mean()),
        "cnn_native_accuracy_fullscope": float(merged["cnn_correct_native"].mean()),
        "cnn_degraded_accuracy_fullscope": float(merged["cnn_correct_degraded"].mean()),
        "dsp_false_accept_native": f"{false_accept_native}/{len(uncovered_isotopes)}",
        "dsp_false_accept_degraded_nai_recalibrated": f"{false_accept_degraded}/{len(uncovered_isotopes)}",
    }

    # ---------------------------------------------------------------------
    # Paired significance tests (McNemar's exact test via sign test on
    # discordant pairs) on the full common scope (len(scope_isotopes)
    # isotopes with >=1 reference peak), native resolution.
    # DSP abstentions are treated as INCORRECT for this paired test (the
    # conservative choice - an abstain is not a correct ID) so every
    # isotope has a defined correct/incorrect outcome for every method.
    # ---------------------------------------------------------------------
    def mcnemar(correct_a, correct_b, label_a, label_b):
        a = correct_a.astype(object).where(correct_a.notna(), False).astype(bool).to_numpy()
        b = correct_b.astype(object).where(correct_b.notna(), False).astype(bool).to_numpy()
        b_wins = int(np.sum((~a) & b))   # a wrong, b right
        a_wins = int(np.sum(a & (~b)))   # a right, b wrong
        n_discordant = a_wins + b_wins
        if n_discordant == 0:
            p = 1.0
        else:
            p = binomtest(min(a_wins, b_wins), n_discordant, 0.5).pvalue
        return {"method_a": label_a, "method_b": label_b, "a_right_b_wrong": a_wins,
                "b_right_a_wrong": b_wins, "n_discordant": n_discordant, "p_value": p}

    dsp_correct_conservative = merged["correct_confident"].where(merged["predicted_confident"].notna(), False)
    tests = [
        mcnemar(dsp_correct_conservative, merged["rf_correct_native"], "DSP (gated, abstain=wrong)", "RF (native)"),
        mcnemar(dsp_correct_conservative, merged["cnn_correct_native"], "DSP (gated, abstain=wrong)", "CNN (native)"),
        mcnemar(merged["rf_correct_native"], merged["cnn_correct_native"], "RF (native)", "CNN (native)"),
        mcnemar(merged["rf_correct_native"], merged["rf_correct_degraded"], "RF (native)", "RF (degraded NaI)"),
        mcnemar(merged["cnn_correct_native"], merged["cnn_correct_degraded"], "CNN (native)", "CNN (degraded NaI)"),
    ]
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT_DIR / "mcnemar_tests.csv", index=False)

    # ---------------------------------------------------------------------
    # 95% exact (Clopper-Pearson) confidence intervals on every headline
    # accuracy number - with n=24-26, a "100%" point estimate alone
    # overstates precision; the CI makes the real uncertainty explicit
    # (matches this project's rule to flag results that look too clean).
    # ---------------------------------------------------------------------
    def ci(successes, n):
        if n == 0:
            return None
        lo, hi = binomtest(successes, n, 0.5).proportion_ci(confidence_level=0.95, method="exact")
        return {"successes": int(successes), "n": int(n), "point": round(successes / n, 4),
                "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)}

    n_covered = int(merged["predicted_confident"].notna().sum())
    n_covered_correct = int(merged.loc[merged["predicted_confident"].notna(), "correct_confident"].sum())
    confidence_intervals = {
        "dsp_native_gated": ci(n_covered_correct, n_covered),
        "rf_native_fullscope": ci(int(merged["rf_correct_native"].sum()), len(merged)),
        "rf_degraded_fullscope": ci(int(merged["rf_correct_degraded"].sum()), len(merged)),
        "cnn_native_fullscope": ci(int(merged["cnn_correct_native"].sum()), len(merged)),
        "cnn_degraded_fullscope": ci(int(merged["cnn_correct_degraded"].sum()), len(merged)),
    }
    for k, v in confidence_intervals.items():
        print(f"  95% CI ({k}): {v['point']:.1%}  [{v['ci95_low']:.1%}, {v['ci95_high']:.1%}]  (n={v['n']})")

    # ---------------------------------------------------------------------
    # Computational cost
    # ---------------------------------------------------------------------
    rf_size_mb = (MODEL_DIR / "rf_model.joblib").stat().st_size / 1e6
    cnn_size_mb = (MODEL_DIR / "cnn_model.keras").stat().st_size / 1e6
    dsp_asset_size_kb = ((PROC_DIR / "reference_peaks.csv").stat().st_size) / 1e3

    dsp_ref_row_count = len(ref_peaks)
    cost = {
        "dsp_model_size_mb": round(dsp_asset_size_kb / 1000, 5),
        "dsp_asset_note": f"No trained model file - just the {dsp_ref_row_count}-row reference peak table + a 2-parameter resolution formula.",
        "rf_model_size_mb": round(rf_size_mb, 2),
        "cnn_model_size_mb": round(cnn_size_mb, 2),
        "dsp_inference_ms_native_mean": round(1000 * float(np.mean(dsp_timings_native)), 3),
        "dsp_inference_ms_degraded_mean": round(1000 * float(np.mean(dsp_timings_degraded)), 3),
        "rf_inference_ms_mean": round(1000 * float(np.mean(rf_timings)), 3),
        "cnn_inference_ms_mean": round(1000 * float(np.mean(cnn_timings)), 3),
        "dsp_input_size": "8192-channel full-resolution spectrum",
        "ml_input_size": "512-bin rebinned spectrum",
        "note": "DSP and ML operate on different-resolution inputs (see input_size fields) - "
                "timings are NOT purely an algorithm-vs-algorithm comparison, they also reflect "
                "that difference. Flagged, not hidden.",
    }

    with open(OUT_DIR / "comparison_summary.json", "w") as f:
        json.dump({
            "scope_n_isotopes": len(scope_isotopes),
            "dsp_gate_operating_point": {"coverage": float(dsp_gate_coverage), "accuracy": float(dsp_gate_accuracy)},
            "coverage_matched_comparison": matched,
            "robustness": robustness,
            "confidence_intervals_95pct": confidence_intervals,
            "computational_cost": cost,
        }, f, indent=2, default=str)

    # Save curves for plotting
    np.savez(OUT_DIR / "coverage_accuracy_curves.npz",
              thresholds=thresholds,
              **{f"{k}__cov".replace(" ", "_"): v[0] for k, v in curves.items()},
              **{f"{k}__acc".replace(" ", "_"): v[1] for k, v in curves.items()})

    print("\nSaved: fair_scope_comparison.csv, mcnemar_tests.csv, comparison_summary.json, coverage_accuracy_curves.npz")
    return merged, matched, robustness, cost, tests_df


if __name__ == "__main__":
    main()
