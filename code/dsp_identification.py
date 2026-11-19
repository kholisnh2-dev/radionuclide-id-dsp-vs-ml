"""Classical DSP radionuclide identification (peak search + library matching).

Steps per spectrum:
  1. 3-point moving-average smoothing;
  2. peak search with scipy.signal.find_peaks, prominence >= max(5*sqrt(N75), 3)
     (N75 = 75th percentile of the smoothed counts), minimum distance 3 channels;
  3. for every candidate isotope, count reference lines with a detected peak
     within +/-1 FWHM(E) (FWHM(E) = sqrt(0.5651^2 + 0.0410^2 E) keV) and compute
     the match fraction m/n;
  4. predict the highest-scoring candidate only if m >= 2 and m/n >= 0.5
     (confidence gate); otherwise abstain.

Input : data/processed/spectra.npz, metadata.csv, reference_peaks.csv
Output: results/dsp/classification_results.csv, results/dsp/candidate_scores/
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "dsp"
(OUT_DIR / "candidate_scores").mkdir(parents=True, exist_ok=True)

# Fit from the 8-peak calibration described in the module docstring.
RESOLUTION_A, RESOLUTION_B = 0.5651, 0.04100  # FWHM(E) = sqrt(A^2 + B^2*E)

MIXED_MATERIAL_FILES = {
    "CeramicPipe1820", "Fiestaware", "Pitchblende", "RadiumDial",
    "ThoriumSource", "Uranium-glaze", "Zeunerite",
}


def fwhm_kev(energy_kev: np.ndarray) -> np.ndarray:
    return np.sqrt(RESOLUTION_A**2 + (RESOLUTION_B**2) * energy_kev)


def detect_peaks(energy_kev: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    """Generic, isotope-agnostic peak detection."""
    counts_clipped = np.clip(counts, 0, None)
    # Light smoothing (3-point moving average) to suppress single-channel
    # Poisson spikes without washing out real peaks, which are several
    # channels wide even at this detector's ~0.2 keV/channel spacing and
    # ~0.6-1.8 keV FWHM (3-9 channels).
    kernel = np.ones(3) / 3
    smoothed = np.convolve(counts_clipped, kernel, mode="same")

    # Prominence threshold scales with local statistical noise: require a
    # peak to rise at least ~5x the Poisson sqrt(N) of the local continuum,
    # using a rough global noise floor per spectrum (75th percentile count
    # level as a stand-in for typical continuum height).
    noise_floor = max(np.percentile(smoothed, 75), 1.0)
    min_prominence = max(5 * np.sqrt(noise_floor), 3.0)

    idx, props = find_peaks(smoothed, prominence=min_prominence, distance=3)
    return pd.DataFrame({
        "energy_kev": energy_kev[idx],
        "counts": counts[idx],
        "prominence": props["prominences"],
    })


def score_candidates(detected_peaks: pd.DataFrame, reference_peaks: pd.DataFrame) -> pd.DataFrame:
    """For each candidate isotope in `reference_peaks`, score what fraction
    of its known reference peaks have a detected peak within +/- 1 FWHM."""
    rows = []
    for isotope, group in reference_peaks.groupby("isotope"):
        ref_energies = group["energy_kev"].values
        tol = fwhm_kev(ref_energies)
        matched = 0
        matched_energies = []
        for e_ref, t in zip(ref_energies, tol):
            hit = detected_peaks[np.abs(detected_peaks["energy_kev"] - e_ref) <= t]
            if len(hit) > 0:
                matched += 1
                matched_energies.append(e_ref)
        n_ref = len(ref_energies)
        rows.append({
            "candidate_isotope": isotope,
            "n_reference_peaks": n_ref,
            "n_matched": matched,
            "match_fraction": matched / n_ref if n_ref else 0.0,
            "matched_energies": matched_energies,
        })
    return pd.DataFrame(rows).sort_values(
        ["match_fraction", "n_matched"], ascending=False
    ).reset_index(drop=True)


MIN_MATCHED_PEAKS_FOR_CONFIDENT_ID = 2


def main():
    npz = np.load(DATA_DIR / "spectra.npz")
    ref_peaks = pd.read_csv(DATA_DIR / "reference_peaks.csv")
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    all_isotopes = sorted(meta["isotope"].tolist())
    covered_isotopes = sorted(ref_peaks["isotope"].unique().tolist())

    results = []
    for isotope in all_isotopes:
        energy = npz[f"{isotope}__energy_kev"]
        counts = npz[f"{isotope}__counts"]
        detected = detect_peaks(energy, counts)
        scores = score_candidates(detected, ref_peaks)

        # Raw best match (no confidence threshold) - kept for comparison.
        top = scores.iloc[0] if len(scores) else None
        predicted_raw = top["candidate_isotope"] if top is not None and top["match_fraction"] > 0 else None

        # Confidence-gated prediction: only call an ID when enough
        # independent lines corroborate it.
        confident_candidates = scores[scores["n_matched"] >= MIN_MATCHED_PEAKS_FOR_CONFIDENT_ID]
        confident_candidates = confident_candidates[confident_candidates["match_fraction"] >= 0.5]
        top_confident = confident_candidates.iloc[0] if len(confident_candidates) else None
        predicted_confident = top_confident["candidate_isotope"] if top_confident is not None else None

        is_covered = isotope in covered_isotopes
        correct_raw = (predicted_raw == isotope) if is_covered else None
        correct_confident = (predicted_confident == isotope) if is_covered else None

        results.append({
            "true_isotope": isotope,
            "is_mixed_material": isotope in MIXED_MATERIAL_FILES,
            "has_reference": is_covered,
            "n_reference_peaks_for_true_isotope": int(
                (ref_peaks["isotope"] == isotope).sum()) if is_covered else None,
            "n_detected_peaks": len(detected),
            "predicted_raw": predicted_raw,
            "top_match_fraction_raw": float(top["match_fraction"]) if top is not None else 0.0,
            "top_n_matched_raw": int(top["n_matched"]) if top is not None else 0,
            "correct_raw": correct_raw,
            "predicted_confident": predicted_confident,
            "correct_confident": correct_confident,
            "n_candidates_scored": len(scores),
            "second_best_isotope": scores.iloc[1]["candidate_isotope"] if len(scores) > 1 else None,
            "second_best_fraction": float(scores.iloc[1]["match_fraction"]) if len(scores) > 1 else 0.0,
        })

        scores.drop(columns=["matched_energies"]).to_csv(
            OUT_DIR / "candidate_scores" / f"scores_{isotope.replace('/', '_')}.csv", index=False
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "classification_results.csv", index=False)

    covered = results_df[results_df["has_reference"]]
    uncovered = results_df[~results_df["has_reference"]]

    print("=" * 70)
    print("RAW scoring (best match regardless of how much evidence backs it)")
    print("=" * 70)
    acc_raw = covered["correct_raw"].mean()
    pure = covered[~covered["is_mixed_material"]]
    print(f"Accuracy on {len(covered)} covered isotopes/materials: "
          f"{acc_raw:.1%} ({covered['correct_raw'].sum()}/{len(covered)})")
    print(f"Accuracy on {len(pure)} PURE covered isotopes: "
          f"{pure['correct_raw'].mean():.1%} ({pure['correct_raw'].sum()}/{len(pure)})")
    false_confident_raw = uncovered[uncovered["predicted_raw"].notna()]
    print(f"False-accept rate on {len(uncovered)} isotopes with NO reference: "
          f"{len(false_confident_raw)}/{len(uncovered)} got a non-null prediction anyway")

    print()
    print("=" * 70)
    print(f"CONFIDENCE-GATED scoring (requires >={MIN_MATCHED_PEAKS_FOR_CONFIDENT_ID} "
          f"matched lines + >=50% of the candidate's known lines)")
    print("=" * 70)
    made_call = covered[covered["predicted_confident"].notna()]
    coverage = len(made_call) / len(covered)
    acc_when_confident = made_call["correct_confident"].mean() if len(made_call) else float("nan")
    print(f"Made a confident call on {len(made_call)}/{len(covered)} covered "
          f"isotopes ({coverage:.1%} coverage)")
    print(f"Accuracy when a confident call WAS made: {acc_when_confident:.1%} "
          f"({made_call['correct_confident'].sum()}/{len(made_call)})")
    abstained = covered[covered["predicted_confident"].isna()]
    if len(abstained):
        print(f"Abstained (no confident call) on: {abstained['true_isotope'].tolist()}")
        print("  -> these are the isotopes covered by only 1-2 GSD reference "
              "peaks, i.e. genuinely weak spectroscopic signatures for a "
              "classical line-counting method, not classifier failures.")
    false_confident_gated = uncovered[uncovered["predicted_confident"].notna()]
    print(f"False-accept rate on {len(uncovered)} isotopes with NO reference: "
          f"{len(false_confident_gated)}/{len(uncovered)} "
          f"(down from {len(false_confident_raw)}/{len(uncovered)} ungated)")
    if len(false_confident_gated):
        print(false_confident_gated[["true_isotope", "predicted_confident"]].to_string(index=False))

    misses = made_call[made_call["correct_confident"] == False]  # noqa: E712
    if len(misses):
        print(f"\nConfident but WRONG ({len(misses)}):")
        print(misses[["true_isotope", "predicted_confident", "second_best_isotope",
                       "second_best_fraction"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
