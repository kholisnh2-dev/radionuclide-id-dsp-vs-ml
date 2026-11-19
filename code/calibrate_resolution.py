"""Verify the HPGe energy-resolution model FWHM(E) = sqrt(a^2 + b^2 E).

A Gaussian plus linear background is fitted to ten strong, isolated single
peaks (59.5-1460.8 keV); the fitted widths are stored for inspection.

Input : data/processed/spectra.npz
Output: results/dsp/resolution_calibration_peaks.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "dsp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate strong, isolated, single-photopeak reference lines: isotopes
# with only 1-2 reference peaks in reference_peaks.csv (naturally the
# strongest, cleanest lines in the dataset). Excludes Na-22 511.2 keV
# (annihilation broadening) and the two near-zero-energy backscatter/X-ray
# lines (Cs-137 31.8 keV, Cr-51 0.17 keV).
CANDIDATES = [
    ("Am-241", 59.54),
    ("Co-57", 122.06),
    ("Co-57", 136.47),
    ("Tc-99m", 140.58),
    ("Cr-51", 319.69),
    ("Cs-137", 661.66),
    ("Co-60", 1173.24),
    ("Co-60", 1332.50),
    ("Na-22", 1274.54),
    ("K-40", 1460.80),
]

WINDOW_KEV = 6.0  # half-width of the local fit window around each nominal energy


def gaussian_plus_baseline(x, amp, mu, sigma, b0, b1):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x


def fit_one_peak(energy_kev, counts, nominal_energy):
    mask = np.abs(energy_kev - nominal_energy) <= WINDOW_KEV
    x = energy_kev[mask]
    y = counts[mask].astype(float)
    if len(x) < 8 or y.max() <= 0:
        return None

    amp0 = y.max() - np.median(y)
    mu0 = x[np.argmax(y)]
    sigma0 = 0.5  # keV, reasonable HPGe starting guess
    b00 = np.median(y)
    b10 = 0.0
    try:
        popt, _ = curve_fit(
            gaussian_plus_baseline, x, y,
            p0=[amp0, mu0, sigma0, b00, b10],
            maxfev=10000,
        )
    except RuntimeError:
        return None

    amp, mu, sigma, b0, b1 = popt
    sigma = abs(sigma)
    fwhm = sigma * 2.3548
    y_pred = gaussian_plus_baseline(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"centroid_kev": mu, "fwhm_kev": fwhm, "r2": r2, "amplitude": amp}


def two_param_model(E, a, b):
    return np.sqrt(a**2 + (b**2) * E)


def main():
    npz = np.load(DATA_DIR / "spectra.npz")
    rows = []
    for isotope, nominal_e in CANDIDATES:
        energy = npz[f"{isotope}__energy_kev"]
        counts = npz[f"{isotope}__counts"].astype(float)
        bg_key = f"{isotope}__background_counts"
        if bg_key in npz.files:
            counts = np.clip(counts - npz[bg_key], 0, None)

        fit = fit_one_peak(energy, counts, nominal_e)
        row = {"isotope": isotope, "nominal_energy_kev": nominal_e}
        if fit is None:
            row.update({"centroid_kev": np.nan, "fwhm_kev": np.nan,
                         "r2": np.nan, "kept_in_fit": False,
                         "reject_reason": "fit_failed"})
        else:
            row.update(fit)
            # Quality gate: reasonable fit (R^2 > 0.9) and physically
            # sane FWHM for HPGe (0.2-5 keV over this energy range).
            keep = fit["r2"] > 0.9 and 0.2 <= fit["fwhm_kev"] <= 5.0
            row["kept_in_fit"] = keep
            row["reject_reason"] = "" if keep else "quality_gate_failed"
        rows.append(row)

    df = pd.DataFrame(rows)

    kept = df[df["kept_in_fit"]]
    a_refit = b_refit = max_residual = np.nan
    if len(kept) >= 3:
        popt, _ = curve_fit(
            two_param_model, kept["centroid_kev"].values, kept["fwhm_kev"].values,
            p0=[0.5, 0.04], maxfev=10000,
        )
        a_refit, b_refit = popt
        pred = two_param_model(kept["centroid_kev"].values, a_refit, b_refit)
        max_residual = np.max(np.abs(pred - kept["fwhm_kev"].values))

    df.to_csv(OUT_DIR / "resolution_calibration_peaks.csv", index=False)

    print(f"Fitted {len(kept)}/{len(df)} candidate peaks kept in final 2-parameter fit.")
    print(df[["isotope", "nominal_energy_kev", "centroid_kev", "fwhm_kev", "r2", "kept_in_fit"]]
          .to_string(index=False))
    print()
    print(f"Refit: FWHM(E) = sqrt({a_refit:.4f}^2 + {b_refit:.4f}^2 * E)")
    print(f"Max residual across kept peaks: {max_residual:.3f} keV")
    print(f"Hardcoded pipeline constants (dsp_identification.py): "
          f"FWHM(E) = sqrt(0.5651^2 + 0.0410^2 * E)")
    print(f"Saved per-peak table -> {OUT_DIR / 'resolution_calibration_peaks.csv'}")


if __name__ == "__main__":
    main()
