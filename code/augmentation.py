"""Physics-based data augmentation and re-binning.

Every measured spectrum is re-binned (counts summed) onto a common grid of
512 bins over 0-1900 keV (about 3.7 keV per bin). 300 variants per class are
then generated, each with:
  - gain ~ N(1, 0.003) and offset ~ N(0, 0.5 keV) applied to the energy axis;
  - acquisition-time factor 10^u, u ~ U(-1.5, 0.3), followed by Poisson sampling;
  - measured background (when available) scaled by U(0, 2), Poisson-sampled, added;
  - with probability 0.5, Gaussian broadening toward a NaI(Tl)-like resolution,
    FWHM(E) = sqrt(3^2 + 1.797^2 E) keV times a fraction U(0.3, 1.0).
Random seed 20260814.

Input : data/processed/spectra.npz, metadata.csv
Output: data/augmented/augmented_train.npz (10,800 variants)
        data/augmented/rebinned_real.npz   (the 36 original spectra on the grid)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "augmented"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 512
GRID_MAX_KEV = 1900.0
GRID_EDGES = np.linspace(0, GRID_MAX_KEV, N_BINS + 1)
GRID_CENTERS = (GRID_EDGES[:-1] + GRID_EDGES[1:]) / 2

RNG = np.random.default_rng(20260814)

NAI_FWHM_AT_662 = 0.07 * 661.66  # keV

NAI_FLOOR_KEV = 3.0  # ASSUMED (no second open-access NaI calibration point available)
NAI_B = np.sqrt(max(NAI_FWHM_AT_662**2 - NAI_FLOOR_KEV**2, 0.0) / 661.66)


def rebin(energy_kev: np.ndarray, counts: np.ndarray,
          gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Histogram-based rebinning onto the common grid - redistributes counts
    (not interpolates the rate curve), which is the physically correct way
    to combine count data into coarser bins and preserves total counts.
    `gain`/`offset` optionally apply a calibration perturbation before
    rebinning: shifted_energy = energy*gain + offset.
    """
    shifted = energy_kev * gain + offset
    rebinned, _ = np.histogram(shifted, bins=GRID_EDGES, weights=counts)
    return rebinned


def nai_like_fwhm_kev(energy_kev: np.ndarray) -> np.ndarray:
    energy_kev = np.clip(energy_kev, 1.0, None)
    return np.sqrt(NAI_FLOOR_KEV**2 + (NAI_B**2) * energy_kev)


def degrade_resolution(binned_counts: np.ndarray, fwhm_fraction: float) -> np.ndarray:
    """Gaussian-broaden a grid-binned spectrum toward a target resolution.
    fwhm_fraction=0 -> no change (native HPGe resolution, already sharp
    relative to this grid's ~3.7 keV/bin width). fwhm_fraction=1 -> fully
    NaI-typical broadening per `nai_like_fwhm_kev`.
    """
    if fwhm_fraction <= 0:
        return binned_counts
    out = np.zeros_like(binned_counts, dtype=float)
    bin_width = GRID_EDGES[1] - GRID_EDGES[0]
    target_fwhm = nai_like_fwhm_kev(GRID_CENTERS) * fwhm_fraction
    sigma_bins = np.clip((target_fwhm / 2.3548) / bin_width, 0.3, None)
    idx = np.arange(len(binned_counts))
    for i in np.nonzero(binned_counts > 0)[0]:
        s = sigma_bins[i]
        window = slice(max(0, i - int(6 * s)), min(len(binned_counts), i + int(6 * s) + 1))
        weights = np.exp(-0.5 * ((idx[window] - i) / s) ** 2)
        weights /= weights.sum()
        out[window] += binned_counts[i] * weights
    return out


def augment_one(energy_kev, counts, background_counts, n_variants: int, degrade_p: float = 0.5):
    """Generate `n_variants` physically-perturbed copies of one real
    spectrum. `degrade_p` = fraction of variants that also get resolution
    degradation applied (kept partial, not universal, so the model also
    sees native-resolution examples - a 50/50 mix supports later comparing
    performance with vs without the NaI-like regime)."""
    variants = []
    has_bg = background_counts is not None
    for _ in range(n_variants):
        gain = RNG.normal(1.0, 0.003)      # ~0.3% gain jitter
        offset = RNG.normal(0.0, 0.5)      # +/-0.5 keV offset jitter (1 sigma)
        time_factor = 10 ** RNG.uniform(-1.5, 0.3)  # ~0.03x-2x the original acquisition

        rebinned_src = rebin(energy_kev, counts, gain=gain, offset=offset)
        rate = np.clip(rebinned_src, 0, None) * time_factor
        resampled = RNG.poisson(rate).astype(float)

        if has_bg:
            bg_rebinned = rebin(energy_kev, background_counts, gain=gain, offset=offset)
            bg_scale = RNG.uniform(0.0, 2.0)
            bg_rate = np.clip(bg_rebinned, 0, None) * bg_scale * time_factor
            resampled = resampled + RNG.poisson(bg_rate).astype(float)

        if RNG.uniform() < degrade_p:
            frac = RNG.uniform(0.3, 1.0)
            resampled = degrade_resolution(resampled, frac)

        variants.append(resampled)
    return np.array(variants)


def main(n_variants_per_isotope: int = 300):
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

        real_rebinned[isotope] = rebin(energy, counts)

        variants = augment_one(energy, counts, background, n_variants_per_isotope)
        aug_X.append(variants)
        aug_y.extend([isotope] * len(variants))

    aug_X = np.concatenate(aug_X, axis=0)
    aug_y = np.array(aug_y)

    np.savez_compressed(
        OUT_DIR / "rebinned_real.npz",
        **{iso: arr for iso, arr in real_rebinned.items()},
        grid_centers=GRID_CENTERS,
    )
    np.savez_compressed(OUT_DIR / "augmented_train.npz", X=aug_X, y=aug_y, grid_centers=GRID_CENTERS)

    print(f"Rebinned {len(real_rebinned)} real spectra onto a {N_BINS}-bin "
          f"(0-{GRID_MAX_KEV:.0f} keV) common grid.")
    print(f"Generated {len(aug_y)} augmented training variants "
          f"({n_variants_per_isotope}/isotope x {len(isotopes)} isotopes).")
    print(f"Isotopes without a background trace (skipped background-mixing "
          f"augmentation): "
          f"{[iso for iso in isotopes if f'{iso}__background_counts' not in npz.files]}")


if __name__ == "__main__":
    main()
