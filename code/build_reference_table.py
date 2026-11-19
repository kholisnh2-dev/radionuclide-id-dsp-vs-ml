"""Build the reference-line table used by the DSP method.

reference_peaks.csv = 179 lines annotated in the GSD pages (26 isotopes)
                    + 40 Lu-177m lines from the INL Gamma-Ray Spectrum Catalog
                      that coincide (within +/-1 FWHM) with peaks detected in the
                      measured Lu-177m spectrum.
Total: 219 lines for 27 isotopes/materials.

Input : data/processed/reference_peaks_gsd.csv, data/processed/reference_lines_inl_lu177m.csv
Output: data/processed/reference_peaks.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main():
    gsd = pd.read_csv(PROC / "reference_peaks_gsd.csv")
    inl = pd.read_csv(PROC / "reference_lines_inl_lu177m.csv")
    table = pd.concat([gsd, inl], ignore_index=True).sort_values("isotope", kind="mergesort")
    table.to_csv(PROC / "reference_peaks.csv", index=False, lineterminator="\n")
    print(f"{len(table)} reference lines for {table.isotope.nunique()} isotopes/materials")


if __name__ == "__main__":
    main()
