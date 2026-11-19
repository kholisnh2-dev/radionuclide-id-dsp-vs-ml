"""Parse the 36 Gamma-Spectrum-Database (GSD) HTML files into arrays.

Each GSD page stores its plot data as JSON inside a Plotly call. For 33 files
two traces exist: "Clean Spectrum" (already background-subtracted by GSD,
used as the isotope spectrum) and "Background" (kept for data augmentation).
Bi-207, Cs-137 and Fe-55 contain a single "Spectrum" trace, used as is.
Energies are already calibrated in keV (about 8192 channels per file).
Peak annotations embedded in the pages are written as reference lines.

Input : data/raw/*.html
Output: data/processed/spectra.npz       energy, counts, background per isotope
        data/processed/metadata.csv      one row per spectrum
        data/processed/reference_peaks_gsd.csv   179 annotated lines (26 isotopes)
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIXED_MATERIAL_FILES = {
    "CeramicPipe1820", "Fiestaware", "Pitchblende", "RadiumDial",
    "ThoriumSource", "Uranium-glaze", "Zeunerite",
}


def extract_plotly_args(js: str) -> list[str]:
    """Split the arguments of `Plotly.newPlot('id', ARG1, ARG2, ARG3)`
    into their raw text, respecting bracket/string nesting (a plain
    top-level comma split would break on commas inside the JSON)."""
    m = re.search(r"Plotly\.newPlot\('([^']+)',", js)
    if not m:
        raise ValueError("Plotly.newPlot(...) call not found")
    start = m.end()
    args, depth, in_str, str_char, escape = [], 0, False, "", False
    arg_start = start
    i = start
    while i < len(js):
        c = js[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == str_char:
                in_str = False
        else:
            if c in "\"'":
                in_str, str_char = True, c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
            elif c == "," and depth == 0:
                args.append(js[arg_start:i])
                arg_start = i + 1
            elif c == ")" and depth == 0:
                args.append(js[arg_start:i])
                break
        i += 1
    return [a.strip() for a in args]


def get_source_trace(traces: list[dict]) -> tuple[dict, str]:
    by_name = {t.get("name"): t for t in traces}
    if "Clean Spectrum" in by_name:
        return by_name["Clean Spectrum"], "Clean Spectrum"
    if "Spectrum" in by_name:
        return by_name["Spectrum"], "Spectrum"
    raise ValueError(f"No recognized source trace among {list(by_name)}")


def parse_file(path: Path) -> dict:
    content = path.read_text()
    m = re.search(r'<script type="text/javascript">(.*?)</script>', content, re.S)
    args = extract_plotly_args(m.group(1))
    data = json.loads(args[0])
    layout = json.loads(args[1])

    source_trace, source_name = get_source_trace(data)
    bg_trace = next((t for t in data if t.get("name") == "Background"), None)

    energy_kev = np.array(source_trace["x"], dtype=float)
    counts = np.array(source_trace["y"], dtype=float)
    background_counts = np.array(bg_trace["y"], dtype=float) if bg_trace else None

    # GSD's own labeled reference peaks: annotations whose hovertext ends
    # in "keV" (filters out the site-credit annotation, which has
    # hovertext=None).
    peaks = []
    for a in layout.get("annotations", []):
        ht = a.get("hovertext") or ""
        if ht.endswith("keV"):
            peaks.append({"isotope_label": a.get("text"), "energy_kev": a.get("x")})

    isotope = path.stem
    return {
        "isotope": isotope,
        "is_mixed_material": isotope in MIXED_MATERIAL_FILES,
        "source_trace_name": source_name,
        "has_separate_background": bg_trace is not None,
        "n_channels": len(energy_kev),
        "energy_min_kev": float(energy_kev.min()),
        "energy_max_kev": float(energy_kev.max()),
        "energy_step_kev": float(np.median(np.diff(energy_kev))),
        "total_counts": float(counts.sum()),
        "negative_bins": int((counts < 0).sum()),
        "n_reference_peaks": len(peaks),
        "energy_kev": energy_kev,
        "counts": counts,
        "background_counts": background_counts,
        "peaks": peaks,
    }


def main():
    files = sorted(RAW_DIR.glob("*.html"))
    parsed = [parse_file(f) for f in files]

    # 1. metadata.csv
    meta_rows = [{k: v for k, v in p.items()
                  if k not in ("energy_kev", "counts", "background_counts", "peaks")}
                 for p in parsed]
    meta_df = pd.DataFrame(meta_rows).sort_values("isotope").reset_index(drop=True)
    meta_df.to_csv(OUT_DIR / "metadata.csv", index=False)

    # 2. spectra.npz - one array per isotope per field, keyed "ISOTOPE__field"
    npz_payload = {}
    for p in parsed:
        key = p["isotope"]
        npz_payload[f"{key}__energy_kev"] = p["energy_kev"]
        npz_payload[f"{key}__counts"] = p["counts"]
        if p["background_counts"] is not None:
            npz_payload[f"{key}__background_counts"] = p["background_counts"]
    np.savez_compressed(OUT_DIR / "spectra.npz", **npz_payload)

    # 3. reference_peaks.csv
    peak_rows = []
    for p in parsed:
        for peak in p["peaks"]:
            peak_rows.append({
                "isotope": p["isotope"],
                "annotation_label": peak["isotope_label"],
                "energy_kev": peak["energy_kev"],
            })
    peaks_df = pd.DataFrame(peak_rows).sort_values(["isotope", "energy_kev"]).reset_index(drop=True)
    peaks_df.to_csv(OUT_DIR / "reference_peaks_gsd.csv", index=False)

    import hashlib
    hashes = {}
    for p in parsed:
        h = hashlib.md5(np.round(p["counts"], 3).tobytes()).hexdigest()
        hashes.setdefault(h, []).append(p["isotope"])
    dupes = {h: v for h, v in hashes.items() if len(v) > 1}

    print(f"Parsed {len(parsed)} files.")
    print(f"Isotopes with >=1 GSD-embedded reference peak: "
          f"{(meta_df['n_reference_peaks'] > 0).sum()} / {len(meta_df)}")
    print(f"Isotopes with a separate background trace: "
          f"{meta_df['has_separate_background'].sum()} / {len(meta_df)}")
    print(f"Files with negative post-subtraction bins: "
          f"{(meta_df['negative_bins'] > 0).sum()} / {len(meta_df)}")
    if dupes:
        print("WARNING - duplicate source spectra detected:", dupes)
    else:
        print("Uniqueness check passed: all 36 source spectra are distinct.")

    return meta_df, peaks_df, parsed


if __name__ == "__main__":
    main()
