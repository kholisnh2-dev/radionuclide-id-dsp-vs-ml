[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24356057.svg)](https://doi.org/10.5281/zenodo.24356057)

# Radionuclide identification from open-access gamma-ray spectra: classical peak matching vs. machine learning

Data, code, trained models and figures for the article

> K. Nurhanafi et al., *Classical Peak Matching Versus Machine Learning for Radionuclide Identification Using Open-Access Gamma-Ray Spectra: A Reproducible Feasibility Study*, submitted to Atom Indonesia.

All inputs are open-access: 36 measured gamma-ray spectra from the OpenGammaProject Gamma-Spectrum-Database (GSD) and reference energy lines from the GSD pages and the INL Gamma-Ray Spectrum Catalog.

## Contents

| Folder | Content |
|---|---|
| `data/raw/` | 36 measured spectra as downloaded from the GSD (HTML, GPL-3.0) |
| `data/processed/` | `spectra.npz` (energy, counts and background per spectrum), `metadata.csv`, reference lines: `reference_peaks_gsd.csv` (179 lines), `reference_lines_inl_lu177m.csv` (40 lines), `reference_peaks.csv` (combined, 219 lines) |
| `data/augmented/` | `rebinned_real.npz` (36 original spectra on the 512-bin grid, used as test data), `augmented_train.npz` (10,800 augmented training spectra) |
| `models/` | `rf_model.joblib` (Random Forest), `cnn_model.keras` (1D CNN), `label_encoder.joblib` |
| `code/` | all scripts; each script lists its inputs, outputs and parameters at the top |
| `results/dsp/` | DSP predictions per spectrum, candidate scores, resolution-calibration peaks |
| `results/ml/` | held-out results, five-seed results, ablation test and its controls |
| `results/comparison/` | 27-isotope comparison table, McNemar tests, coverage-accuracy curves, computational cost, `supplementary_tests.json` (DSP native vs NaI-like McNemar, 27-isotope five-seed CNN, CNN parameter count) |
| `results/numbers_summary.json` | every number quoted in the article |
| `figures/` | Figs. 1-8 of the article (PNG 600 dpi and PDF) |

## Main parameters

| Step | Parameters |
|---|---|
| DSP | 3-point moving average; `find_peaks` prominence ≥ max(5√N75, 3), distance 3 channels; match window ±1 FWHM(E), FWHM(E) = √(0.5651² + 0.0410²E) keV; gate m ≥ 2 and m/n ≥ 0.5 |
| Re-binning | 512 bins, 0-1900 keV (counts summed) |
| Augmentation | gain N(1, 0.003); offset N(0, 0.5 keV); time factor 10^u, u ~ U(-1.5, 0.3), Poisson sampling; background × U(0, 2); with p = 0.5 broadening to FWHM(E) = √(3² + 1.797²E) keV × U(0.3, 1.0); 300 variants per class; seed 20260814 |
| Features | normalised to 10^5 counts, ln(1 + x) |
| Random Forest | 300 trees |
| CNN | Conv1D 16/9, MaxPool 4, Conv1D 32/7, MaxPool 4, Conv1D 64/5, global average pooling, dropout 0.3, dense 64, dropout 0.2, softmax 36; Adam 1e-3, 30 epochs, batch 64 |

## Reproducing

Python 3.11. Install the packages listed in `requirements.txt`, then run from the repository root:

```bash
python code/parse_spectra.py          # data/raw -> data/processed
python code/build_reference_table.py  # 179 GSD + 40 INL lines -> reference_peaks.csv
python code/calibrate_resolution.py   # HPGe resolution check
python code/dsp_identification.py     # classical DSP -> results/dsp
python code/augmentation.py           # -> data/augmented
python code/train_models.py           # -> models, results/ml
python code/ablation_test.py          # -> results/ml
python code/ablation_control.py       # controls: only peak regions kept; 256 random bins (20 seeds)
python code/multiseed_test.py         # five seeds -> results/ml
python code/comparison.py             # -> results/comparison
python code/supplementary_tests.py    # McNemar DSP native vs NaI-like, 27-isotope five-seed CNN, CNN parameter count
python code/summarize_numbers.py      # -> results/numbers_summary.json
python code/make_figures.py           # -> figures
```

Reproducibility check (re-running every script in a fresh copy of this repository):
- `spectra.npz`, `metadata.csv`, `reference_peaks.csv`, `rebinned_real.npz`, `augmented_train.npz`, the candidate scores, the ablation results and the comparison results (`fair_scope_comparison.csv`, `mcnemar_tests.csv`, `coverage_accuracy_curves.npz`) are reproduced identically.
- All DSP predictions are reproduced identically. With SciPy 1.18 one spectrum (Co-57) yields 77 instead of 76 detected peaks; its prediction is unchanged.
- Random Forest results are reproduced identically.
- CNN training is not bit-identical across TensorFlow versions. The published models were trained with Keras 3.8. A rerun with TensorFlow 2.21 / Keras 3.15 gave a five-seed CNN accuracy of 93.9 % ± 2.3 % (native) and 85.6 % ± 2.3 % (NaI-like), compared with 92.2 % ± 2.3 % and 86.1 % ± 5.9 % reported.
- The canonical models (`train_models.py`) and the five-seed runs (`multiseed_test.py`) are trained separately. In the canonical run the seed 20260814 controls only the augmentation (train/validation split and RF use `random_state=0`; CNN initialisation is not seeded). In the five-seed runs each seed controls augmentation, split, RF and CNN initialisation. The five-seed run with seed 20260814 is therefore a different model from the canonical one (CNN 94.4 % vs 97.2 % on the 36 spectra).
- Inference times depend on the computer.

`models/rf_model.joblib` (61.7 MB) and `data/augmented/augmented_train.npz` (23 MB) are distributed through the Zenodo archive (they are excluded from git by `.gitignore`); both can be regenerated with the scripts above. Loading `rf_model.joblib` requires scikit-learn 1.5.2.

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Measured spectra and GSD reference lines | https://github.com/OpenGammaProject/Gamma-Spectrum-Database (commit 60d3382af94f3db69775de0f7795dbd21386c768) | GPL-3.0 |
| Lu-177m reference lines | Idaho National Laboratory, Gamma-Ray Spectrum Catalog, https://inl.gov/national-security/nuclear-nonproliferation/gamma-ray/ | line energies only (factual data) |
| Code, processed data, models, results, figures | this repository | GPL-3.0 (see `LICENSE`) |

## Citation

See `CITATION.cff`. Zenodo DOI: [10.5281/zenodo.24356057](https://doi.org/10.5281/zenodo.24356057).
