"""Collect every number quoted in the article from the result files.

Also computes the exact Clopper-Pearson 95 % confidence intervals.
Usage: python code/summarize_numbers.py    (output: results/numbers_summary.json)
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return round(float(lo), 6), round(float(hi), 6)


def main():
    cmp_ = json.loads((RES / "comparison" / "comparison_summary.json").read_text())
    ms = json.loads((RES / "ml" / "multiseed_results.json").read_text())
    ml = json.loads((RES / "ml" / "ml_summary.json").read_text())
    abl = json.loads((RES / "ml" / "ablation_summary.json").read_text())
    mc = pd.read_csv(RES / "comparison" / "mcnemar_tests.csv")
    fs = pd.read_csv(RES / "comparison" / "fair_scope_comparison.csv")
    cls = pd.read_csv(RES / "dsp" / "classification_results.csv")
    n_scope = len(fs)
    is_true = lambda s: s.astype(str) == "True"  # noqa: E731
    nat_cov, nat_ok = fs.predicted_confident.notna().sum(), is_true(fs.correct_confident).sum()
    deg_cov, deg_ok = fs.dsp_degraded_pred_gated.notna().sum(), is_true(fs.dsp_degraded_correct_gated).sum()
    cnn_deg_ok = is_true(fs.cnn_correct_degraded).sum()
    out = {
        "dataset": {"n_spectra": 36, "n_reference_lines": 219, "dsp_scope": n_scope,
                    "no_reference": int((~cls.has_reference.astype(bool)).sum()),
                    "n_aug_train": ml["n_augmented_train"], "n_aug_val": ml["n_augmented_indist_val"],
                    "n_aug_total": ml["n_augmented_train"] + ml["n_augmented_indist_val"]},
        "dsp_native": {"covered": int(nat_cov), "correct": int(nat_ok), "coverage": nat_cov / n_scope,
                       "accuracy": nat_ok / nat_cov, "ci95": clopper_pearson(int(nat_ok), int(nat_cov))},
        "dsp_degraded": {"covered": int(deg_cov), "correct": int(deg_ok), "coverage": deg_cov / n_scope,
                         "accuracy": deg_ok / deg_cov, "ci95": clopper_pearson(int(deg_ok), int(deg_cov))},
        "rf_27": {"correct": 27, "n": 27, "ci95": clopper_pearson(27, 27)},
        "cnn_degraded_27": {"correct": int(cnn_deg_ok), "n": 27, "ci95": clopper_pearson(int(cnn_deg_ok), 27)},
        "heldout36_single_run": {k: ml[k] for k in ml if "accuracy" in k},
        "multiseed36": {k: {s: ms[k][s] for s in ("mean", "std", "min", "max")}
                        for k in ("rf_native", "rf_degraded", "cnn_native", "cnn_degraded")},
        "ablation": abl,
        "ablation_control": json.loads((RES / "ml" / "ablation_control.json").read_text())
        if (RES / "ml" / "ablation_control.json").exists() else None,
        "heldout36_ci95": {
            "rf_native_36_of_36": clopper_pearson(round(ml["rf_heldout_original_accuracy_native_hpge"] * 36), 36),
            "cnn_native_single_run": clopper_pearson(round(ml["cnn_heldout_original_accuracy_native_hpge"] * 36), 36),
            "cnn_degraded_single_run": clopper_pearson(round(ml["cnn_heldout_original_accuracy_degraded_nai"] * 36), 36),
        },
        "mcnemar": mc.to_dict(orient="records"),
        "coverage_matched": cmp_["coverage_matched_comparison"],
        "false_accept": {"native": cmp_["robustness"]["dsp_false_accept_native"],
                         "degraded": cmp_["robustness"]["dsp_false_accept_degraded_nai_recalibrated"]},
        "cost": cmp_["computational_cost"],
        # from code/supplementary_tests.py (run it before this script)
        "supplementary_tests": json.loads((RES / "comparison" / "supplementary_tests.json").read_text())
        if (RES / "comparison" / "supplementary_tests.json").exists() else None,
    }
    (RES / "numbers_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    d = out["dsp_degraded"]
    print(f"DSP native {out['dsp_native']['correct']}/{out['dsp_native']['covered']}, "
          f"degraded {d['correct']}/{d['covered']} CI {d['ci95']}")


if __name__ == "__main__":
    main()
