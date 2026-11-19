"""Supplementary statistics quoted in the article (rev4).

1. Exact McNemar tests on the 27 isotopes for the resolution change:
   DSP native vs DSP NaI(Tl)-like (abstentions counted as wrong), and
   DSP NaI(Tl)-like vs RF / CNN NaI(Tl)-like.
2. Five-seed CNN accuracy restricted to the 27 isotopes covered by DSP
   (native resolution; per-seed misses from multiseed_results.csv).
3. Trainable-parameter count of the CNN architecture in train_models.py
   (computed analytically, no TensorFlow needed).

Input : results/comparison/fair_scope_comparison.csv, results/ml/multiseed_results.csv
Output: results/comparison/supplementary_tests.json
"""
import ast
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def mcnemar(a, b):
    """Exact two-sided McNemar test. a, b: boolean Series (correct = True)."""
    d_a = int((a & ~b).sum())
    d_b = int((~a & b).sum())
    d = d_a + d_b
    p = 1.0 if d == 0 else float(binomtest(min(d_a, d_b), d, 0.5).pvalue)
    return {"a_right_b_wrong": d_a, "b_right_a_wrong": d_b, "n_discordant": d, "p_value": p}


def cnn_params(n_bins_unused=512, n_classes=36):
    convs = [(1, 16, 9), (16, 32, 7), (32, 64, 5)]
    total = sum(cin * cout * k + cout for cin, cout, k in convs)
    total += 64 * 64 + 64          # dense 64
    total += 64 * n_classes + n_classes  # softmax
    return total


def main():
    f = pd.read_csv(RES / "comparison" / "fair_scope_comparison.csv")
    b = lambda c: f[c].astype(str).eq("True")
    out = {
        "mcnemar_dsp_native_vs_dsp_nai": mcnemar(b("correct_confident"), b("dsp_degraded_correct_gated")),
        "mcnemar_dsp_nai_vs_rf_nai": mcnemar(b("dsp_degraded_correct_gated"), b("rf_correct_degraded")),
        "mcnemar_dsp_nai_vs_cnn_nai": mcnemar(b("dsp_degraded_correct_gated"), b("cnn_correct_degraded")),
    }
    iso27 = set(f["isotope"])
    m = pd.read_csv(RES / "ml" / "multiseed_results.csv")
    per_seed = {}
    for _, r in m.iterrows():
        miss = [x for x in ast.literal_eval(r["cnn_misses_native"]) if x in iso27]
        per_seed[str(r["seed"])] = {"misses": miss, "accuracy": (len(iso27) - len(miss)) / len(iso27)}
    accs = pd.Series([v["accuracy"] for v in per_seed.values()])
    out["cnn_multiseed_27_isotopes_native"] = {
        "per_seed": per_seed, "mean": float(accs.mean()), "std": float(accs.std(ddof=1)),
        "min": float(accs.min()), "max": float(accs.max()),
    }
    out["cnn_trainable_parameters"] = cnn_params()
    (RES / "comparison" / "supplementary_tests.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
