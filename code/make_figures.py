"""Make the eight article figures (PNG 600 dpi + PDF) from the repository data.

Fig. 1 research workflow
Fig. 2 example measured spectra with reference lines
Fig. 3 classical DSP processing steps for one spectrum (Ba-133)
Fig. 4 detector resolution models and an example of resolution degradation
Fig. 5 data augmentation examples (Eu-152)
Fig. 6 DSP outcome versus number of reference lines
Fig. 7 five-seed results, ablation test and its controls
Fig. 8 coverage-accuracy curves

The DSP and augmentation steps are imported from dsp_identification.py and
augmentation.py, so the illustrations follow exactly the code used for the
results. Colours: red = DSP, blue = RF, green = CNN in result figures; black,
grey, orange and purple are used for illustrative curves.
Usage: python code/make_figures.py   (output: figures/)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import augmentation as aug  # noqa: E402
import dsp_identification as dsp  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
RED, BLUE, GREEN = "#D00000", "#0033CC", "#008A00"          # DSP, RF, CNN
GREY, ORANGE, PURPLE = "#7F7F7F", "#E07B00", "#7030A0"      # illustrative curves
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.9, "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.grid": False, "lines.linewidth": 1.2, "lines.markersize": 4.5,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "legend.borderpad": 0.35, "legend.handlelength": 1.6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42, "mathtext.default": "regular",
})
W2, W1 = 7.0, 3.4  # two-column / one-column width (inch)
SPEC_LW = 0.8      # spectrum line width (pt)


def label(ax, s, x=0.03, y=0.95, ha="left", va="top"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=9.5, fontweight="bold")


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


spectra = np.load(PROC / "spectra.npz")
ref = pd.read_csv(PROC / "reference_peaks.csv")


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    ax.set_axis_off(); ax.set_xlim(0, 104); ax.set_ylim(0, 47)

    def box(x, y, w, h, title, body):
        ax.add_patch(Rectangle((x, y), w, h, fc="#E7E6E6", ec="#404040", lw=0.8))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=7.4, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=7.4, linespacing=1.22)

    def elbow(pts, dashed=False):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="black", lw=0.9, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                    color="black", lw=0.9, shrinkA=0, shrinkB=0))

    box(0, 12, 18, 22, "Open-access data", "36 measured\nspectra\n(GSD; 35 HPGe)\n219 reference lines\n(GSD + INL)")
    box(22, 12, 18, 22, "Preprocessing", "HTML to arrays:\nenergy (keV),\ncounts,\nbackground trace")
    box(45, 30, 17, 14, "Peak search", "3-point\nsmoothing,\nprominence\nthreshold")
    box(66, 30, 17, 14, "Library matching", "±1 FWHM window,\n$f_k = m_k/r_k$,\nconfidence gate")
    box(45, 5, 17, 14, "Augmentation", "512-bin\nre-binning,\n10,800 variants")
    box(66, 5, 17, 14, "Model training", "Random Forest,\n1D CNN")
    box(87, 7, 17, 35, "Evaluation", "36 original\nspectra;\n27-isotope\ncommon scope;\nnative and\nNaI(Tl)-like\nresolution;\nMcNemar test,\nClopper–Pearson\nCI; model size")
    ax.text(45, 45.3, "Classical DSP", fontsize=8, style="italic", va="bottom")
    ax.text(45, 20.0, "Machine learning", fontsize=8, style="italic", va="bottom")
    elbow([(18, 23), (22, 23)])
    elbow([(40, 23), (42.5, 23), (42.5, 37), (45, 37)])
    elbow([(42.5, 23), (42.5, 12), (45, 12)])
    elbow([(62, 37), (66, 37)])
    elbow([(62, 12), (66, 12)])
    elbow([(83, 37), (85, 37), (85, 30), (87, 30)])
    elbow([(83, 12), (85, 12), (85, 19), (87, 19)])
    elbow([(31, 12), (31, 1.5), (95.5, 1.5), (95.5, 7)], dashed=True)
    ax.text(63, 2.1, "original spectra used directly as test data (never augmented)", fontsize=7.4,
            ha="center", va="bottom", style="italic")
    save(fig, "Fig1_workflow")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.3))
    for ax, iso, lab in zip(axes.ravel(), ["Cs-137", "Co-60", "Ba-133", "Eu-152"], "abcd"):
        e, c = spectra[f"{iso}__energy_kev"], np.clip(spectra[f"{iso}__counts"], 0.5, None)
        m = (e > 20) & (e < 1600)
        for E in ref.loc[ref.isotope == iso, "energy_kev"]:
            if 20 < E < 1600:
                ax.axvline(E, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.plot(e[m], c[m], color="black", lw=SPEC_LW, zorder=2)
        logy(ax); ax.set_xlim(0, 1600); ax.set_ylim(0.5, c[m].max() * 6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel("Energy (keV)"); ax.set_ylabel("Counts per channel")
        label(ax, f"({lab}) {iso}", x=0.97, ha="right")
    h = [plt.Line2D([], [], color="black", lw=1), plt.Line2D([], [], color=GREY, lw=1, ls=(0, (4, 2)))]
    fig.legend(h, ["Measured spectrum (GSD, HPGe)", "Reference line energy"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Fig2_example_spectra")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    iso = "Ba-133"
    e, c = spectra[f"{iso}__energy_kev"], spectra[f"{iso}__counts"]
    sm = np.convolve(np.clip(c, 0, None), np.ones(3) / 3, mode="same")
    det = dsp.detect_peaks(e, c)
    scores = dsp.score_candidates(det, ref)
    lines = ref.loc[ref.isotope == iso, "energy_kev"].values
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W2, 2.8), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    m = (e > 20) & (e < 450)
    a1.plot(e[m], np.clip(c[m], 0.5, None), color=GREY, lw=SPEC_LW, label="Measured")
    a1.plot(e[m], np.clip(sm[m], 0.5, None), color="black", lw=SPEC_LW, label="Smoothed")
    d = det[(det.energy_kev > 20) & (det.energy_kev < 450)]
    a1.plot(d.energy_kev, d.counts * 1.8, "v", color=RED, ms=3, label="Detected peak")
    logy(a1); a1.set_xlim(20, 450); a1.set_ylim(0.5, 4e5)
    a1.set_xlabel("Energy (keV)"); a1.set_ylabel("Counts per channel")
    a1.legend(loc="upper right"); label(a1, "(a)", y=0.05, va="bottom")
    lo, hi = 260, 400
    m = (e > lo) & (e < hi)
    for E in lines:
        if lo < E < hi:
            w = dsp.fwhm_kev(np.array([E]))[0]
            a2.axvspan(E - w, E + w, color=GREY, alpha=0.35, lw=0)
            a2.axvline(E, color="black", lw=0.8, ls=(0, (4, 2)))
    a2.plot(e[m], np.clip(sm[m], 0.5, None), color="black", lw=SPEC_LW)
    d = det[(det.energy_kev > lo) & (det.energy_kev < hi)]
    a2.plot(d.energy_kev, d.counts * 1.8, "v", color=RED, ms=3.5)
    logy(a2); a2.set_xlim(lo, hi); a2.set_ylim(0.5, 4e5); a2.set_xlabel("Energy (keV)")
    h = [plt.Line2D([], [], color="black", lw=0.8, ls=(0, (4, 2))), Patch(color=GREY, alpha=0.35),
         plt.Line2D([], [], color=RED, marker="v", ls="", ms=3.5)]
    a2.legend(h, ["Reference line", "±1 FWHM window", "Detected peak"], loc="upper right", fontsize=8)
    label(a2, "(b)", y=0.05, va="bottom")
    top = scores.head(6).iloc[::-1]
    ok = (top.n_matched >= 2) & (top.match_fraction >= 0.5)
    y = np.arange(len(top))
    a3.barh(y, top.match_fraction, color=["#BFBFBF" if o else "white" for o in ok], edgecolor="black",
            lw=0.7, height=0.62)
    for yi, (f, mm, n) in enumerate(zip(top.match_fraction, top.n_matched, top.n_reference_peaks)):
        a3.text(f + 0.03, yi, f"{mm}/{n}", va="center", fontsize=8)
    a3.axvline(0.5, color=RED, lw=0.9, ls=(0, (4, 2)))
    a3.set_yticks(y); a3.set_yticklabels(top.candidate_isotope)
    a3.tick_params(axis="y", which="minor", left=False, right=False)
    a3.set_xlim(0, 1.35); a3.set_xlabel("Match fraction $f_k = m_k/r_k$")
    a3.text(0.53, -0.95, "gate", color=RED, fontsize=8)
    a3.set_ylim(-1.15, len(top) - 0.4); label(a3, "(c)", x=0.97, y=0.05, ha="right", va="bottom")
    fig.tight_layout(w_pad=0.6)
    save(fig, "Fig3_dsp_steps")
    return scores.head(3)


# ---------------------------------------------------------------- Fig. 4
def fig4():
    iso = "Eu-152"
    e, c = spectra[f"{iso}__energy_kev"], np.clip(spectra[f"{iso}__counts"], 0, None)
    native = aug.rebin(e, c)
    deg = aug.degrade_resolution(native, 1.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1, 1.6]})
    ee = np.linspace(10, 1900, 400)
    a1.plot(ee, np.sqrt(0.5651**2 + 0.0410**2 * ee), color="black", label="HPGe (fitted)")
    a1.plot(ee, aug.nai_like_fwhm_kev(ee), color=ORANGE, ls=(0, (5, 2)), label="NaI(Tl)-like (assumed)")
    a1.set_xlim(0, 1900); a1.set_ylim(0, 85)
    a1.set_xlabel("Energy (keV)"); a1.set_ylabel("FWHM (keV)")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86)); label(a1, "(a)")
    norm = native.sum()
    a2.step(aug.GRID_CENTERS, native / norm, where="mid", color="black", lw=SPEC_LW, label="Native (HPGe)")
    a2.plot(aug.GRID_CENTERS, deg / norm, color=ORANGE, lw=1.1, ls=(0, (5, 2)), label="NaI(Tl)-like")
    logy(a2); a2.set_xlim(0, 1500); a2.set_ylim(1e-5, 0.6)
    a2.set_xlabel("Energy (keV)"); a2.set_ylabel("Normalized counts per bin")
    a2.legend(loc="upper right"); label(a2, f"(b) {iso}", y=0.05, va="bottom")
    fig.tight_layout()
    save(fig, "Fig4_resolution")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    iso = "Eu-152"
    e = spectra[f"{iso}__energy_kev"]
    c = np.clip(spectra[f"{iso}__counts"], 0, None)
    bg = np.clip(spectra[f"{iso}__background_counts"], 0, None)
    x = aug.GRID_CENTERS
    base = aug.rebin(e, c)
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6))
    A = axes.ravel()

    def show(ax, y, **kw):
        ax.step(x, np.clip(y, 0.5, None), where="mid", lw=kw.pop("lw", SPEC_LW), **kw)

    show(A[0], base, color="black", label="Original")
    for tf_, col in ((0.3, PURPLE), (0.03, ORANGE)):
        show(A[1], rng.poisson(base * tf_), color=col, label=f"Time factor {tf_}")
    show(A[1], base, color="black", label="Original")
    for g, o, col, ls, lab in ((1.006, 1.0, ORANGE, "-", "Gain 1.006, +1.0 keV"),
                               (0.994, -1.0, PURPLE, "-", "Gain 0.994, −1.0 keV")):
        show(A[2], aug.rebin(e, c, gain=g, offset=o), color=col, ls=ls, label=lab, lw=1.1)
    show(A[2], base, color="black", label="Original", lw=1.1)
    A[2].set_xlim(325, 365)
    show(A[3], base + rng.poisson(aug.rebin(e, bg) * 2.0), color=ORANGE, label="+ 2 × background")
    show(A[3], base, color="black", label="Original")
    show(A[3], aug.rebin(e, bg), color=GREY, label="Background only")
    for fr, col in ((1.0, ORANGE), (0.5, PURPLE)):
        show(A[4], aug.degrade_resolution(base, fr), color=col, label=f"Width fraction {fr}", lw=1.0)
    show(A[4], base, color="black", label="Original")
    aug.RNG = np.random.default_rng(5)
    variants = aug.augment_one(e, c, bg, 3)
    for k, (v, col) in enumerate(zip(variants, (ORANGE, PURPLE, GREY))):
        A[5].step(x, np.clip(v / v.sum(), 1e-9, None) * 10.0 ** (-2 * k), where="mid", lw=SPEC_LW,
                  color=col, label=f"Variant {k + 1}")
    for i, ax in enumerate(A):
        if i != 2:
            logy(ax)
            ax.set_xlim(0, 1500)
            ax.set_ylim(0.5, None)
        ax.set_xlabel("Energy (keV)")
        if i % 3 == 0:
            ax.set_ylabel("Counts per bin")
        ax.legend(loc="upper right", fontsize=7.6, handlelength=1.3, borderpad=0.3, labelspacing=0.3)
        label(ax, f"({'abcdef'[i]})", x=0.04, y=0.05, va="bottom")
    win = (x > 325) & (x < 365)
    A[2].set_ylim(0, 1.6 * base[win].max())
    A[2].set_ylabel("Counts per bin")
    A[5].set_ylim(1e-9, 1); A[5].set_ylabel("Normalized counts (offset)")
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, "Fig5_augmentation_examples")


# ---------------------------------------------------------------- Fig. 6
def fig6():
    df = pd.read_csv(RES / "dsp" / "classification_results.csv")
    cov = df[df.has_reference == True].copy()  # noqa: E712
    cov["outcome"] = np.where(cov.predicted_confident.isna(), "Abstained",
                              np.where(cov.correct_confident.astype(str) == "True", "Correct", "Wrong"))
    rows = {"Correct": 2, "Wrong": 1, "Abstained": 0}
    sty = {"Correct": ("black", "o"), "Wrong": (ORANGE, "X"), "Abstained": (GREY, "s")}
    fig, ax = plt.subplots(figsize=(W2, 2.85))
    rng = np.random.default_rng(1)
    for name, y0 in rows.items():
        sub = cov[cov.outcome == name]
        col, mk = sty[name]
        jit = rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(sub.n_reference_peaks_for_true_isotope, y0 + jit, s=30, color=col, marker=mk,
                   edgecolor="black", linewidth=0.4, zorder=3, label=f"{name} (n = {len(sub)})")
        if name != "Correct":
            for k, ((_, r), j) in enumerate(zip(sub.iterrows(), jit)):
                dy = 3 if name == "Wrong" else (8 if k == 0 else -10)
                ax.annotate(r.true_isotope, (r.n_reference_peaks_for_true_isotope, y0 + j),
                            xytext=(5, dy), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20, 40]); ax.set_xticklabels([1, 2, 3, 5, 10, 20, 40])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Abstained", "Wrong", "Correct"])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_ylim(-0.6, 2.6); ax.set_xlim(0.8, 50)
    ax.set_xlabel("Number of reference lines of the true isotope, $r_k$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    save(fig, "Fig6_reference_lines")


# ---------------------------------------------------------------- Fig. 7
def fig7():
    ms = json.loads((RES / "ml" / "multiseed_results.json").read_text())
    p4 = json.loads((RES / "ml" / "ml_summary.json").read_text())
    ctl = json.loads((RES / "ml" / "ablation_control.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    groups = [("rf_native", "RF\nnative", BLUE, "s"), ("rf_degraded", "RF\nNaI(Tl)-\nlike", BLUE, "s"),
              ("cnn_native", "CNN\nnative", GREEN, "o"), ("cnn_degraded", "CNN\nNaI(Tl)-\nlike", GREEN, "o")]
    for i, (k, lab, col, mk) in enumerate(groups):
        v = np.array(ms[k]["values"]) * 100
        xs = i + np.linspace(-0.14, 0.14, len(v))
        a1.plot(xs, v, mk, color=col, mec="black", mew=0.4, ms=5, mfc=col if "native" in lab else "white")
        a1.errorbar(i + 0.32, v.mean(), yerr=v.std(), fmt="_", color="black", ms=9, capsize=2.5, lw=0.9)
        if v.std() == 0:
            a1.text(i, v.mean() + 0.8, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=8)
        else:
            a1.text(i + 0.42, v.mean(), f"{v.mean():.1f}", va="center", fontsize=8)
    a1.set_xticks(range(4)); a1.set_xticklabels([g[1] for g in groups])
    a1.tick_params(axis="x", which="minor", bottom=False, top=False)
    a1.set_xlim(-0.5, 3.95); a1.set_ylim(76, 103.5)
    a1.set_ylabel("Accuracy on 36 original spectra (%)"); label(a1, "(a)", y=0.05, va="bottom")
    rnd = ctl["C_random_bins_blanked"]
    vals = {
        "Complete spectrum": ([p4["rf_heldout_original_accuracy_native_hpge"], p4["cnn_heldout_original_accuracy_native_hpge"]], None),
        "256 random bins removed": ([rnd["rf_mean"], rnd["cnn_mean"]], [rnd["rf_std"], rnd["cnn_std"]]),
        "Peak regions removed": ([ctl["A_peak_regions_blanked"]["rf"], ctl["A_peak_regions_blanked"]["cnn"]], None),
        "Only peak regions kept": ([ctl["B_only_peak_regions_kept"]["rf"], ctl["B_only_peak_regions_kept"]["cnn"]], None),
    }
    hatches = [None, "....", "////", "xxxx"]
    xx, w = np.arange(2), 0.2
    for j, ((lab, (v, err)), hatch) in enumerate(zip(vals.items(), hatches)):
        v = np.array(v) * 100
        pos = xx + (j - 1.5) * w
        if j == 0:
            a2.bar(pos, v, w, color=[BLUE, GREEN], edgecolor="black", lw=0.7, label=lab)
        else:
            a2.bar(pos, v, w, color="white", edgecolor="black", hatch=hatch, lw=0.7, label=lab,
                   yerr=None if err is None else np.array(err) * 100, capsize=2,
                   error_kw={"lw": 0.8})
        errs = [0, 0] if err is None else list(np.array(err) * 100)
        for xi, vi, ei in zip(pos, v, errs):
            a2.text(xi, vi + ei + 2.5, f"{vi:.1f}", ha="center", fontsize=7.0)
    a2.axhline(100 / 36, color=GREY, lw=0.9, ls=(0, (4, 2)))
    a2.set_xticks(xx); a2.set_xticklabels(["Random Forest", "CNN"])
    a2.tick_params(axis="x", which="minor", bottom=False, top=False)
    a2.set_xlim(-0.55, 1.65); a2.set_ylim(0, 175); a2.set_yticks(range(0, 101, 20))
    a2.set_ylabel("Accuracy (%)")
    leg = [Patch(fc="#9DB3E8", ec="black", label="Complete spectrum")] + \
          [Patch(fc="white", ec="black", hatch=h, label=l) for h, l in zip(hatches[1:], list(vals)[1:])] + \
          [plt.Line2D([], [], color=GREY, lw=0.9, ls=(0, (4, 2)), label="Chance level (1/36)")]
    a2.legend(handles=leg, loc="upper center", ncol=2, fontsize=7.6, columnspacing=0.8, handlelength=1.4)
    label(a2, "(b)", x=0.97, y=0.62, ha="right")
    fig.tight_layout()
    save(fig, "Fig7_multiseed_ablation")


# ---------------------------------------------------------------- Fig. 8
def fig8():
    d = np.load(RES / "comparison" / "coverage_accuracy_curves.npz")
    s = json.loads((RES / "comparison" / "comparison_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ser = [("DSP_(native,_raw_score,_n>=2_fixed)", "DSP, raw score (no gate)", RED, "-", "^", 1.2),
           ("RF_(native)", "RF, native", BLUE, "-", "s", 2.6),
           ("RF_(degraded_NaI)", "RF, NaI(Tl)-like", BLUE, (0, (4, 2)), "s", 1.2),
           ("CNN_(native)", "CNN, native", GREEN, "-", "o", 1.5),
           ("CNN_(degraded_NaI)", "CNN, NaI(Tl)-like", GREEN, (0, (4, 2)), "o", 1.2)]
    for k, lab, col, ls, mk, lw in ser:
        cv, ac = d[f"{k}__cov"] * 100, d[f"{k}__acc"] * 100
        ok = ~np.isnan(ac)
        ax.plot(cv[ok], ac[ok], color=col, ls=ls, lw=lw, marker=mk, markevery=10, ms=4,
                mfc="white" if "NaI" in lab else col, label=lab)
    g = s["dsp_gate_operating_point"]
    ax.plot([g["coverage"] * 100], [g["accuracy"] * 100], "*", ms=11, color=RED, mec="black", mew=0.6,
            ls="", label="DSP with gate")
    ax.axvline(g["coverage"] * 100, color=GREY, lw=0.8, ls=(0, (2, 2)), label="DSP coverage (92.6%)")
    ax.set_xlim(0, 102); ax.set_ylim(55, 103)
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower left", fontsize=7.6, labelspacing=0.3)
    save(fig, "Fig8_coverage_accuracy")


if __name__ == "__main__":
    fig1(); fig2(); top = fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("Ba-133 top candidates:\n", top.drop(columns=["matched_energies"]).to_string(index=False))
    print("written:", sorted(p.name for p in OUT.glob("Fig*.png")))
