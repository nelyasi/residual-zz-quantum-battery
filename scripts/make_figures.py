"""Figures 4-7 of the manuscript as vector PDFs.

The figures use LaTeX text rendering (Times text, Computer Modern mathematics) and
are exactly \\textwidth wide for the REVTeX two-column layout (508.1 pt), so they are
included at natural size, e.g. ``\\includegraphics{figures/fig4_energy_work.pdf}``.

Inputs:
    results/energy-budget.csv                        Fig. 4
    data/processed/fig5a-populations-by-outcome.csv  Fig. 5(a)
    data/processed/fig5b-damping-bound.csv           Fig. 5(b)
    data/processed/fig6-ramsey-check.csv             Fig. 6
    results/branch-populations.json                  Fig. 7

Usage: python scripts/make_figures.py [OUTPUT_DIR]      (default: figures/)
Requires a LaTeX installation with the psnfss (Times) and cm-super fonts.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update(
    {
        "text.usetex": True,
        # Times text (\fontfamily{ptm}), Computer Modern math, as in the pgfplots figures
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}\renewcommand{\rmdefault}{ptm}"
        r"\newcommand{\id}{\mathbb{I}}",
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 0.5,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.25,
        "ytick.major.size": 4.25,  # pgfplots: 0.15 cm
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.pad": 2.3,
        "ytick.major.pad": 3.3,
        "lines.scale_dashes": False,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": (0.7, 0.7, 0.7),  # black!30
        "legend.borderpad": 0.44,
        "legend.labelspacing": 0.2,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.5,
        "pdf.fonttype": 42,
        "savefig.transparent": False,
    }
)

BLACK = "#000000"
RED = "#A5282E"  # feedbackred / branchred
GRAY = "#666C73"  # storedgray
CHECK_RED = "#A83135"  # fbcheckred / ramseyred
PURPLE = "#63388B"
GREEN = "#287048"
DOTGRAY = (0.35, 0.35, 0.35)  # black!65
DOTGRAY55 = (0.45, 0.45, 0.45)  # black!55

# pgfplots dash patterns (points)
DASHED = (0, (3, 3))
DENSELY_DASHED = (0, (3, 2))


def DENSELY_DOTTED(lw):
    return (0, (lw, 1))


TEXTWIDTH = 508.1  # pt, x = 54 ... 562.1 on the page
PAGE_X0 = 54.0


class Canvas:
    """Figure of width \\textwidth whose coordinates follow the typeset page (pt, y down)."""

    def __init__(self, top, bottom):
        self.top, self.bottom = top, bottom
        self.H = bottom - top
        self.fig = plt.figure(figsize=(TEXTWIDTH / 72, self.H / 72))

    def axes(self, x, y, w, h):
        return self.fig.add_axes([(x - PAGE_X0) / TEXTWIDTH, (self.bottom - y - h) / self.H, w / TEXTWIDTH, h / self.H])

    def xy(self, x, y):  # page point -> figure fraction
        return (x - PAGE_X0) / TEXTWIDTH, (self.bottom - y) / self.H

    def save(self, name):
        path = OUT / name
        self.fig.savefig(path)
        plt.close(self.fig)
        print("wrote", path)


def fmt(v):
    """pgfplots /pgf/number format/fixed: 0.10 -> 0.1, 1.0 -> 1, -0.05 -> -0.05."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "$0$" if s in ("-0", "0") else f"${s}$"


def ticks(ax, xt=None, yt=None):
    if xt is not None:
        ax.set_xticks(xt, [fmt(v) for v in xt])
    if yt is not None:
        ax.set_yticks(yt, [fmt(v) for v in yt])
        shift = matplotlib.transforms.ScaledTranslation(0, 0.75 / 72, ax.figure.dpi_scale_trans)
        for lab in ax.get_yticklabels():
            lab.set_transform(lab.get_transform() + shift)


def band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0, zorder=1)


def labels(ax, title, xlabel=None, ylabel=None, title_pad=12.2, xpad=4.0, ypad=4.0):
    ax.set_title(title, pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=xpad)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=ypad)


def legend_at(canvas, ax, handles, texts, loc, x, y, **kw):
    """Legend whose `loc` corner sits at page point (x, y), as in the typeset original."""
    leg = ax.legend(
        handles,
        texts,
        loc=loc,
        bbox_to_anchor=canvas.xy(x, y),
        bbox_transform=canvas.fig.transFigure,
        borderaxespad=0,
        **kw,
    )
    leg.get_frame().set_linewidth(0.4)
    return leg


# ============================================================================ Fig. 4
def fig4():
    rows = list(csv.DictReader(open(RESULTS / "energy-budget.csv")))

    def col(k):
        return np.array([float(r[k]) for r in rows])

    N = col("cycles")

    c = Canvas(52.4, 282.6)
    axa = c.axes(120.8, 74.7, 181.3, 128.1)
    axb = c.axes(348.8, 74.7, 181.3, 128.1)

    thick, dlw = 0.8, 0.85
    # (a) energies before and after extraction
    for key, colr in (("before_energy", GRAY), ("unconditional_energy", BLACK), ("conditional_energy", RED)):
        m, s = col(key + "_measured"), col(key + "_SD")
        band(axa, N, m - s, m + s, colr)
    h = {}
    (h["pre"],) = axa.plot(
        N, col("before_energy_measured"), color=GRAY, lw=thick, marker="o", ms=4.8, mfc="none", mew=thick, zorder=4
    )
    (h["cal"],) = axa.plot(N, col("before_energy_model"), color=BLACK, lw=dlw, ls=DASHED, zorder=3)
    (h["U"],) = axa.plot(
        N, col("unconditional_energy_measured"), color=BLACK, lw=thick, marker="s", ms=4.4, mew=thick, zorder=4
    )
    (h["Cpas"],) = axa.plot(N, col("conditional_energy_model"), color=RED, lw=dlw, ls=DASHED, zorder=3)
    (h["C"],) = axa.plot(
        N, col("conditional_energy_measured"), color=RED, lw=thick, marker="^", ms=4.6, mew=thick, zorder=4
    )
    axa.set(xlim=(0.25, 20.75), ylim=(0, 0.50))
    ticks(axa, [1, 4, 8, 12, 16, 20], [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    labels(axa, "(a) Before and after extraction", r"Number of cycles $N$", r"Battery energy / $\hbar\omega$", ypad=7.1)
    # pgfplots fills the legend row by row; matplotlib column by column
    order = ["pre", "U", "C", "cal", "Cpas"]
    text = {
        "pre": r"$E_{\rm pre}^{\rm exp}$",
        "cal": r"$E_{\rm pre}^{\rm cal}=E_{\rm U}^{\rm pas}$",
        "U": r"$E_{\rm U}^{\rm exp}$",
        "Cpas": r"$E_{\rm C}^{\rm pas}$",
        "C": r"$E_{\rm C}^{\rm exp}$",
    }
    axa.legend(
        [h[k] for k in order],
        [text[k] for k in order],
        ncol=2,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        borderaxespad=0,
        handlelength=1.89,
        columnspacing=0.75,
    ).get_frame().set_linewidth(0.4)

    # (b) work
    for key, colr in (("unconditional_work", BLACK), ("conditional_work", RED)):
        m, s = col(key + "_measured"), col(key + "_SD")
        band(axb, N, m - s, m + s, colr)
    g = {}
    (g["Ue"],) = axb.plot(
        N, col("unconditional_work_measured"), color=BLACK, lw=thick, marker="s", ms=4.4, mew=thick, zorder=4
    )
    (g["Ui"],) = axb.plot(N, col("unconditional_work_model"), color=BLACK, lw=dlw, ls=DASHED, zorder=3)
    (g["Ce"],) = axb.plot(
        N, col("conditional_work_measured"), color=RED, lw=thick, marker="^", ms=4.6, mew=thick, zorder=4
    )
    (g["Ci"],) = axb.plot(N, col("conditional_work_model"), color=RED, lw=dlw, ls=DASHED, zorder=3)
    axb.set(xlim=(0.25, 20.75), ylim=(-0.055, 0.252))
    ticks(axb, [1, 4, 8, 12, 16, 20], [-0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25])
    labels(axb, "(b) Work extraction", r"Number of cycles $N$", r"Work / $\hbar\omega$", ypad=4.7)
    order = ["Ue", "Ce", "Ui", "Ci"]
    text = {
        "Ue": r"$W_{\rm U}^{\rm exp}$",
        "Ui": r"$\mathcal{W}_{\rm U}^{\rm id}$",
        "Ce": r"$W_{\rm C}^{\rm exp}$",
        "Ci": r"$\mathcal{W}_{\rm C}^{\rm id}$",
    }
    axb.legend(
        [g[k] for k in order],
        [text[k] for k in order],
        ncol=2,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        borderaxespad=0,
        handlelength=1.89,
        columnspacing=0.75,
    ).get_frame().set_linewidth(0.4)
    c.save("fig4_energy_work.pdf")


# ============================================================================ Fig. 5
def errorbars(ax, x, y, xe, ye, color, marker, ms, label=None):
    return ax.errorbar(
        x,
        y,
        yerr=ye,
        xerr=xe,
        fmt=marker,
        color=color,
        ms=ms,
        mew=0.75,
        elinewidth=0.75,
        capsize=1.7,
        capthick=0.65,
        zorder=4,
        label=label,
    )


def fig5():
    with open(PROCESSED / "fig5a-populations-by-outcome.csv") as stream:
        outcomes = list(csv.DictReader(stream))
    with open(PROCESSED / "fig5b-damping-bound.csv") as stream:
        dm = {key: float(value) for key, value in next(csv.DictReader(stream)).items()}
    groups = {}
    for gate in ("I", "X"):
        chosen = [row for row in outcomes if row["applied_gate"] == gate]
        groups[gate] = {
            key: [float(row[column]) for row in chosen]
            for key, column in (
                ("x", "reference_excitation"),
                ("xe", "reference_se"),
                ("y", "feedback_excitation"),
                ("ye", "feedback_se"),
            )
        }
    c = Canvas(52.4, 214.6)
    axa = c.axes(119.1, 75.1, 181.3, 111.1)
    axb = c.axes(354.2, 75.1, 181.3, 111.1)

    axa.plot([0, 1], [0, 1], color=DOTGRAY, lw=0.75, ls=DENSELY_DOTTED(0.75), zorder=2)
    axa.plot(
        [0, 1], [1, 0], color=matplotlib.colors.to_rgb(CHECK_RED) + (0.75,), lw=0.75, ls=DENSELY_DOTTED(0.75), zorder=2
    )
    i, x = groups["I"], groups["X"]
    errorbars(axa, i["x"], i["y"], i["xe"], i["ye"], BLACK, "o", 4.2)
    errorbars(axa, x["x"], x["y"], x["xe"], x["ye"], CHECK_RED, "s", 4.2)
    hi = Line2D([], [], color=BLACK, marker="o", ms=4.2, ls="none")
    hx = Line2D([], [], color=CHECK_RED, marker="s", ms=4.2, ls="none")
    axa.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04))
    ticks(axa, [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
    labels(axa, "(a) Population by outcome", "Reference excitation", "Feedback excitation", ypad=4.7)
    legend_at(
        c,
        axa,
        [hi, hx],
        ["Identity", r"Pauli $X$"],
        "upper right",
        294.7,
        78.7,
        fontsize=10,
        handlelength=0.3,
        handletextpad=0.18,
        labelspacing=0.3,
    )

    t = np.linspace(0, 375, 90)
    (l1,) = axb.plot(t, dm["control_population_20us"] * (1 - np.exp(-t / dm["battery_T1_us"])), color=PURPLE, lw=0.75)
    (l2,) = axb.plot([0, 375], [dm["measured_energy_difference"]] * 2, color=BLACK, lw=0.75, ls=DENSELY_DASHED)
    axb.plot(
        [dm["minimum_extra_wait_us"]] * 2,
        [0, dm["measured_energy_difference"]],
        color=DOTGRAY,
        lw=0.75,
        ls=DENSELY_DOTTED(0.75),
    )
    axb.plot([dm["minimum_extra_wait_us"]], [dm["measured_energy_difference"]], "o", color=BLACK, ms=3.7, zorder=5)
    axb.annotate(
        r"$324\,\mu$s",
        (dm["minimum_extra_wait_us"], dm["measured_energy_difference"] + 0.012),
        xytext=(-3.3, 3.6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
    )
    axb.set(xlim=(0, 375), ylim=(0, 0.435))
    ticks(axb, [0, 100, 200, 300], [0, 0.1, 0.2, 0.3, 0.4])
    labels(
        axb,
        "(b) Bound from amplitude damping",
        r"Maximum extra wait $\delta$ ($\mu$s)",
        r"Energy decrease / $\hbar\omega$",
        ypad=7.1,
    )
    legend_at(
        c,
        axb,
        [l1, l2],
        ["Maximum damping loss", "Measured energy difference"],
        "lower right",
        529.8,
        182.7,
        fontsize=10,
        handlelength=1.47,
        labelspacing=0.3,
    )
    c.save("fig5_feedback_check.pdf")


# ============================================================================ Fig. 6
def fig6():
    with open(PROCESSED / "fig6-ramsey-check.csv") as stream:
        table = list(csv.DictReader(stream))
    tau = np.array(sorted({float(row["nominal_delay_us"]) for row in table}))
    d = {}
    for q in ("Q76", "Q85", "Q87", "Q96"):
        rows = [row for row in table if row["spectator"] == q]
        fitted = [row for row in rows if row["differential_phase_rad"]]
        d[q] = {
            "phase": [float(row["differential_phase_rad"]) for row in fitted],
            "err": [float(row["differential_phase_se_rad"]) for row in fitted],
            "fit": [float(row["weighted_fit_rad"]) for row in fitted],
            "vis": [float(row["smaller_ramsey_amplitude"]) for row in rows],
        }
    c = Canvas(246.0, 408.1)
    axa = c.axes(117.8, 268.7, 181.3, 111.1)
    axb = c.axes(352.9, 268.7, 181.3, 111.1)
    style = {
        "Q76": (BLACK, "o", 4.0),
        "Q85": (CHECK_RED, "s", 4.0),
        "Q87": (PURPLE, "d", 3.8),
        "Q96": (GREEN, "^", 4.2),
    }
    handles = []
    for q, (colr, mk, ms) in style.items():
        n = len(d[q]["phase"])
        axa.errorbar(
            tau[:n],
            d[q]["phase"],
            yerr=d[q]["err"],
            fmt=mk,
            color=colr,
            ms=ms,
            mew=0.75,
            elinewidth=0.75,
            capsize=2.2,
            capthick=0.65,
            zorder=4,
        )
        axa.plot(tau[:n], d[q]["fit"], color=colr, lw=0.75, ls=DENSELY_DASHED if q == "Q87" else "-", zorder=3)
        handles.append(Line2D([], [], color=colr, marker=mk, ms=ms, ls="none"))
        axb.plot(tau, d[q]["vis"], color=colr, lw=0.75, ls=DASHED, marker=mk, ms=ms, mew=0.75, zorder=4)
    axa.set(xlim=(-2, 46), ylim=(-4.7, 0.65))
    ticks(axa, [0, 10, 20, 30, 40], [-4, -3, -2, -1, 0])
    labels(
        axa,
        "(a) Conditional phase shift",
        r"Nominal sensing time $\tau$ ($\mu$s)",
        "Differential phase (rad)",
        ypad=7.1,
    )
    legend_at(
        c,
        axa,
        handles,
        list(style),
        "upper right",
        293.4,
        272.3,
        fontsize=10,
        handlelength=0.3,
        handletextpad=0.16,
        labelspacing=0.3,
    )

    axb.plot([-2, 46], [0.1, 0.1], color=DOTGRAY55, lw=0.75, ls=DENSELY_DOTTED(0.75), zorder=2)
    axb.annotate(
        r"$0.10$ cutoff",
        (45, 0.107),
        xytext=(-1.2, 0.2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    )
    axb.set(xlim=(-2, 46), ylim=(-0.035, 1.09))
    ticks(axb, [0, 10, 20, 30, 40], [0, 0.2, 0.4, 0.6, 0.8, 1])
    labels(axb, "(b) Ramsey visibility", r"Nominal sensing time $\tau$ ($\mu$s)", "Smaller Ramsey amplitude", ypad=7.1)
    c.save("fig6_ramsey_check.pdf")


# ============================================================================ Fig. 7
def fig7():
    report = json.load(open(RESULTS / "branch-populations.json"))
    N = np.array([row["cycles"] for row in report["rows"]])
    d = {}
    for operation in ("X", "I"):
        series = {}
        for arm, key in (("unconditional", "fixed"), ("conditional", "cond")):
            mean = np.array([row[operation][arm]["mean"] for row in report["rows"]])
            sd = np.array([row[operation][arm]["SD"] for row in report["rows"]])
            series.update({key: mean, f"{key}_lo": mean - sd, f"{key}_hi": mean + sd})
        d[operation] = series
    c = Canvas(49.9, 237.4)
    axa = c.axes(125.6, 75.2, 178.7, 115.3)
    axb = c.axes(346.8, 75.2, 178.7, 115.3)
    for ax, key in ((axa, "X"), (axb, "I")):
        s = d[key]
        band(ax, N, s["fixed_lo"], s["fixed_hi"], BLACK)
        band(ax, N, s["cond_lo"], s["cond_hi"], RED)
        (hf,) = ax.plot(N, s["fixed"], color=BLACK, lw=0.8, marker="s", ms=4.2, mew=0.8, zorder=4)
        (hc,) = ax.plot(N, s["cond"], color=RED, lw=0.8, marker="^", ms=4.4, mew=0.8, zorder=4)
        ax.set_xlim(0.2, 20.8)
    axa.set_ylim(0.13, 0.90)
    ticks(axa, [1, 4, 8, 12, 16, 20], [0.2, 0.4, 0.6, 0.8])
    axb.set_ylim(0.03, 0.26)
    ticks(axb, [1, 4, 8, 12, 16, 20], [0.1, 0.2])
    labels(axa, r"(a) Records selecting $X$", r"Number of cycles $N$", "Final excited-state population", ypad=7.1)
    labels(axb, r"(b) Records selecting $\id$", r"Number of cycles $N$")
    legend_at(
        c,
        axa,
        [hf, hc],
        ["Fixed operation", "Conditional operation"],
        "upper center",
        331.1,
        218.4,
        ncol=2,
        fontsize=9,
        handlelength=1.59,
        columnspacing=0.33,
        borderpad=0.51,
    )
    c.save("fig7_branch_populations.pdf")


if __name__ == "__main__":
    fig4()
    fig5()
    fig6()
    fig7()
