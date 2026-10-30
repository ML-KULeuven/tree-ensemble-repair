#!/usr/bin/env python3
"""
Runtime analysis figure: three subplots.
  1. MILP runtime vs dataset size (one point per dataset, error bars across folds).
  2. MILP runtime vs initial #unfair boxes (one point per fold).
  3. MILP runtime vs model size in #leaves of the compressed model (one point per fold).

Only folds where the MILP repair converged (status optimal/suboptimal) are plotted.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from plot_style import (C_DOT, C_MEAN, C_SUBTITLE, SUBTITLE_SIZE,
                        save, set_style, style_axes)
from utils import CACHE_DIR

# ── Config ────────────────────────────────────────────────────────────────────

# per fold: {ds}_{model}_{fold}_lop.json  (#unfair boxes, #leaves of the compressed model)
#           {ds}_{model}_{fold}_milp.json (MILP runtime)
RESULTS_DIR = f"{CACHE_DIR}/results"
OUT_DIR     = "figures"
MODEL_TYPE  = "xgb"
RESULT_KEY  = "composite_repaired_at"

# From the paper's Table 1
DATASET_SIZES = {
    "ACSIncome":              195665,
    "ACSEmployment":          378817,
    "ACSMobility":             80329,
    "ACSPublicCoverage":      138554,
    "ACSTravelTime":          172508,
    "BankMarketingFull":       11162,
    "CompasTwoYears":           4966,
    "DefaultCreditCardClients": 13272,
    "DutchCensus":             60420,
    "LawSchool":               18692,
    "MEPSPanel21":             15675,
}

DATASET_SHORT = {
    "ACSIncome":               "ACSIncome",
    "ACSEmployment":           "ACSEmploy",
    # "ACSMobility":             "ACSMobility",
    "ACSPublicCoverage":       "ACSPublic",
    "ACSTravelTime":           "ACSTravel",
    "BankMarketingFull":       "Bank",
    "CompasTwoYears":          "Compas",
    "DefaultCreditCardClients": "Credit",
    "DutchCensus":             "Dutch",
    "LawSchool":               "LawSchool",
    "MEPSPanel21":             "MEPS",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(directory, model_type):
    pattern = re.compile(r"^(.+)_" + re.escape(model_type) + r"_(\d+)_milp\.json$")
    records = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("._"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        dname, fold = m.group(1), int(m.group(2))
        if dname not in DATASET_SIZES:
            continue
        lop_path = os.path.join(directory, f"{dname}_{model_type}_{fold}_lop.json")
        if not os.path.exists(lop_path):
            print(f"  WARNING: {dname} fold {fold}: no lop results -- dropped.")
            continue
        with open(os.path.join(directory, fname)) as f:
            repaired = json.load(f).get(RESULT_KEY, {})
        with open(lop_path) as f:
            compressed = json.load(f).get("compressed_at", {})
        unfair     = compressed.get("#unfair_boxes")
        milp_time  = repaired.get("milp_time")
        num_leaves = compressed.get("num_leaves")
        if unfair is None or milp_time is None or num_leaves is None:
            continue
        status = repaired.get("status")
        if status in ("fair", "not_enumerable"):
            continue      # no MILP work done
        if status not in ("optimal", "suboptimal"):
            print(f"  WARNING: {dname} fold {fold}: MILP did not converge "
                  f"(status={status}) -- dropped.")
            continue
        records.append({
            "dataset":      dname,
            "fold":         fold,
            "n_samples":    DATASET_SIZES[dname],
            "unfair_boxes": unfair,
            "milp_time":    milp_time,
            "num_leaves":   num_leaves,
        })
    return records


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fmt_k(n):
    return f"{n//1000}k" if n >= 1000 else str(n)


def _spearman(records, xkey, ykey):
    """Spearman rank correlation between two record fields, as a caption string.

    Rank-based rather than Pearson: both runtime and #unfair boxes span several
    orders of magnitude, so a linear correlation would just track the extremes.
    """
    x = [r[xkey] for r in records]
    y = [r[ykey] for r in records]
    rho, _ = spearmanr(x, y)
    return f"Spearman $\\rho$ = {rho:+.2f}"


def _title(ax, main, sub):
    """Bold title with a smaller grey caption (Spearman rho) underneath."""
    ax.set_title(main, pad=28)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=SUBTITLE_SIZE, color=C_SUBTITLE)


def _save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in save(fig, f"{OUT_DIR}/{stem}"):
        print(f"Saved {path}")


def _panel_runtime_vs_unfair(ax, records, size=35):
    """MILP runtime vs initial #unfair boxes (one point per fold)."""
    for r in records:
        ax.scatter(r["unfair_boxes"], r["milp_time"],
                   color=C_DOT, alpha=0.6, s=size, zorder=3)

    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_k(int(x))))
    ax.set_xlabel("#counterexamples", labelpad=12)
    ax.set_ylabel("MILP runtime (s)", labelpad=12)
    _title(ax, "MILP runtime vs. #counterexamples",
           _spearman(records, "unfair_boxes", "milp_time"))
    style_axes(ax)


def _plot_runtime_vs(records, xkey, xlabel, stem, fmt_k_xaxis=False):
    """Standalone scatter of MILP runtime against one record field.

    Same canvas as the generalization figures, so they sit together in the paper.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    for r in records:
        ax.scatter(r[xkey], r["milp_time"], color=C_DOT, alpha=0.6, s=90, zorder=3)

    ax.set_yscale("log")
    if fmt_k_xaxis:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_k(int(x))))
    # labelpad: the big tick labels crowd the axis titles
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel("MILP runtime (s)", labelpad=12)
    _title(ax, f"MILP runtime vs. {xlabel.split(' (')[0]}",
           _spearman(records, xkey, "milp_time"))
    style_axes(ax)

    fig.tight_layout()
    _save(fig, stem)


def plot_runtime_vs_unfair(records):
    _plot_runtime_vs(records, "unfair_boxes", "#counterexamples",
                     "runtime_vs_unfair_boxes", fmt_k_xaxis=True)


def plot_runtime_vs_numleaves(records):
    _plot_runtime_vs(records, "num_leaves", "#leaves (compressed model)",
                     "runtime_vs_numleaves")


def plot(records):
    set_style()

    # datasets ordered by size (small → large)
    datasets = sorted(DATASET_SIZES.keys(), key=lambda d: DATASET_SIZES[d])
    # only keep datasets present in records
    present  = {r["dataset"] for r in records}
    datasets = [d for d in datasets if d in present]
    pos_of   = {d: i for i, d in enumerate(datasets)}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    # ── Subplot 1: categorical x ordered by dataset size ─────────────────────
    by_dataset = {}
    for r in records:
        by_dataset.setdefault(r["dataset"], []).append(r["milp_time"])

    rng = np.random.default_rng(0)
    for r in records:
        jitter = rng.uniform(-0.25, 0.25)
        ax1.scatter(pos_of[r["dataset"]] + jitter, r["milp_time"],
                    color=C_DOT, alpha=0.4, s=22, zorder=2)

    for dname in datasets:
        times = by_dataset[dname]
        i   = pos_of[dname]
        mu  = np.mean(times)
        std = np.std(times, ddof=1) if len(times) > 1 else 0.0
        ax1.errorbar(i, mu, yerr=std, fmt="D", color=C_MEAN,
                     markersize=7, capsize=4, linewidth=1.5, zorder=3)

    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(
        [f"{DATASET_SHORT[d]} ({_fmt_k(DATASET_SIZES[d])})" for d in datasets],
        rotation=45, ha="right", rotation_mode="anchor",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset (ordered by # examples)", labelpad=12)
    ax1.set_ylabel("MILP runtime (s)", labelpad=12)
    _title(ax1, "MILP runtime vs. dataset size",
           _spearman(records, "n_samples", "milp_time"))
    style_axes(ax1, grid="y")

    # ── Subplot 2: runtime vs #unfair boxes ───────────────────────────────────
    _panel_runtime_vs_unfair(ax2, records)

    # ── Subplot 3: runtime vs model size (#leaves after compression) ──────────
    for r in records:
        ax3.scatter(r["num_leaves"], r["milp_time"],
                    color=C_DOT, alpha=0.6, s=35, zorder=3)

    ax3.set_yscale("log")
    ax3.set_xlabel("#leaves (compressed model)", labelpad=12)
    ax3.set_ylabel("MILP runtime (s)", labelpad=12)
    _title(ax3, "MILP runtime vs. model size",
           _spearman(records, "num_leaves", "milp_time"))
    style_axes(ax3)

    # ── Subplot 4: #unfair boxes vs model size ───────────────────────────────
    for r in records:
        ax4.scatter(r["num_leaves"], r["unfair_boxes"],
                    color=C_DOT, alpha=0.6, s=35, zorder=3)

    ax4.set_yscale("log")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: _fmt_k(int(y))))
    ax4.set_xlabel("#leaves (compressed model)", labelpad=12)
    ax4.set_ylabel("#unfair boxes", labelpad=12)
    _title(ax4, "#unfair boxes vs. model size",
           _spearman(records, "num_leaves", "unfair_boxes"))
    style_axes(ax4)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.30, hspace=0.55)
    _save(fig, "runtime_analysis")


if __name__ == "__main__":
    records = load_results(RESULTS_DIR, MODEL_TYPE)
    print(f"Loaded {len(records)} folds across "
          f"{len({r['dataset'] for r in records})} datasets")
    plot(records)
    plot_runtime_vs_unfair(records)
    plot_runtime_vs_numleaves(records)
