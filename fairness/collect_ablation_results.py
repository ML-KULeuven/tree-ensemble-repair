#!/usr/bin/env python3
"""
Summarize ablation study results and generate plots.

Ablation 1 (lambda sensitivity, --experiment 1):
  Double-y line plot per dataset + one averaged figure.
  Left y: Δ balanced accuracy vs original XGB model (pp).
  Right y: MILP runtime (s, log scale).

Ablation 2 (objective variants, --experiment 2):
  Side-by-side boxplots (bacc loss + runtime) across datasets × folds.

Usage:
    python collect_ablation_results.py [--abl 1] [--abl 2]

"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import (C_ACC, C_ACC_ONLY, C_BOX_LINE, C_L1, C_MILP, C_REF, C_RT,
                        save, set_style, style_axes)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

from utils import CACHE_DIR

ABLATION1_DIR = f"{CACHE_DIR}/results/ablation1"
ABLATION2_DIR = f"{CACHE_DIR}/results/ablation2"
OUT_DIR = "figures"

# not run (too long)
EXCLUDE_DATASETS = [
    "ACSPublicCoverage", 
    "ACSIncome", 
    "MEPSPanel21", 
    "LawSchool", 
]

# Short row labels for the figures, kept in sync with DATASET_LABELS in
# collect_results.py. Anything not listed here is shown as-is.
DATASET_LABELS = {
    "ACSEmployment": "ACSEmploy",
    "ACSPublicCoverage": "ACSPublic",
    "ACSTravelTime": "ACSTravel",
    "BankMarketingFull": "Bank",
    "CompasTwoYears": "Compas",
    "DefaultCreditCardClients": "Credit",
    "MEPSPanel21": "MEPS",
}

ABL1_VARIANTS = [
    (1e-5,   "composite_lam000001", r"$10^{-5}$"),
    (0.0001, "composite_lam00001", r"$10^{-4}$"),
    (0.001,  "composite_lam0001",  r"$10^{-3}$"),
    (0.01,   "composite_lam001",   r"$10^{-2}$"),
    (0.1,    "composite_lam01",    r"$10^{-1}$"),
    (1.0,    "composite_lam1",     r"$1$"),
    (10.0,   "composite_lam10",    r"$10$"),
]
ABL1_KEYS   = [v[1] for v in ABL1_VARIANTS]
ABL1_LABELS = [v[2] for v in ABL1_VARIANTS]

ABL2_METHODS = [
    ("composite",          "Composite"),
    ("composite_acc_only", "Acc. only"),
    ("l1",             "L1 only"),
]

# Ablation 1 is our MILP alone, so it colours the metric (plot_style.C_ACC /
# C_RT); ablation 2 compares variants, each in its own fixed colour.
ABL2_COLORS = {
    "composite":          C_MILP,
    "composite_acc_only": C_ACC_ONLY,
    "l1":                 C_L1,
}

# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dname_fold(path):
    m = re.match(r"^(.+)_[A-Za-z0-9]+_(\d+)\.json$", os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load_results(directory):
    results = {}
    if not os.path.isdir(directory):
        return results
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json") or fname.startswith("._"):
            continue
        dname, fold = _extract_dname_fold(fname)
        if dname is None or dname in EXCLUDE_DATASETS:
            continue
        with open(os.path.join(directory, fname)) as f:
            results[(dname, fold)] = json.load(f)
    return results


def _bacc_loss_pp(result, variant_key):
    """Δ test balanced accuracy (pp) = variant − original_xgb."""
    orig    = result.get("original_at", {}).get("test_bacc")
    variant = result.get(variant_key, {}).get("test_bacc")
    if orig is None or variant is None:
        return None
    return (variant - orig) * 100.0


def _runtime(result, variant_key):
    return result.get(variant_key, {}).get("milp_time")


def _converged(result, key, dname, fold):
    """True if the variant repaired the model to 0 unfair boxes; warn and return False otherwise."""
    v = result.get(key)
    if v is None:
        print(f"  WARNING: {dname} fold {fold}: variant '{key}' missing -- dropped.")
        return False
    status = v.get("status")
    if status not in ("optimal", "suboptimal") or v.get("#unfair_boxes") != 0:
        print(f"  WARNING: {dname} fold {fold}: variant '{key}' did not converge "
              f"(status={status}, #unfair_boxes={v.get('#unfair_boxes')}) -- dropped.")
        return False
    return True


def _already_fair(result):
    # nothing to repair: compressed model already fair, or its OC space is not enumerable
    compr = result.get("compressed_at", {})
    return compr.get("#unfair_boxes") == 0 or compr.get("enumerable") is False


# ──────────────────────────────────────────────────────────────────────────────
# Ablation 1
# ──────────────────────────────────────────────────────────────────────────────

def parse_ablation1(results):
    """Return {dname: {key: {bacc_loss: [...], runtime: [...]}}}."""
    data = defaultdict(lambda: {k: {"bacc_loss": [], "runtime": []} for k in ABL1_KEYS})
    for (dname, fold), result in results.items():
        if _already_fair(result):
            continue
        for _, key, _ in ABL1_VARIANTS:
            if not _converged(result, key, dname, fold):
                continue
            bl = _bacc_loss_pp(result, key)
            rt = _runtime(result, key)
            if bl is not None:
                data[dname][key]["bacc_loss"].append(bl)
            if rt is not None:
                data[dname][key]["runtime"].append(rt)
    return data


def _stats(vals):
    m = np.mean(vals) if vals else np.nan
    e = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return m, e


def _row_label(ax, dname):
    """Dataset name as a bold row label in the left margin of a grid row."""
    ax.annotate(DATASET_LABELS.get(dname, dname),
                xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-0.42, 0.5), textcoords="axes fraction",
                fontweight="bold", ha="right", va="center", annotation_clip=False)


def _save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    save(fig, f"{OUT_DIR}/{stem}")
    print(f"  Saved {OUT_DIR}/{stem}.{{pdf,png}}")


def _abl1_bacc_ax(ax, data, show_ylabel=True, xlabel=r"$\lambda$"):
    x = np.arange(len(ABL1_KEYS))
    bacc_m, bacc_e = [], []
    for key in ABL1_KEYS:
        m, e = _stats(data[key]["bacc_loss"])
        bacc_m.append(m); bacc_e.append(e)
    ax.errorbar(x, bacc_m, yerr=bacc_e,
                color=C_ACC, marker="o", linewidth=2.5, markersize=8,
                capsize=4, zorder=3)
    ax.axhline(0, color=C_REF, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ABL1_LABELS)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Δ balanced accuracy")
    style_axes(ax, grid="y")


def _fmt_log_ax(ax, n=3):
    """Force exactly n ticks on a log-scale axis at nice 1/2/5×decade positions."""
    ymin, ymax = ax.get_ylim()
    lo = int(np.floor(np.log10(ymin)))
    hi = int(np.ceil(np.log10(ymax)))
    # all 1×/2×/5× candidates inside the current axis limits
    cands = sorted({round(s * 10.0**d, 12)
                    for d in range(lo - 1, hi + 2)
                    for s in [1, 2, 5]
                    if ymin <= s * 10.0**d <= ymax})
    if len(cands) >= n:
        idx = np.round(np.linspace(0, len(cands) - 1, n)).astype(int)
        ticks = [cands[i] for i in idx]
    else:
        # range too narrow — fall back to n geometric ticks rounded to 1 sig fig
        def _r1(x):
            d = np.floor(np.log10(x))
            return float(f"{round(x / 10**d) * 10**d:.6g}")
        ticks = [_r1(v) for v in np.geomspace(ymin, ymax, n)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _abl1_rt_ax(ax, data, show_ylabel=True, xlabel=r"$\lambda$"):
    x = np.arange(len(ABL1_KEYS))
    rt_m, rt_e = [], []
    for key in ABL1_KEYS:
        m, e = _stats(data[key]["runtime"])
        rt_m.append(m); rt_e.append(e)
    rt_m, rt_e = np.array(rt_m), np.array(rt_e)
    rt_yerr = [np.minimum(rt_e, rt_m * 0.99), rt_e]
    ax.errorbar(x, rt_m, yerr=rt_yerr,
                color=C_RT, marker="s", linewidth=2.5, markersize=8,
                capsize=4, zorder=3)
    ax.set_yscale("log")
    _fmt_log_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(ABL1_LABELS)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("MILP Runtime (s)")
    style_axes(ax, grid="y")


def plot_ablation1(by_dataset, suffix=""):
    set_style()

    datasets = sorted(by_dataset.keys())
    xlabel = r"$\lambda$"

    # ── Aggregate data ────────────────────────────────────────────────────────
    avg_data = {k: {"bacc_loss": [], "runtime": []} for k in ABL1_KEYS}
    for dname in datasets:
        for k in ABL1_KEYS:
            avg_data[k]["bacc_loss"].extend(by_dataset[dname][k]["bacc_loss"])
            avg_data[k]["runtime"].extend(by_dataset[dname][k]["runtime"])

    # ── Average figure: 1 row × 2 subplots ───────────────────────────────────
    fig_avg, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    _abl1_bacc_ax(ax_b, avg_data, xlabel=xlabel)
    _abl1_rt_ax(ax_r, avg_data, xlabel=xlabel)
    ax_b.set_title("Δ balanced accuracy vs XGB")
    ax_r.set_title("MILP Runtime (s)")
    # fig_avg.suptitle("Average across all datasets", fontsize=12, y=1.03)
    fig_avg.tight_layout()
    fig_avg.subplots_adjust(wspace=0.45)

    # ── Per-dataset grid: N rows × 2 subplots ────────────────────────────────
    n = len(datasets)
    # extra left margin for row labels
    fig_grid, axes = plt.subplots(n, 2, figsize=(13.0, 4.5 * n), squeeze=False)

    for i, dname in enumerate(datasets):
        _abl1_bacc_ax(axes[i, 0], by_dataset[dname], show_ylabel=True, xlabel=xlabel)
        _abl1_rt_ax(axes[i, 1], by_dataset[dname], show_ylabel=True, xlabel=xlabel)
        # column titles on first row only
        if i == 0:
            axes[0, 0].set_title("Δ balanced accuracy vs XGB")
            axes[0, 1].set_title("MILP Runtime (s)")
        _row_label(axes[i, 0], dname)

    fig_grid.tight_layout()
    fig_grid.subplots_adjust(left=0.18, wspace=0.45)

    _save(fig_avg, f"ablation1_avg{suffix}")
    _save(fig_grid, f"ablation1_grid{suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# Ablation 2
# ──────────────────────────────────────────────────────────────────────────────

def parse_ablation2(results):
    """Return {dname: {key: {bacc_loss: [...], runtime: [...]}}}."""
    keys = [k for k, _ in ABL2_METHODS]
    data = defaultdict(lambda: {k: {"bacc_loss": [], "runtime": []} for k in keys})
    for (dname, fold), result in results.items():
        if _already_fair(result):
            continue
        for key, _ in ABL2_METHODS:
            if not _converged(result, key, dname, fold):
                continue
            bl = _bacc_loss_pp(result, key)
            rt = _runtime(result, key)
            if bl is not None:
                data[dname][key]["bacc_loss"].append(bl)
            if rt is not None:
                data[dname][key]["runtime"].append(rt)
    return data


def plot_ablation2(by_dataset, suffix=""):
    set_style()

    labels  = [lab for _, lab in ABL2_METHODS]
    keys    = [k   for k, _   in ABL2_METHODS]
    datasets = sorted(by_dataset.keys())
    palette  = [ABL2_COLORS[k] for k in keys]

    # ── Average figure: 1 row × 2 subplots (seaborn boxplot, main-exp style) ─
    bacc_per_method = {k: [] for k in keys}
    rt_per_method   = {k: [] for k in keys}
    for dname in datasets:
        for k in keys:
            bl = by_dataset[dname][k]["bacc_loss"]
            rt = by_dataset[dname][k]["runtime"]
            if bl:
                bacc_per_method[k].append(statistics.mean(bl))
            if rt:
                rt_per_method[k].append(statistics.mean(rt))

    fig_avg, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    bacc_df = pd.DataFrame({lab: pd.Series(bacc_per_method[k]) for k, lab in zip(keys, labels)})
    sns.boxplot(data=bacc_df, palette=palette, saturation=1, linecolor=C_BOX_LINE, ax=ax_b)
    ax_b.axhline(0, color=C_REF, linestyle="--", linewidth=0.8)
    ax_b.set_ylabel("Δ balanced accuracy")
    ax_b.set_title("Δ balanced accuracy vs XGB")
    style_axes(ax_b, grid="y")

    rt_df = pd.DataFrame({lab: pd.Series(rt_per_method[k]) for k, lab in zip(keys, labels)})
    sns.boxplot(data=rt_df, palette=palette, saturation=1, linecolor=C_BOX_LINE, ax=ax_r)
    ax_r.set_yscale("log")
    ax_r.set_ylabel("MILP Runtime (s)")
    ax_r.set_title("MILP Runtime (s)")
    style_axes(ax_r, grid="y")

    fig_avg.tight_layout()
    fig_avg.subplots_adjust(wspace=0.45)

    # ── Per-dataset grid: N rows × 2 subplots (mean ± std per variant) ───────
    # x is categorical (one position per variant), so the points are not joined
    # by a line; each takes its variant's colour.
    n = len(datasets)
    fig_grid, axes = plt.subplots(n, 2, figsize=(13.0, 4.5 * n), squeeze=False)

    for i, dname in enumerate(datasets):
        d = by_dataset[dname]
        for j, k in enumerate(keys):
            # left: Δ bacc
            m, e = _stats(d[k]["bacc_loss"])
            if not np.isnan(m):
                axes[i, 0].errorbar(j, m, yerr=e, color=ABL2_COLORS[k], marker="o",
                                    markersize=8, capsize=4, linewidth=2.5, zorder=3)
            # right: runtime (log); clamp the lower error so the bar never
            # reaches <= 0 (a log scale cannot show that)
            m, e = _stats(d[k]["runtime"])
            if not np.isnan(m):
                axes[i, 1].errorbar(j, m, yerr=[[min(e, m * 0.99)], [e]],
                                    color=ABL2_COLORS[k], marker="s",
                                    markersize=8, capsize=4, linewidth=2.5, zorder=3)

        axes[i, 0].axhline(0, color=C_REF, linestyle="--", linewidth=0.8, zorder=1)
        axes[i, 0].set_ylabel("Δ balanced accuracy")
        axes[i, 1].set_yscale("log")
        _fmt_log_ax(axes[i, 1])
        axes[i, 1].set_ylabel("MILP Runtime (s)")
        for ax in axes[i]:
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.5, len(keys) - 0.5)
            style_axes(ax, grid="y")
        if i == 0:
            axes[0, 0].set_title("Δ balanced accuracy vs XGB")
            axes[0, 1].set_title("MILP Runtime (s)")
        _row_label(axes[i, 0], dname)

    fig_grid.tight_layout()
    fig_grid.subplots_adjust(left=0.18, wspace=0.45)

    _save(fig_avg, f"ablation2_avg{suffix}")
    _save(fig_grid, f"ablation2_grid{suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(results):
    """Print which datasets are used and how many folds per dataset."""
    folds_by_dataset = defaultdict(set)
    for (dname, fold), _ in results.items():
        folds_by_dataset[dname].add(fold)

    print("  Datasets and folds:")
    for dname in sorted(folds_by_dataset.keys()):
        fold_count = len(folds_by_dataset[dname])
        fold_list = sorted(folds_by_dataset[dname])
        print(f"    {dname}: {fold_count} folds {fold_list}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--abl", type=int, choices=[1, 2], action="append",
                   help="Which ablation(s) to run (default: both)")
    return p.parse_args()


def main():
    args = parse_args()
    run = set(args.abl) if args.abl else {1, 2}
    suffix = ""

    if 1 in run:
        print(f"\n=== Ablation 1: lambda sweep ===")
        abl1_dir = f"{ABLATION1_DIR}{suffix}"
        results1 = load_results(abl1_dir)
        if not results1:
            print(f"  No results found in {abl1_dir}")
        else:
            print(f"  Loaded {len(results1)} result files from {abl1_dir}")
            print_dataset_summary(results1)
            by_dataset1 = parse_ablation1(results1)
            print(f"  After filtering: {len(by_dataset1)} datasets")
            plot_ablation1(by_dataset1, suffix=suffix)

    if 2 in run:
        print(f"\n=== Ablation 2: objective comparison ===")
        abl2_dir = f"{ABLATION2_DIR}{suffix}"
        results2 = load_results(abl2_dir)
        if not results2:
            print(f"  No results found in {abl2_dir}")
        else:
            print(f"  Loaded {len(results2)} result files from {abl2_dir}")
            print_dataset_summary(results2)
            by_dataset2 = parse_ablation2(results2)
            print(f"  After filtering: {len(by_dataset2)} datasets")
            plot_ablation2(by_dataset2, suffix=suffix)


if __name__ == "__main__":
    main()
