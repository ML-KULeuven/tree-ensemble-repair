#!/usr/bin/env python3
"""
Compare basic composite MILP vs. hard balanced-accuracy budget MILP.

One combined table, printed to the terminal and then as LaTeX:
  - outcome of the budget MILP only (#folds needing repair / done / infeasible /
    timeout) — the basic MILP completes on every fold, so its counts add nothing;
  - then basic vs. budget side by side for Δ BAcc, iterations and runtime.

Reads, from {CACHE_DIR}/results/, per (dataset, fold):
  {ds}_{model}_{fold}_milp.json    basic composite MILP        (main.py -e milp)
  {ds}_{model}_{fold}_milp-b.json  hard-budget MILP            (main.py -e milp-b)
  {ds}_{model}_{fold}_orig.json    original model: Δ BAcc is measured against it
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_style import (C_BOX_LINE, C_BUDGET, C_MILP, C_REF,
                        ANNOT_SIZE, save, set_style, style_axes)
from utils import CACHE_DIR

# hard balanced-accuracy budget used by `main.py -e milp-b`
BUDGET = 0.02

DATASET_ORDER = [
    "ACSIncome", "ACSEmployment", "ACSPublicCoverage",
    "ACSTravelTime", "BankMarketingFull", "CompasTwoYears",
    "DefaultCreditCardClients", "DutchCensus", "LawSchool", "MEPSPanel21",
]

RESULTS_DIR = f"{CACHE_DIR}/results"
FIGURES_DIR = "figures"

# Colours from plot_style: green = the unconstrained MILP repair, blue = the
# balanced-accuracy-constrained one. The outcome bars (Figure 1) are black for
# "Repaired" and grey for its timeouts / infeasible runs.
C_BASIC = C_MILP


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", "-m", default="xgb",
                   choices=["xgb", "rf"])
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="Skip the three summary figures")
    return p.parse_args()


def result_filename(dname, model_type, fold, exp):
    """exp: "milp" (basic), "milp-b" (budget) or "orig" (original model)."""
    return f"{dname}_{model_type}_{fold}_{exp}.json"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def classify_status(block):
    """Return 'completed', 'infeasible', 'timeout', 'already_fair', or 'missing'.

    'max_iterations' (still unfair after MAX_IT repair iterations) is counted
    with the timeouts: both ran out of compute before reaching a fair model.
    'not_enumerable' folds run no MILP at all, like already-fair ones.
    """
    if not block:
        return "missing"
    status = block.get("status")
    if status in ("fair", "not_enumerable"):
        return "already_fair"
    if status == "infeasible":
        return "infeasible"
    if status in ("timeout", "max_iterations"):
        return "timeout"
    if status in ("optimal", "suboptimal"):
        return "completed"
    return "missing"


def fmt(mean, std, decimals=2):
    if mean is None:
        return "---"
    if std is None or std == 0.0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def fmt_runtime(seconds):
    if seconds is None:
        return "---"
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    return f"{seconds:.1f}s"


def fmt_rt(mean, std):
    if mean is None:
        return "---"
    return fmt_runtime(mean) + (f" ± {fmt_runtime(std)}" if std else "")


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset(dname, model_type):
    folds_data = {}
    for fold in range(5):
        def _load(exp):
            return load_json(os.path.join(RESULTS_DIR, result_filename(dname, model_type, fold, exp)))
        basic_result  = _load("milp")
        budget_result = _load("milp-b")
        if basic_result is None and budget_result is None:
            continue
        # Δ BAcc is measured against the original (uncompressed) model
        orig_result = _load("orig") or {}
        orig_bacc = orig_result.get("original_at", {}).get("test_bacc")
        folds_data[fold] = {
            "basic":     basic_result.get("composite_repaired_at", {}) if basic_result  else {},
            "budget":    budget_result.get("composite_repaired_at", {}) if budget_result else {},
            "orig_bacc": orig_bacc,
        }
    return folds_data


# ── LaTeX ─────────────────────────────────────────────────────────────────────

DATASET_SHORT = {
    "ACSIncome": "ACSIncome", "ACSEmployment": "ACSEmploy",
    "ACSPublicCoverage": "ACSPublic", "ACSTravelTime": "ACSTravel",
    "BankMarketingFull": "Bank", "CompasTwoYears": "Compas",
    "DefaultCreditCardClients": "Credit", "DutchCensus": "Dutch",
    "LawSchool": "LawSchool", "MEPSPanel21": "MEPS",
}


def _tex(cell):
    """'1.23 ± 0.45' -> '$1.23{\\pm}0.45$'; '1.2h ± 3m' -> '1.2h $\\pm$ 3m'.

    Runtime cells carry unit letters (h/m/s), which would be typeset as italic
    variables inside math mode, so those keep the \\pm outside the math.
    """
    cell = str(cell)
    if cell == "---":
        return "--"
    if "±" not in cell:
        return cell
    a, b = (p.strip() for p in cell.split("±"))
    if any(ch.isalpha() for ch in cell):
        return f"{a} $\\pm$ {b}"
    return f"${a}{{\\pm}}{b}$"


def print_latex(df, model_type, budget):
    mt = model_type.upper()
    print("\n" + "=" * 70)
    print("LATEX")
    print("=" * 70 + "\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\small")
    print(f"\\caption{{Hard balanced-accuracy budget ($b={budget:g}$) vs.\\ the "
          f"unconstrained MILP on {mt}. The outcome columns count only the folds "
          f"that required a repair ($n$); the unconstrained MILP completes on all "
          f"of them. Averages are over completed runs.}}")
    print(f"\\label{{tab:budget-{model_type}}}")
    print("\\begin{tabular}{l rrrr cc cc cc}")
    print("\\toprule")
    print(" & \\multicolumn{4}{c}{Budget MILP outcome} & "
          "\\multicolumn{2}{c}{$\\Delta$ BAcc (\\%)} & "
          "\\multicolumn{2}{c}{Iterations} & "
          "\\multicolumn{2}{c}{MILP runtime} \\\\")
    print("\\cmidrule(lr){2-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}")
    print("Dataset & $n$ & Done & Inf. & T/O & "
          "Basic & Budget & Basic & Budget & Basic & Budget \\\\")
    print("\\midrule")
    for _, r in df.iterrows():
        if r["dataset"] == "OVERALL":
            print("\\midrule")
            name = "\\textbf{Overall}"
        else:
            name = DATASET_SHORT.get(r["dataset"], r["dataset"])
        print(f"{name} & {r['n']} & {r['done']} & {r['infeasible']} & {r['timeout']} & "
              f"{_tex(r['basic_bacc_loss'])} & {_tex(r['budget_bacc_loss'])} & "
              f"{_tex(r['basic_iters'])} & {_tex(r['budget_iters'])} & "
              f"{_tex(r['basic_time'])} & {_tex(r['budget_time'])} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ── figures ───────────────────────────────────────────────────────────────────

def paired_folds(records, column):
    """-> {method: [values]} over the folds where *both* MILPs completed.

    Figures 2 and 3 compare the two solvers, so they may only use folds the
    budget MILP actually solved -- an infeasible or timed-out fold has no
    accuracy and no meaningful runtime to show. Restricting both columns to the
    same folds keeps the two boxes paired.
    """
    by_key = defaultdict(dict)
    for r in records:
        by_key[(r["dataset"], r["fold"])][r["method"]] = r
    out = {"basic": [], "budget": []}
    for methods in by_key.values():
        if len(methods) < 2 or any(m["status"] != "completed" for m in methods.values()):
            continue
        if any(m[column] is None for m in methods.values()):
            continue
        for name, rec in methods.items():
            out[name].append(rec[column])
    return out


def make_figures(records, model_type, budget):
    """Three summary figures: budget-MILP outcome, Δ BAcc, MILP runtime."""
    set_style()

    basic_label  = "MILP Repair"
    # the budget is a fraction of balanced accuracy; show it in the pp the
    # y-axes of figures 2-3 are already in
    budget_pct   = f"{budget * 100:g}%"
    budget_label = f"Hard Budget MILP Repair"

    # -- Figure 1: what the budget MILP did on the folds needing a repair -----
    # Folds already fair run no MILP at all, so they are not part of the
    # denominator; the two failure modes are drawn flush against each other.
    counts = defaultdict(int)
    for r in records:
        if r["method"] == "budget":
            counts[r["status"]] += 1
    n_folds = counts["completed"] + counts["infeasible"] + counts["timeout"]
    if n_folds == 0:
        print(f"\nNo budget runs for {model_type} at b={budget:g} -- no figures written.")
        return

    fig1, ax = plt.subplots(figsize=(6.5, 5))
    bar_labels = ["Repaired", "Timeout", "Infeasible"]
    bar_counts = [counts["completed"], counts["timeout"], counts["infeasible"]]
    percents = [c / max(n_folds, 1) * 100.0 for c in bar_counts]
    # labels: largest-remainder rounding, so the shown percentages sum to 100
    shown = [int(p) for p in percents]
    by_remainder = sorted(range(len(percents)), key=lambda i: percents[i] - shown[i], reverse=True)
    for i in by_remainder[:100 - sum(shown)]:
        shown[i] += 1
    width = 0.8
    # positions = [0.0, 1.15, 1.15 + width]   # failure bars share an edge
    positions = [0.0, 1.15, 2.30]             # equidistant bars
    # black = repaired, grey = the two failure modes (timeout, infeasible)
    bars = ax.bar(positions, percents, width=width,
                  color=["black", "#9E9E9E", "#9E9E9E"], edgecolor="none")
    y_top = 116.0
    ax.set_ylim(0, y_top)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("% folds needing repair")
    ax.set_title(f"Hard Budget MILP Outcomes ($b$={budget_pct})")
    for rect, count, pct in zip(bars, bar_counts, shown):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + y_top * 0.03,
                f"{count}/{n_folds}\n({pct}%)", ha="center", va="bottom",
                fontsize=ANNOT_SIZE)
    ax.axhline(100.0, color=C_REF, linestyle="--", linewidth=0.8)
    style_axes(ax, grid="y")
    fig1.tight_layout(pad=0.8)

    # -- Figure 2: per-fold balanced-accuracy change vs. the original model ---
    deltas = paired_folds(records, "delta_bacc")
    fig2, ax = plt.subplots(figsize=(6.5, 5))
    if deltas["basic"]:
        data = pd.DataFrame({basic_label:  pd.Series(deltas["basic"]),
                             budget_label: pd.Series(deltas["budget"])})
        sns.boxplot(data=data, palette=[C_BASIC, C_BUDGET], saturation=1,
                    linecolor=C_BOX_LINE, ax=ax, showfliers=False)
        ax.axhline(0, color=C_REF, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Δ balanced accuracy")
        ax.set_title(f"Δ balanced accuracy vs {model_type.upper()}")
    style_axes(ax, grid="y")
    fig2.tight_layout(pad=0.8)

    # -- Figure 3: per-fold MILP runtime -------------------------------------
    times = paired_folds(records, "time")
    fig3, ax = plt.subplots(figsize=(6.5, 5))
    if times["basic"]:
        data = pd.DataFrame({basic_label:  pd.Series(times["basic"]),
                             budget_label: pd.Series(times["budget"])})
        sns.boxplot(data=data, palette=[C_BASIC, C_BUDGET], saturation=1,
                    linecolor=C_BOX_LINE, ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel("MILP runtime (s)")
        ax.set_title("MILP Runtime")
    style_axes(ax, grid="y")
    fig3.tight_layout(pad=0.8)

    n_paired = len(deltas["basic"])
    print(f"\nFigures 2-3 use the {n_paired} folds both MILPs solved "
          f"(of {n_folds} needing a repair).")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    suffix = f"_{model_type}_budget{budget:g}"
    for fig, name in [(fig1, "budget_plot1_outcome"),
                      (fig2, "budget_plot2_bacc_loss"),
                      (fig3, "budget_plot3_runtime")]:
        # side-by-side panels: no tight bbox (see plot_style.save); tight_layout
        # above already keeps the labels inside the 6.5x5 canvas
        for path in save(fig, f"{FIGURES_DIR}/{name}{suffix}", tight=False):
            print(f"Saved summary plot to: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    budget = BUDGET
    model_type = args.model_type

    print(f"\n{'='*70}")
    print(f"Budget comparison  |  model={model_type.upper()}  |  budget={budget}")
    print(f"{'='*70}\n")

    rows = []
    fold_records = []   # one row per (dataset, fold, method), for the figures

    all_bacc_loss = {"basic": [], "budget": []}
    all_iters     = {"basic": [], "budget": []}
    all_times     = {"basic": [], "budget": []}
    all_status    = {"basic": defaultdict(int), "budget": defaultdict(int)}

    for dname in DATASET_ORDER:
        folds_data = load_dataset(dname, model_type)
        if not folds_data:
            continue

        bacc_loss  = {"basic": [], "budget": []}
        iters      = {"basic": [], "budget": []}
        times      = {"basic": [], "budget": []}
        status_cnt = {"basic": defaultdict(int), "budget": defaultdict(int)}

        for fold, fd in folds_data.items():
            orig_bacc = fd["orig_bacc"]
            for method in ("basic", "budget"):
                block = fd[method]
                s = classify_status(block)
                status_cnt[method][s] += 1
                all_status[method][s] += 1

                delta_bacc = it = t = None
                if s == "completed":
                    rep_bacc = block.get("test_bacc")
                    if orig_bacc is not None and rep_bacc is not None:
                        delta_bacc = (rep_bacc - orig_bacc) * 100.0
                        bacc_loss[method].append(-delta_bacc)
                    it = block.get("iterations")
                    if it is not None:
                        iters[method].append(it)
                    t = block.get("milp_time")
                    if t is not None:
                        times[method].append(t)

                fold_records.append({"dataset": dname, "fold": fold,
                                     "method": method, "status": s,
                                     "delta_bacc": delta_bacc,
                                     "iterations": it, "time": t})

        rec = {"dataset": dname}

        # ── budget-MILP outcome. Folds already fair before repair run no MILP at
        # all, so they are excluded: n is the number of folds that needed one,
        # and done + infeasible + timeout = n.
        b = status_cnt["budget"]
        rec["n"]          = b["completed"] + b["infeasible"] + b["timeout"]
        rec["done"]       = b["completed"]
        rec["infeasible"] = b["infeasible"]
        rec["timeout"]    = b["timeout"]

        for method in ("basic", "budget"):
            m, s = mean_std(bacc_loss[method])
            rec[f"{method}_bacc_loss"] = fmt(-m if m is not None else None, s)
            if m is not None:
                all_bacc_loss[method].append(m)

            im, is_ = mean_std(iters[method])
            tm, ts  = mean_std(times[method])
            rec[f"{method}_iters"] = fmt(im, is_, decimals=1)
            rec[f"{method}_time"]  = fmt_rt(tm, ts)
            # macro-average like the BAcc column above (and like
            # collect_results.py): carry the per-dataset mean, not the raw
            # folds, so a dataset with 5 solved folds does not outweigh one
            # with a single fold in the OVERALL row.
            if im is not None:
                all_iters[method].append(im)
            if tm is not None:
                all_times[method].append(tm)

        rows.append(rec)

    # ── summary row: counts are totals, the rest are means over datasets ───────
    tot = all_status["budget"]
    summary = {
        "dataset":    "OVERALL",
        "n":          tot["completed"] + tot["infeasible"] + tot["timeout"],
        "done":       tot["completed"],
        "infeasible": tot["infeasible"],
        "timeout":    tot["timeout"],
    }
    for method in ("basic", "budget"):
        m, s = mean_std(all_bacc_loss[method])
        summary[f"{method}_bacc_loss"] = fmt(-m if m is not None else None, s)
        im, is_ = mean_std(all_iters[method])
        tm, ts  = mean_std(all_times[method])
        summary[f"{method}_iters"] = fmt(im, is_, decimals=1)
        summary[f"{method}_time"]  = fmt_rt(tm, ts)
    rows.append(summary)

    # explicit order: the records are built per method, so the natural dict order
    # would group basic/budget instead of pairing them under each metric
    COLUMNS = ["dataset", "n", "done", "infeasible", "timeout",
               "budget_bacc_loss", "basic_bacc_loss", 
               "budget_iters", "basic_iters", 
               "budget_time","basic_time"]
    df = pd.DataFrame(rows)[COLUMNS]

    print("Budget MILP outcome (folds needing repair) | basic vs. budget: "
          "Δ BAcc [%], iterations, runtime")
    print("Δ BAcc is w.r.t. the original model; negative = loss, positive = gain.")
    print("T/O also counts runs still unfair after MAX_IT repair iterations.")
    print("Averages cover completed runs only, so the budget columns are "
          "conditioned on the folds it solved.\n")
    print(df.rename(columns={
        "infeasible":  "inf.",
        "timeout":     "T/O",
        "basic_bacc_loss":  "Δbacc% (basic)",
        "budget_bacc_loss": "Δbacc% (budget)",
        "basic_iters":  "it. (basic)",
        "budget_iters": "it. (budget)",
        "basic_time":   "time (basic)",
        "budget_time":  "time (budget)",
    }).to_string(index=False))

    print_latex(df, model_type, budget)

    if args.plot:
        make_figures(fold_records, model_type, budget)


if __name__ == "__main__":
    main()
