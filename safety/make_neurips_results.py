#!/usr/bin/env python3
"""
Build every NeurIPS (camera-ready) figure and table from the raw result files in one go.

    python3 make_neurips_results.py                 # -> neurips_results/
    python3 make_neurips_results.py --out somewhere --lambda 0.001

Outputs (all in --out):
    rainclouds_eval.pdf       %Safe / %Goal on the evaluation states, per method
    rainclouds_train.pdf      same on the repair (debugging) states
    performance_eval.tex      %Goal and %Safe on the evaluation states
    performance_train.tex     %Goal and %Safe on the repair states
    loop_statistics.tex       |E| (number of faults) and #Iterations
    runtime.tex               CE (fault analysis), Repair and Total time (s)
    loop_outcomes.tex         did each method's loop converge, and was it verified?
    sources.json              which cells came from the new runs and which from the defaults (manual logs inspected)

Methods: Original, MILP (lambda given by --lambda), MH (retraining-augment),
FALoss (retraining-loss), PT (penalty trees).

Rainclouds: one point per benchmark; NA = the method's loop did not converge
(fault analysis / repair out of resources, iteration or time limit).
Values of non-converged runs are still plotted when they exist (last policy), matching the earlier figures.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import RESULT_DIR                                     # noqa: E402
from results_processors.repair_loop import parse_results_file          # noqa: E402
from results_processors.helpers.utils import BASE_RENAMES, PRINT_PREFIXES  # noqa: E402

METHODS = ["MILP", "MH", "FALoss", "PT"]
PERF_METHODS = ["Original"] + METHODS
FILE_PREFIX = "results-xgb_20_5_0.4-1000it_"
FILE_SUFFIX = "_all-data-20000N_plaja.txt"
RESULT_FILES = {                      # method -> results file (lambda filled in later)
    "MILP": "milp_{lam}lambda_100000samples",
    "MH": "retraining-augment",
    "FALoss": "retraining-loss",
    "PT": "penalty",
}

# ----------------------------------------------------------------------------- names

def short_name(benchmark, prefix):
    """Row name used in the tables: '1WL (15, 10) I+P', 'Beluga (3 Parts)', ..."""
    keys = [k for k in BASE_RENAMES if benchmark.startswith(k)]
    base = BASE_RENAMES[max(keys, key=len)] if keys else benchmark
    pp = PRINT_PREFIXES.get(prefix or "", prefix or "")
    name = f"{base} {pp}".strip()
    for old, new in [("1 way line", "1WL"), ("2 way line", "2WL"), ("Jigs", "Parts"),
                     ("Icy + Park", "I+P"), ("Inverted Pendulum", "Inv. Pend."),
                     ("Next Jig", ""), ("Bouncing Ball (0.01, 0.01)", r"Bounc. Ball ($\tau=0.01$)"),
                     ("Bouncing Ball (0.05, 0.05)", r"Bounc. Ball ($\tau=0.05$)"),
                     ("Bouncing Ball", "Bounc. Ball")]:
        name = name.replace(old, new)
    return re.sub(r"\s+", " ", name).strip()


def norm(name):
    return re.sub(r"\s+", "", name)


# ----------------------------------------------------------------------------- defaults

def read_default_table(path, ncols):
    """Rows of a default LaTeX table: [(group header or None, name, cells)].
    Group headers come as ('header', text); separators between row blocks as
    ('hline', None). Average rows are skipped (recomputed)."""
    rows = []
    for line in open(path):
        s = line.strip()
        if s.startswith(r"\multicolumn") and "textbf" in s:
            rows.append(("header", re.search(r"\\textbf\{(.*?)\}", s).group(1), None))
        elif s.startswith(r"\hline") and rows and rows[-1][0] == "row":
            rows.append(("hline", None, None))
        elif "&" in s and not s.startswith(("%", r"\cmidrule", "Benchmark", r"\textbf{Average}")) \
                and not s.startswith("&") and not s.startswith(r"\multicolumn"):
            cells = [c.strip() for c in s.split(r"\\")[0].split("&")]
            if len(cells) == ncols + 1 and cells[0]:
                rows.append(("row", cells[0], cells[1:]))
    return rows


def parse_num(cell):
    """LaTeX cell -> float (NaN if not a number); '${<}1$' -> 0.5 (shown as <1)."""
    c = cell.replace("$", "").replace("{", "").replace("}", "").replace("\\textit", "").strip()
    if c in ("<1", "{<}1"):
        return 0.5
    try:
        return float(c)
    except ValueError:
        return np.nan


def load_defaults(defaults_dir):
    """Benchmark layout (from eval.tex) and default cells for every table."""
    perf = {s: read_default_table(os.path.join(defaults_dir, f"{s}.tex"), 10) for s in ("eval", "train")}
    layout = [(k, n) for k, n, _ in perf["eval"] if k != "row"] and perf["eval"]
    stats = read_default_table(os.path.join(defaults_dir, "loop_statistics.tex"), 8)
    runtime = read_default_table(os.path.join(defaults_dir, "runtime.tex"), 12)
    outcomes = read_default_table(os.path.join(defaults_dir, "loop_outcomes.tex"), 4)

    def by_name(rows):
        return {norm(n): cells for k, n, cells in rows if k == "row"}
    layout = [(k, n) for k, n, _ in layout]                           # order, headers, hlines
    while layout and layout[-1][0] == "hline":
        layout.pop()
    d = {
        "layout": layout,
        "perf": {s: by_name(perf[s]) for s in perf},                  # name -> 10 cells
        "stats": by_name(stats),                                      # name -> 8 cells
        "runtime": by_name(runtime),                                  # name -> 12 cells
        "outcomes": by_name(outcomes),                                # name -> 4 macros
    }
    return d


# ----------------------------------------------------------------------------- new runs

def load_runs(result_dir, lam):
    """{method: {short name: parsed row}} from the new result files."""
    runs = {}
    for m, stem in RESULT_FILES.items():
        path = os.path.join(result_dir, f"{FILE_PREFIX}{stem.format(lam=f'{lam:g}')}{FILE_SUFFIX}")
        if not os.path.exists(path) and m == "MILP":      # files spell 1 as 1.0, 10 as 10.0
            alt = os.path.join(result_dir, f"{FILE_PREFIX}{stem.format(lam=str(float(lam)))}{FILE_SUFFIX}")
            path = alt if os.path.exists(alt) else path
        if not os.path.exists(path) and m == "MILP":
            cands = glob.glob(os.path.join(result_dir, f"{FILE_PREFIX}milp_*lambda_*samples{FILE_SUFFIX}"))
            sys.exit(f"MILP results for lambda={lam:g} not found at {path}\n  available: {cands}")
        rows = {}
        if os.path.exists(path):
            for r in parse_results_file(path):
                key = norm(short_name(r["benchmark"], r["prefix"]))
                rows.setdefault(key, r)            # first line per benchmark
        runs[m] = rows
        print(f"  {m:7s} {len(rows):2d} benchmarks  {os.path.basename(path) if rows else '(no file: defaults only)'}")
    return runs


CONVERGED = {"optimal", "suboptimal", "safe", "no_faults"}


def outcome_macro(r):
    """loop_outcomes macro for a parsed row (see the \\newcommand block)."""
    st, ver = r["testing_status"], r["verification_status"]
    if st in CONVERGED:
        if ver == "SAFE":
            return r"\cS"
        if ver == "UNSAFE":
            return r"\cU"
        return r"\cK"                                   # TIMEOUT / OOM / WALLTIME / PENDING / N/A
    if st == "oom":
        return r"\CE"
    if st in ("timeout", "infeasible"):
        return r"\RTO"
    if st == "maxit":
        return r"\MI"
    if st in ("walltime", "oom-repair"):
        return r"\MT"
    return r"\nc{?} & \nc{--}"


NON_CONVERGED_MACROS = {r"\CE", r"\RTO", r"\MI", r"\MT"}


# ----------------------------------------------------------------------------- assemble

class Cells:
    """Collects table cells and remembers where each came from."""
    def __init__(self):
        self.sources = {}

    def pick(self, table, name, column, new_value, default_cell, fmt, prefer_new=True):
        """Value from the preferred source, else the other one. Returns LaTeX text."""
        ok = new_value is not None and not (isinstance(new_value, float) and np.isnan(new_value))
        src = self.sources.setdefault(table, {}).setdefault(name, {})
        if (prefer_new or default_cell is None) and ok:
            src[column] = "new"
            return fmt(new_value)
        if default_cell is not None:
            src[column] = "default"
            return default_cell
        if ok:
            src[column] = "new"
            return fmt(new_value)
        src[column] = "missing"
        return r"\textit{NA}"


def fmt_pct(v):
    return f"{v:.2f}"


def fmt_int(v):
    return f"{int(round(v))}"


def fmt_sec(v):
    return r"${<}1$" if 0 <= v < 1 else f"{int(round(v))}"


def perf_values(r, split):
    """(goal, safe) of the original and the final policy for one parsed row."""
    gf_o, bf_o, gf_f, bf_f = (r[f"original_gf_{split}"], r[f"original_bf_{split}"],
                              r[f"final_gf_{split}"], r[f"final_bf_{split}"])
    return (gf_o, 100 - bf_o), (gf_f, 100 - bf_f)


def build(runs, d, lam, from_runs):
    """Return dict table -> list of row tuples, plus the numeric data for plots."""
    cells = Cells()
    pref = {m: (m in from_runs) for m in METHODS}
    pref["Original"] = True
    layout = d["layout"]
    names = [n for k, n in layout if k == "row"]
    out = {}
    plot = {s: {"safe": OrderedDict((m, []) for m in PERF_METHODS),
                "goal": OrderedDict((m, []) for m in PERF_METHODS)} for s in ("eval", "train")}
    na = {m: 0 for m in METHODS}
    verified = {m: [] for m in METHODS}

    # --- loop outcomes (needed first: NA counts for the plots)
    rows = []
    for name in names:
        key = norm(name)
        default = d["outcomes"].get(key)
        macros = []
        for i, m in enumerate(METHODS):
            r = runs[m].get(key)
            new = outcome_macro(r) if r is not None else None
            cell = cells.pick("loop_outcomes", name, m, new, default[i] if default else None, str, pref[m])
            macros.append(cell)
            if cell in NON_CONVERGED_MACROS:
                na[m] += 1
            if cell == r"\cS":
                verified[m].append(name)
        rows.append((name, macros))
    out["loop_outcomes"] = rows

    # --- performance tables and plot data
    for split in ("eval", "train"):
        rows = []
        for name in names:
            key = norm(name)
            default = d["perf"][split].get(key)
            vals = {}
            # Original: from any new run that has the benchmark, else default
            orig = None
            for m in METHODS:
                r = runs[m].get(key)
                if r is not None:
                    o, _ = perf_values(r, split)
                    if np.isfinite(o[0]):
                        orig = o
                        break
            vals["Original"] = orig
            for m in METHODS:
                r = runs[m].get(key)
                vals[m] = perf_values(r, split)[1] if r is not None else None
            goal_cells, safe_cells = [], []
            for i, m in enumerate(PERF_METHODS):
                v = vals[m]
                g = cells.pick(f"performance_{split}", name, f"{m}/goal", v[0] if v else None,
                               default[i] if default else None, fmt_pct, pref[m])
                s_ = cells.pick(f"performance_{split}", name, f"{m}/safe", v[1] if v else None,
                                default[5 + i] if default else None, fmt_pct, pref[m])
                goal_cells.append(g)
                safe_cells.append(s_)
                plot[split]["goal"][m].append(parse_num(g))
                plot[split]["safe"][m].append(parse_num(s_))
            rows.append((name, goal_cells + safe_cells))
        out[f"performance_{split}"] = rows

    # --- loop statistics: |E| and #Iterations (= recorded iterations + 1, as before)
    rows = []
    for name in names:
        key = norm(name)
        default = d["stats"].get(key)
        c = []
        for i, m in enumerate(METHODS):
            r = runs[m].get(key)
            c.append(cells.pick("loop_statistics", name, f"{m}/faults",
                                float(r["num_faults"]) if r is not None else None,
                                default[i] if default else None, fmt_int, pref[m]))
        for i, m in enumerate(METHODS):
            r = runs[m].get(key)
            its = None
            if r is not None:
                its = float(r["gurobi_iterations"] + r["verification_iterations"] + 1)
            c.append(cells.pick("loop_statistics", name, f"{m}/its", its,
                                default[4 + i] if default else None, fmt_int, pref[m]))
        rows.append((name, c))
    out["loop_statistics"] = rows

    # --- runtime: CE (fault analysis), Repair (fix timer), Total = CE + Repair
    rows = []
    for name in names:
        key = norm(name)
        default = d["runtime"].get(key)
        c = []
        for block, field in (("ce", "fa"), ("repair", "gurobi"), ("total", None)):
            for i, m in enumerate(METHODS):
                r = runs[m].get(key)
                v = None
                if r is not None:
                    rt = r["runtimes"]
                    v = rt["fa"] + rt["gurobi"] if field is None else rt[field]
                    v = None if (isinstance(v, float) and np.isnan(v)) else float(v)
                col = {"ce": 0, "repair": 4, "total": 8}[block] + i
                c.append(cells.pick("runtime", name, f"{m}/{block}", v,
                                    default[col] if default else None, fmt_sec, pref[m]))
        rows.append((name, c))
    out["runtime"] = rows

    return out, plot, na, verified, cells.sources


# ----------------------------------------------------------------------------- latex

def body_rows(layout, table_rows, ncols, hline_after_header=r"\\\hline"):
    """Interleave group headers / hlines from the layout with the table rows."""
    rows = dict(table_rows)
    lines = []
    for kind, name in layout:
        if kind == "header":
            lines.append(rf"\multicolumn{{{ncols + 1}}}{{|c|}}{{\textbf{{{name}}}}}{hline_after_header}")
        elif kind == "hline":
            lines.append(r"\hline")
        else:
            lines.append(f"{name} & " + " & ".join(rows[name]) + r" \\")
    return lines


def average_row(table_rows, ncols):
    cols = [[parse_num(c) for c in cells] for _, cells in table_rows]
    avgs = []
    for j in range(ncols):
        v = [row[j] for row in cols if np.isfinite(row[j])]
        avgs.append(f"\\textbf{{{np.mean(v):.2f}}}" if v else r"\textbf{-}")
    return r"\textbf{Average} & " + " & ".join(avgs) + r" \\"


def write_performance(path, layout, rows, split, lam):
    lines = [
        f"% generated by make_neurips_results.py; MILP lambda={lam:g}; split={split}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{|l|rrrrr|rrrrr|}",
        r"\toprule",
        r"Benchmark & \multicolumn{5}{|c|}{$\%Goal$} & \multicolumn{5}{|c|}{$\%Safe$} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        " & " + " & ".join(PERF_METHODS) + " & " + " & ".join(PERF_METHODS) + r" \\",
        r"\midrule",
        *body_rows(layout, rows, 10),
        r"\hline",
        average_row(rows, 10),
        r"\hline",
        r"\end{tabular}}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


def write_stats(path, layout, rows, lam):
    lines = [
        f"% generated by make_neurips_results.py; MILP lambda={lam:g}; #Iterations = loop iterations incl. the final check",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{|l|rrrr|rrrr|}",
        r"\toprule",
        r"Benchmark & \multicolumn{4}{|c|}{$|E|$} & \multicolumn{4}{|c|}{$\#$Iterations} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        " & " + " & ".join(METHODS) + " & " + " & ".join(METHODS) + r" \\",
        r"\midrule",
        *body_rows(layout, rows, 8),
        r"\hline",
        r"\end{tabular}}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


def write_runtime(path, layout, rows, lam):
    lines = [
        f"% generated by make_neurips_results.py; MILP lambda={lam:g}",
        r"% CE = fault analysis (counterexample search); Repair = MILP solve / retraining; Total = CE + Repair.",
        r"% ${<}1$: recorded value below one second. NA: not available in either source.",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{|l|rrrr|rrrr|rrrr|}",
        r"\toprule",
        r"Benchmark & \multicolumn{4}{|c|}{CE time (s)} & \multicolumn{4}{|c|}{Repair time (s)} & \multicolumn{4}{|c|}{Total time (s)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        " & " + " & ".join(METHODS * 3) + r" \\",
        r"\midrule",
        *body_rows(layout, rows, 12),
        r"\hline",
        r"\end{tabular}}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


def write_outcomes(path, layout, rows, na, verified, lam):
    n = len(rows)
    counts = {m: {"S": 0, "U": 0, "K": 0} for m in METHODS}
    for _, macros in rows:
        for m, mac in zip(METHODS, macros):
            if mac == r"\cS": counts[m]["S"] += 1
            elif mac == r"\cU": counts[m]["U"] += 1
            elif mac == r"\cK": counts[m]["K"] += 1
    lines = [
        f"% generated by make_neurips_results.py; MILP lambda={lam:g}",
        r"\newcommand{\nc}[1]{\textcolor{gray}{\textit{#1}}}",
        r"% each macro fills two cells: Converged & Verif.",
        r"\newcommand{\cS}{\checkmark & \checkmark}      % verified safe",
        r"\newcommand{\cU}{\checkmark & $\times$}        % verifier finds unsafe path",
        r"\newcommand{\cK}{\checkmark & \nc{OOR}}        % verifier out of resources",
        r"\newcommand{\CE}{\nc{CE-OOR} & \nc{--}}        % fault analysis out of resources",
        r"\newcommand{\RTO}{\nc{Repair-TO} & \nc{--}}    % repair timed out / infeasible",
        r"\newcommand{\MI}{\nc{MaxIt} & \nc{--}}         % iteration limit",
        r"\newcommand{\MT}{\nc{MaxTime} & \nc{--}}       % job time limit",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|l |ll |ll |ll |ll|}",
        r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c|}}{{{m}}}" for m in METHODS) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        "Benchmark & " + " & ".join("Converged & Verif." for _ in METHODS) + r" \\",
        r"\midrule",
        *body_rows(layout, rows, 8, hline_after_header=r"\\" + "\n" + r"\midrule"),
        r"\midrule",
        rf"\textbf{{Converged (of {n})}} & " + " & ".join(rf"\textbf{{{n - na[m]}}} &" for m in METHODS) + r" \\",
        r"\quad verified safe ($\checkmark$) & " + " & ".join(f"& {counts[m]['S']}" for m in METHODS) + r" \\",
        r"\quad unsafe path found ($\times$) & " + " & ".join(f"& {counts[m]['U']}" for m in METHODS) + r" \\",
        r"\quad inconclusive (OOR) & " + " & ".join(f"& {counts[m]['K']}" for m in METHODS) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- plots

def write_rainclouds(path, plot_split, na, stat):
    from plotting.rainclouds_neurips import draw
    from plotting.plot_styles import METHOD_COLORS, set_style
    set_style()
    label = {"MILP": "MILP Repair"}                      # plot label; tables keep "MILP"
    series = {metric: OrderedDict((label.get(m, m), plot_split[metric][m]) for m in PERF_METHODS)
              for metric in ("safe", "goal")}
    n_total = len(series["safe"]["Original"])
    colors = [METHOD_COLORS[m] for m in PERF_METHODS]
    na_plot = {label.get(m, m): v for m, v in na.items()}
    return draw(series, colors, n_total, stat, path, na_counts=na_plot)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", default=RESULT_DIR, help="raw results (default: store/neurips/results)")
    # Some runs are annoying to parse automatically. Manuallz inspect them
    ap.add_argument("--defaults-dir", default=os.path.join("neurips_results", "default_tables"),
                    help="hand-curated tables used as defaults")
    ap.add_argument("--out", default="neurips_results")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.001, help="MILP lambda (default 1e-3)")
    ap.add_argument("--stat", choices=["median", "mean", "both"], default="both", help="raincloud markers")
    ap.add_argument("--from-runs", default="MILP",
                    help="comma-separated methods read from --result-dir first (others: defaults first)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"defaults: {args.defaults_dir}\nnew runs: {args.result_dir}")
    d = load_defaults(args.defaults_dir)
    runs = load_runs(args.result_dir, args.lam)
    names = [n for k, n in d["layout"] if k == "row"]
    unknown = {m: sorted(k for k in runs[m] if k not in {norm(n) for n in names}) for m in METHODS}
    for m, extra in unknown.items():
        if extra:
            print(f"  (i) {m}: {len(extra)} benchmarks not in the default tables are ignored")

    from_runs = [m.strip() for m in args.from_runs.split(",") if m.strip()]
    print(f"from runs first: {from_runs}; from defaults first: {[m for m in METHODS if m not in from_runs]}")
    tables, plot, na, verified, sources = build(runs, d, args.lam, from_runs)
    layout = d["layout"]
    write_performance(os.path.join(args.out, "performance_eval.tex"), layout, tables["performance_eval"], "eval", args.lam)
    write_performance(os.path.join(args.out, "performance_train.tex"), layout, tables["performance_train"], "train", args.lam)
    write_stats(os.path.join(args.out, "loop_statistics.tex"), layout, tables["loop_statistics"], args.lam)
    write_runtime(os.path.join(args.out, "runtime.tex"), layout, tables["runtime"], args.lam)
    write_outcomes(os.path.join(args.out, "loop_outcomes.tex"), layout, tables["loop_outcomes"], na, verified, args.lam)
    stats = {}
    for split in ("eval", "train"):
        stats[split] = write_rainclouds(os.path.join(args.out, f"rainclouds_{split}.pdf"), plot[split], na, args.stat)

    # provenance + headline numbers
    n_default = {t: sum(v == "default" for row in s.values() for v in row.values()) for t, s in sources.items()}
    n_new_fallback = {t: sum(v == "new" for row in s.values() for c, v in row.items()
                             if c.split("/")[0] not in from_runs + ["Original"]) for t, s in sources.items()}
    n_missing = {t: sum(v == "missing" for row in s.values() for v in row.values()) for t, s in sources.items()}
    json.dump({"lambda": args.lam, "result_dir": os.path.abspath(args.result_dir),
               "defaults_dir": os.path.abspath(args.defaults_dir), "benchmarks": names,
               "na_counts": na, "verified_safe": verified, "raincloud_stats": stats,
               "cells_from_defaults": n_default, "cells_missing": n_missing, "cells": sources},
              open(os.path.join(args.out, "sources.json"), "w"), indent=2)

    print(f"\n{len(names)} benchmarks; MILP lambda = {args.lam:g}; wrote {os.path.abspath(args.out)}/")
    print("cells taken from the default tables:", n_default)
    if any(n_new_fallback.values()):
        print("cells of default-first methods that had to come from the raw runs:", n_new_fallback)
    if any(n_missing.values()):
        print("(!) cells with no value in either source:", n_missing)
    print("\nnon-converged (NA) per method:", na)
    print("verified safe:", {m: len(v) for m, v in verified.items()}, verified)
    for split in ("eval", "train"):
        print(f"\n{split}: {'method':9s} %Safe med / mean    %Goal med / mean")
        for m in stats[split]["safe"]:
            s, g = stats[split]["safe"][m], stats[split]["goal"][m]
            print(f"       {m:12s} {s['median']:5.1f} / {s['mean']:5.1f}       {g['median']:5.1f} / {g['mean']:5.1f}")


if __name__ == "__main__":
    main()
