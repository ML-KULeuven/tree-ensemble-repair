#!/usr/bin/env python3
"""
Aggregate the generalization-study results produced by main_generalization.py.

For each dataset (averaged over folds) it reports, per repair-set
fraction, the share of HELD-OUT counterexamples that the MILP fixed as a side
effect, plus the MILP runtime and the accuracy cost.

The 100% fraction has no held-out set: it only contributes the runtime and the
accuracy cost of the complete repair.

Outputs: terminal tables (normal + latex) and a line plot (figures/generalization.{pdf,png}).

Usage:
    python collect_generalization_results.py [-m xgb|rf] [--no-plot]
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from utils import CACHE_DIR

RESULTS_DIR = f"{CACHE_DIR}/results/generalization"
OUT_DIR = "figures"

# datasets are printed in this order; any other dataset is appended alphabetically
DATASET_ORDER = [
    "CompasTwoYears", "DutchCensus", "BankMarketingFull", "LawSchool",
    "ACSIncome", "ACSEmployment", "ACSPublicCoverage", # "ACSMobility",
    "ACSTravelTime", "MEPSPanel21", "DefaultCreditCardClients"
]

OK_STATUS = ("optimal", "suboptimal")

# metrics gathered per (dataset, fraction)
METRICS = ["fixed", "milp_time", "dbacc", "n_repair", "n_holdout"]


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _parse_name(fname, model_type):
    """Return dataset name and fold, or (None, None) if the file does not match."""
    m = re.match(rf"^(.+)_{model_type}_(\d+)\.json$", fname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_results(model_type):
    results = {}
    if not os.path.isdir(RESULTS_DIR):
        print(f"  No results directory {RESULTS_DIR}")
        return results
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json") or fname.startswith("._"):
            continue
        dname, fold = _parse_name(fname, model_type)
        if dname is None:
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                res = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # truncated (killed job / full disk) or not a result file: don't kill the run
            print(f"  (!) Skipping {fname}: unreadable ({type(e).__name__}) — rerun that fold")
            continue
        if not res.get("generalization"):
            # already fair, not enumerable, or too few counterexamples to split
            # (main_generalization.py runs no splits for those folds)
            print(f"  (-) Skipping {dname} fold {fold}: {res.get('status', 'no results')}")
            continue
        # The 100% solve is the reference for every speedup in this fold. If it hit
        # the Gurobi cap its runtime is a floor, not the cost of the full repair, so
        # every ratio against it is understated -> drop the fold rather than report
        # lower bounds as measurements.
        recs = res["generalization"]
        full_frac = max(r["fraction"] for r in recs)
        capped = [r for r in recs if r["fraction"] == full_frac and r.get("status") != "optimal"]
        if capped:
            print(f"  (!) Skipping {dname} fold {fold}: full ({full_frac:.0%}) repair "
                  f"{capped[0].get('status')} at {capped[0]['milp_time']:.0f}s "
                  f"(timeout {res.get('milp_timeout', '?')}s) — speedups would be lower bounds")
            continue
        results[(dname, fold)] = res
    return results


def aggregate(results):
    """-> data[dname][fraction][metric] = list of per-fold values."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_ce = defaultdict(list)
    for (dname, fold), res in results.items():
        # NOTE: Δ bacc here is vs the COMPRESSED model (all other collectors: vs the original).
        #       Only printed in a terminal table, not used in the paper or figures.
        base_bacc = res.get("compressed_at", {}).get("test_bacc")
        n_ce[dname].append(res.get("n_counterexamples", np.nan))

        # Speedup is measured against the *same fold's* full repair:
        # the fractions of one fold share a model and a counterexample set, so their
        # solve times are comparable; across folds/datasets they are not.
        recs = res["generalization"]
        full_frac = max((r["fraction"] for r in recs), default=None)
        tf = next((r["milp_time"] for r in recs
                   if r["fraction"] == full_frac and r.get("status") in OK_STATUS), None)

        for rec in res["generalization"]:
            frac = rec["fraction"]
            cell = data[dname][frac]
            if rec.get("status") not in OK_STATUS:
                cell["failed"].append(rec.get("status"))
                continue
            cell["n_repair"].append(rec["n_repair"])
            cell["n_holdout"].append(rec["n_holdout"])
            if rec.get("holdout_fixed_frac") is not None:
                cell["fixed"].append(100 * rec["holdout_fixed_frac"])
            cell["milp_time"].append(rec["milp_time"])
            cell["status"].append(rec.get("status"))
            if tf is not None and rec["milp_time"] > 0:
                cell["speedup"].append(tf / rec["milp_time"])
            if base_bacc is not None and "test_bacc" in rec:
                cell["dbacc"].append(100 * (rec["test_bacc"] - base_bacc))
    return data, n_ce


def _ordered(datasets):
    known = [d for d in DATASET_ORDER if d in datasets]
    return known + sorted(d for d in datasets if d not in DATASET_ORDER)


def _ms(vals, geo=False):
    """mean, std (0 if a single value, nan if empty).

    geo=True switches to the geometric mean, with the std as a multiplicative
    factor. Speedups are ratios: a 10x and a 0.1x average to 1x geometrically
    (they cancel, which is right) but to 5.05x arithmetically.
    """
    if not vals:
        return np.nan, np.nan
    if geo:
        logs = np.log([v for v in vals if v > 0])
        if len(logs) == 0:
            return np.nan, np.nan
        m = float(np.exp(np.mean(logs)))
        s = float(np.exp(np.std(logs, ddof=1))) if len(logs) > 1 else 1.0
        return m, s
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Terminal tables
# ──────────────────────────────────────────────────────────────────────────────

def print_table(data, fracs, metric, title, fmt="{:.1f}", unit="", geo=False):
    datasets = _ordered(data.keys())
    w = max([len(d) for d in datasets] + [12]) + 1
    header = f"{'dataset':<{w}}" + "".join(f"{f'{f:.0%}':>16}" for f in fracs)
    print(f"\n  {title}")
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for dname in datasets:
        row = f"{dname:<{w}}"
        for frac in fracs:
            m, s = _ms(data[dname][frac][metric], geo=geo)
            sep = "/" if geo else "±"   # geometric std is a factor, not an offset
            row += "{:>16}".format("--" if np.isnan(m)
                                   else f"{fmt.format(m)}{unit}{sep}{fmt.format(s)}")
        print("  " + row)
    # average over datasets (unweighted, so big datasets don't dominate)
    row = f"{'AVERAGE':<{w}}"
    for frac in fracs:
        per_ds = [_ms(data[d][frac][metric], geo=geo)[0] for d in datasets]
        per_ds = [v for v in per_ds if not np.isnan(v)]
        if not per_ds:
            row += "{:>16}".format("--")
        else:
            avg = _ms(per_ds, geo=geo)[0]
            row += "{:>16}".format(f"{fmt.format(avg)}{unit}")
    print("  " + "-" * len(header))
    print("  " + row)


def print_overview(data, n_ce, fracs, results):
    datasets = _ordered(data.keys())
    w = max([len(d) for d in datasets] + [12]) + 1
    full_frac = max(fracs) if fracs else None
    print(f"\n  Runs per dataset")
    print(f"  {'dataset':<{w}}{'folds':>7}{'#counterex./fold (mean±std)':>28}{'failed MILPs':>14}"
          f"{'full repair [s]':>17}")
    for dname in datasets:
        folds = sorted({f for (d, f) in results if d == dname})
        failed = sum(len(data[dname][fr]["failed"]) for fr in fracs)
        m, s = _ms(n_ce[dname])
        # absolute anchor for the speedup table: what 100% actually costs
        t_full, _ = _ms(data[dname][full_frac]["milp_time"]) if full_frac else (np.nan, 0)
        t_str = "--" if np.isnan(t_full) else f"{t_full:.1f}"
        print(f"  {dname:<{w}}{len(folds):>7}{f'{m:.0f}±{s:.0f}':>28}{failed:>14}{t_str:>17}")


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX
# ──────────────────────────────────────────────────────────────────────────────

def write_latex(data, fracs, metric, path):
    datasets = _ordered(data.keys())
    lines = [
        r"% generated by collect_generalization_results.py",
        r"\begin{tabular}{l" + "rr" * len(fracs) + "}",
        r"\toprule",
        r"& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{f:.0%} repaired}}".replace("%", r"\%")
                           for f in fracs) + r" \\",
        "Dataset & " + " & ".join(r"\% fixed & time (s)" for _ in fracs) + r" \\",
        r"\midrule",
    ]
    for dname in datasets:
        cells = []
        for frac in fracs:
            fx, _ = _ms(data[dname][frac][metric])
            rt, _ = _ms(data[dname][frac]["milp_time"])
            cells.append("--" if np.isnan(fx) else f"{fx:.1f}")
            cells.append("--" if np.isnan(rt) else f"{rt:.1f}")
        lines.append(f"{dname.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    cells = []
    for frac in fracs:
        fx = [_ms(data[d][frac][metric])[0] for d in datasets]
        rt = [_ms(data[d][frac]["milp_time"])[0] for d in datasets]
        fx = [v for v in fx if not np.isnan(v)]
        rt = [v for v in rt if not np.isnan(v)]
        cells.append("--" if not fx else f"{np.mean(fx):.1f}")
        cells.append("--" if not rt else f"{np.mean(rt):.1f}")
    lines += [r"\textbf{Average} & " + " & ".join(cells) + r" \\",
              r"\bottomrule", r"\end{tabular}"]

    # os.makedirs(os.path.dirname(path), exist_ok=True)
    # with open(path, "w") as f:
    #     f.write("\n".join(lines) + "\n")
    # print(f"\n  Saved LaTeX table to {path}")
    # just print it to the terminal for now
    print(f"\n  LaTeX table ({path}):\n")
    print("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot(data, fracs, metric, path_stem):
    """Two standalone figures, one aggregate line each: mean over datasets +/- 1 std.

    Each dataset first collapses to its own mean at a fraction (so a dataset with
    more folds does not dominate); the line is the mean of those, the
    shadow their std across datasets. The runtime figure is on a log scale, so it
    aggregates geometrically (the shadow is a multiplicative factor).

    Writes {path_stem}.{pdf,png} (generalization) and
    {path_stem}_runtime.{pdf,png} (MILP runtime).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import C_MILP, C_RT, save, set_style, style_axes

    set_style()
    datasets = _ordered(data.keys())
    x = np.array([100 * f for f in fracs])

    def _band(key, geo=False):
        mu, lo, hi = [], [], []
        for f in fracs:
            vals = [v for v in (_ms(data[d][f][key], geo=geo)[0] for d in datasets)
                    if not np.isnan(v)]
            m, s = _ms(vals, geo=geo)
            mu.append(m)
            if geo:
                lo.append(m / s if s and not np.isnan(s) else m)
                hi.append(m * s if s and not np.isnan(s) else m)
            else:
                lo.append(m - s)
                hi.append(m + s)
        return np.array(mu), np.array(lo), np.array(hi)

    def _save(fig, stem):
        fig.tight_layout()
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        save(fig, stem)
        print(f"  Saved {stem}.{{pdf,png}}")

    def _xaxis(ax, xs):
        ax.set_xlim(min(xs) - 4, max(xs) + 4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.set_xlabel("Repair set size\n(% of counterexamples)", labelpad=12)
        style_axes(ax)

    # ── Generalization ────────────────────────────────────────────────────────
    # The 100% split has an empty held-out set, so it yields no measurement: the
    # curve is drawn only over the fractions that actually have one.
    fx, fx_lo, fx_hi = _band(metric)
    keep = ~np.isnan(fx)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.fill_between(x[keep], np.clip(fx_lo[keep], 0, 100), np.clip(fx_hi[keep], 0, 100),
                    color=C_MILP, alpha=0.20, linewidth=0)
    ax.plot(x[keep], fx[keep], color=C_MILP, linewidth=2.5, marker="o", markersize=8)
    # the y-axis starting at 0 already shows the no-repair baseline (0% fixed)
    ax.set_ylabel("Held-out counterexamples fixed (%)", labelpad=12)
    ax.set_title("Generalization of MILP repair")
    ax.set_ylim(0, 101)
    _xaxis(ax, x[keep])
    _save(fig, path_stem)

    # ── MILP runtime ──────────────────────────────────────────────────────────
    # linear axis, so aggregate arithmetically (the band is clipped at 0: a
    # runtime cannot be negative)
    rt, rt_lo, rt_hi = _band("milp_time")
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.fill_between(x, np.clip(rt_lo, 0, None), rt_hi, color=C_RT,
                    alpha=0.20, linewidth=0)
    ax.plot(x, rt, color=C_RT, linewidth=2.5, marker="s", markersize=8)
    ax.set_ylabel("MILP runtime (s)", labelpad=12)
    ax.set_ylim(bottom=0)
    ax.set_title("MILP runtime (s)")
    _xaxis(ax, x)
    _save(fig, f"{path_stem}_runtime")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_type", default="xgb", choices=["xgb", "rf"])
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def main():
    global RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = args.results_dir
    metric = "fixed"

    results = load_results(args.model_type)
    if not results:
        print(f"No results found in {RESULTS_DIR} for model_type={args.model_type}.")
        return
    print(f"\n  Loaded {len(results)} result files from {RESULTS_DIR}")

    data, n_ce = aggregate(results)
    fracs = sorted({rec["fraction"] for res in results.values()
                    for rec in res["generalization"]})

    print(f"\n{'=' * 78}\n GENERALIZATION OF THE MILP REPAIR ({args.model_type})"
          f"\n{'=' * 78}")
    print_overview(data, n_ce, fracs, results)
    print_table(data, fracs, metric,
                "Held-out counterexamples fixed [%], mean±std over folds")
    print_table(data, fracs, "speedup",
                "MILP speedup vs. the same fold's full (100%) repair [x, geometric mean]",
                fmt="{:.1f}", unit="x", geo=True)
    # NOTE: vs the COMPRESSED model, unlike the other collectors; not used in the paper
    print_table(data, fracs, "dbacc", "Δ test balanced accuracy vs compressed model [pp]",
                fmt="{:.2f}")

    # write_latex(data, fracs, metric,
    #             f"{OUT_DIR}/generalization_{args.model_type}.tex")
    if not args.no_plot:
        plot(data, fracs, metric, f"{OUT_DIR}/generalization_{args.model_type}")
    print()


if __name__ == "__main__":
    main()
