"""
Raincloud plots for the NeurIPS camera-ready result files (store/neurips/results), built directly
from the raw results lines (no LaTeX table / loop-outcome sidecars needed).

For each method: a jittered point strip (one point per benchmark), a half-violin
density cloud, a star at the median, a diamond at the mean, the change of
median/mean vs. Original under the axis, and the number of benchmarks without
an evaluated repaired policy ("NA": repair OOM/timeout/infeasible, or no final
evaluation).

Two figures are written (eval split by default):
  plots/rainclouds_neurips_<split>_<stat>.pdf          Original, MILP (--main-lambda, default 1e-3), MH, FALoss
  plots/rainclouds_neurips_lambda_<split>_<stat>.pdf   Original, MILP for every lambda (--with-l1 adds MILP-L1)
plus plots/rainclouds_neurips_summary.json with the plotted statistics, the repair
and verification status counts per method, and the benchmarks verified SAFE.

Penalty trees (PT) were not re-run for the camera-ready sweep; their column is taken
from the previously stored results: values from the LaTeX performance table
results/<split>.tex (recovered checkpoints included, 'Imputed' cells use the
Original value as in that table), NA from the previous figure's
plots/rainclouds_<split>_both.failures.json, and verification outcomes from
results/loop_statistics_sources.json. Use --no-pt to drop the column.

Benchmarks: the union over all result files, restricted to benchmarks where at
least one method found faults and produced an evaluated repaired policy. A
benchmark missing from a method's file (e.g. pruned from the MILP sweep) counts
as NA for that method.

Usage (repo root):
    python3 -m plotting.rainclouds_neurips                      # eval, median+mean
    python3 -m plotting.rainclouds_neurips --split train --stat median
    python3 -m plotting.rainclouds_neurips --main-lambda 0.01
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import RESULT_DIR
from results_processors.repair_loop import parse_results_file
from results_processors.helpers.utils import PRINT_PREFIX_BENCHMARK, BASE_RENAMES, PRINT_PREFIXES
import plotting.plot_styles as ps
from plotting.plot_styles import (
    C_ORIGINAL, C_MILP, C_L1, C_MH, C_FALOSS, C_PT, lambda_ramp, set_style, style_axes,
)

FILE_PREFIX = "results-xgb_20_5_0.4-1000it_"
FILE_SUFFIX = "_all-data-20000N_plaja.txt"

# fixed colors per entity (never re-assigned when a column is dropped);
# shared with every other figure through plotting/plot_styles.py
COLOR_ORIGINAL, COLOR_L1, COLOR_PT = C_ORIGINAL, C_L1, C_PT
COLOR_MILP, COLOR_MH, COLOR_FALOSS = C_MILP, C_MH, C_FALOSS
FIGSIZE = (14.5, 5.8)

NON_CONVERGED = {"oom", "timeout", "infeasible", "maxit", "walltime", "oom-repair", "BUG"}


# ----------------------------------------------------------------------------- data

def pretty_name(row):
    """Paper name of a benchmark, e.g. '2 way line (20, 10) Icy + Park'."""
    full, prefix = row["benchmark"], row["prefix"] or ""
    base_keys = [k for k in BASE_RENAMES if full.startswith(k)]
    base = BASE_RENAMES[max(base_keys, key=len)] if base_keys else (row["base"] or full)
    pp = PRINT_PREFIXES.get(prefix, prefix)
    return PRINT_PREFIX_BENCHMARK.get((pp, base), f"{base} {pp}".strip())


def lambda_label(method):
    """'MILP λ=1e-05' -> 'λ=1e-5' (short tick label for the sweep figure)."""
    lam = float(method.split("=")[1])
    return f"λ={lam:g}" if lam >= 0.01 else f"λ=1e{int(np.log10(lam))}"


def table_key(row):
    """Row name used in results/<split>.tex: '<base> <prefix>' with pretty prefix."""
    full, prefix = row["benchmark"], row["prefix"] or ""
    base_keys = [k for k in BASE_RENAMES if full.startswith(k)]
    base = BASE_RENAMES[max(base_keys, key=len)] if base_keys else (row["base"] or full)
    return f"{base} {PRINT_PREFIXES.get(prefix, prefix)}".strip()


def read_pt_table(path):
    """{table name: (goal, safe)} for the PT column of results/<split>.tex.
    Columns: Original, MILP, MH, FALoss, PT for %Goal, then the same for %Safe.
    'Imputed' / 'No policy' cells take the Original value (table convention)."""
    import re as _re
    out = {}
    for line in open(path):
        if "&" not in line or line.lstrip().startswith(("%", r"\multicolumn", "Benchmark", r"\textbf{Average}")):
            continue
        cells = [c.strip() for c in line.split(r"\\")[0].split("&")]
        if len(cells) != 11 or not cells[0]:
            continue
        def val(i, orig):
            c = cells[i]
            if _re.search(r"Imputed|No policy", c):
                return float(cells[orig])
            try:
                return float(c)
            except ValueError:
                return np.nan
        out[cells[0]] = (val(5, 1), val(10, 6))
    return out


def load_pt(data, benches, split, table_path, failures_path, outcomes_path):
    """PT values (goal, safe), NA benchmark set and verification outcomes from
    the previously stored result files. Returns None if the table is missing."""
    if not table_path or not os.path.exists(table_path):
        return None
    table = read_pt_table(table_path)
    goal, safe, missing = [], [], []
    for b in benches:
        row = next(rows[b] for rows in data.values() if b in rows)
        key = table_key(row)
        if key in table:
            g, sf = table[key]
            goal.append(g); safe.append(sf)
        else:
            goal.append(np.nan); safe.append(np.nan); missing.append(key)
    na = set()
    if failures_path and os.path.exists(failures_path):
        ev = json.load(open(failures_path)).get("evidence", {}).get("PT", {})
        na = set(ev.keys())
    outcomes = {}
    if outcomes_path and os.path.exists(outcomes_path):
        runs = json.load(open(outcomes_path))
        runs = runs.get("benchmarks", runs.get("runs", {}))
        for b, methods in runs.items():
            r = methods.get("PT", {})
            r = r.get("full", r)
            if "outcome" in r:
                outcomes[b] = r["outcome"]
    keys = [table_key(next(rows[b] for rows in data.values() if b in rows)) for b in benches]
    return {"goal": goal, "safe": safe, "keys": keys, "missing": missing,
            "na": len([k for k in keys if k in na or not np.isfinite(table[k][1]) if k in table] + [k for k in keys if k not in table]),
            "na_benchmarks": sorted(k for k in keys if k in na),
            "verification": Counter(outcomes.get(k, "N/A") for k in keys),
            "verified_safe": sorted(k for k in keys if outcomes.get(k) == "Safe"),
            "table": os.path.abspath(table_path)}


def discover(result_dir):
    """Map method label -> results file. MILP files are keyed 'MILP λ=<value>'."""
    files = OrderedDict()
    paths = sorted(glob.glob(os.path.join(result_dir, f"{FILE_PREFIX}*{FILE_SUFFIX}")))
    lambdas = {}
    for p in paths:
        key = os.path.basename(p)[len(FILE_PREFIX):-len(FILE_SUFFIX)]
        m = re.match(r"milp_(.+)lambda_(\d+)samples", key)
        if m:
            lambdas[float(m.group(1))] = p
        elif key == "milp-l1":
            files["MILP-L1"] = p
        elif key == "retraining-augment":
            files["MH"] = p
        elif key == "retraining-loss":
            files["FALoss"] = p
    for lam in sorted(lambdas):
        files[f"MILP λ={lam:g}"] = lambdas[lam]
    return files


def load(files, split):
    """Return {method: {benchmark: row}} with the metric values attached."""
    gf_o, bf_o = f"original_gf_{split}", f"original_bf_{split}"
    gf_f, bf_f = f"final_gf_{split}", f"final_bf_{split}"
    data = OrderedDict()
    for method, path in files.items():
        rows = {}
        for r in parse_results_file(path):
            if r["benchmark"] in rows:          # keep the first line per benchmark
                continue
            r["orig_safe"], r["orig_goal"] = 100 - r[bf_o], r[gf_o]
            r["final_safe"], r["final_goal"] = 100 - r[bf_f], r[gf_f]
            rows[r["benchmark"]] = r
        data[method] = rows
    return data


def select_benchmarks(data):
    """Union of benchmarks over all files where some method repaired and evaluated."""
    common = set.union(*[set(rows) for rows in data.values()])

    def repaired(b):
        for rows in data.values():
            r = rows.get(b)
            if r and r["num_faults"] > 0 and np.isfinite(r["final_safe"]):
                return True
        return False
    return sorted(b for b in common if repaired(b))


def na_counts_for(data, benches, methods):
    """Benchmarks per method whose run did not converge (repair OOM/timeout/
    infeasible, iteration limit) or has no evaluated repaired policy. Measured
    values of non-converged runs are still plotted (as in the original figure)."""
    out = {}
    for m in methods:
        n = 0
        for b in benches:
            r = data[m].get(b)
            if r is None or not np.isfinite(r["final_safe"]) or r["testing_status"] in NON_CONVERGED:
                n += 1
        out[m] = n
    return out


def series_for(data, benches, methods):
    """{metric: OrderedDict(label -> list of values)}; Original from any file."""
    out = {"safe": OrderedDict(), "goal": OrderedDict()}
    for metric in out:
        out[metric]["Original"] = []
    for b in benches:
        for metric in out:
            v = np.nan
            for rows in data.values():
                r = rows.get(b)
                if r and np.isfinite(r[f"orig_{metric}"]):
                    v = r[f"orig_{metric}"]
                    break
            out[metric]["Original"].append(v)
    for m in methods:
        for metric in out:
            out[metric][m] = [data[m][b][f"final_{metric}"] if b in data[m] else np.nan
                              for b in benches]
    return out


# ----------------------------------------------------------------------------- plot

def half_violin(ax, x0, values, color, width=0.30, gap=0.08):
    v = values[~np.isnan(values)]
    if len(v) < 3 or np.std(v) < 1e-9:
        return
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(v, bw_method=0.35)
    ys = np.linspace(0, 100, 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    ax.fill_betweenx(ys, x0 + gap, x0 + gap + dens, color=color, alpha=0.75, lw=0, zorder=2)


def raincloud_panel(ax, series, colors, title, ylabel, n_total, rng, stat="both", na_counts=None):
    """One panel: jittered points, half-violin, median star / mean diamond,
    'k/n NA' above each method and the change vs the first column (Original)
    in text rows under the axis. Layout and sizes as in the earlier rainclouds."""
    show_median = stat in ("median", "both")
    show_mean = stat in ("mean", "both")
    orig = np.asarray(list(series.values())[0], dtype=float)
    orig_med, orig_mean = np.nanmedian(orig), np.nanmean(orig)
    stats = {}
    for i, (label, vals) in enumerate(series.items()):
        vals = np.asarray(vals, dtype=float)
        notna = ~np.isnan(vals)
        ok = vals[notna]
        color = colors[i]
        jitter = rng.uniform(-0.10, 0.10, size=int(notna.sum()))
        xs = np.full(int(notna.sum()), i - 0.25) + jitter
        ax.scatter(xs, ok, s=14, color=color, alpha=0.55, lw=0, zorder=3)
        half_violin(ax, i, vals, color)
        if show_median:
            ax.scatter([i - 0.05], [np.median(ok)], marker="*", s=340, color=color,
                       edgecolor="black", linewidth=1.1, zorder=5)
        if show_mean:
            ax.scatter([i - 0.05], [np.mean(ok)], marker="D", s=55, color=color,
                       edgecolor="black", linewidth=1.1, zorder=5)
        n_na = na_counts.get(label, int((~notna).sum())) if na_counts else int((~notna).sum())
        stats[label] = {"n": int(notna.sum()), "na": n_na,
                        "median": float(np.median(ok)), "mean": float(np.mean(ok)),
                        "d_median": float(np.median(ok) - orig_med),
                        "d_mean": float(np.mean(ok) - orig_mean)}
        if i > 0:
            ax.text(i, 103.5, f"{n_na}/{n_total} NA", ha="center",
                    fontsize=ps.ANNOT_SIZE, style="italic", color="0.25")
            rows = ([(np.median(ok) - orig_med, -0.13)] if show_median else []) + \
                   ([(np.mean(ok) - orig_mean, -0.13 if not show_median else -0.205)] if show_mean else [])
            for d, y in rows:
                ax.annotate(f"{d:+.1f}", xy=(i, 0), xycoords=("data", "axes fraction"),
                            xytext=(i, y), textcoords=("data", "axes fraction"),
                            ha="center", fontsize=ps.DELTA_SIZE, color="0.15", annotation_clip=False)
    strip = ([(r"$\Delta$ med", -0.13)] if show_median else []) + \
            ([(r"$\Delta$ mean", -0.13 if not show_median else -0.205)] if show_mean else [])
    for lab, y in strip:
        ax.annotate(lab, xy=(0, 0), xycoords=("data", "axes fraction"),
                    xytext=(0.45, y), textcoords=("data", "axes fraction"),
                    ha="right", fontsize=ps.SUBTITLE_SIZE, style="italic", color="0.35", annotation_clip=False)
    style_axes(ax, grid="y")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(list(series.keys()))
    ax.set_xlim(-0.6, len(series) - 0.25)
    ax.set_ylim(-4, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    return stats


def draw(series, colors, n_total, stat, out, figsize=None, na_counts=None, tick_labels=None):
    """Two panels (%Safe, %Goal) on the 14.5 x 5.8 in canvas of the earlier
    rainclouds; a wider canvas is used automatically for many columns."""
    from matplotlib.lines import Line2D
    n_cols = len(series["safe"])
    w, h = figsize or (max(FIGSIZE[0], 2.3 * n_cols), FIGSIZE[1])
    rng = np.random.default_rng(0)
    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(w, h))
    st_safe = raincloud_panel(ax_s, series["safe"], colors, "%Safe of Initial States", "%Safe", n_total, rng, stat, na_counts)
    st_goal = raincloud_panel(ax_g, series["goal"], colors, "%Goal of policy runs", "%Goal", n_total, rng, stat, na_counts)
    if tick_labels:
        for ax in (ax_s, ax_g):
            ax.set_xticklabels(tick_labels)
    handles = []
    if stat in ("median", "both"):
        handles.append(Line2D([], [], marker="*", ls="", ms=15, mfc="0.75", mec="black", label="median"))
    if stat in ("mean", "both"):
        handles.append(Line2D([], [], marker="D", ls="", ms=7, mfc="0.75", mec="black", label="mean"))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.23, wspace=0.14)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, 0.13), ncols=1,
               frameon=True, handletextpad=0.3, labelspacing=0.4, columnspacing=1.2)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"safe": st_safe, "goal": st_goal}


def status_summary(data, benches):
    """Repair / verification status counts per method, on the plotted benchmark
    set and on every benchmark the method's file contains."""
    summary = OrderedDict()
    for method, rows in data.items():
        def counts(bs):
            rep = Counter(rows[b]["testing_status"] for b in bs if b in rows)
            ver = Counter(rows[b]["verification_status"] for b in bs if b in rows)
            safe = sorted(pretty_name(rows[b]) for b in bs if b in rows
                          and rows[b]["verification_status"] == "SAFE")
            return {"n": sum(1 for b in bs if b in rows), "repair": dict(rep),
                    "verification": dict(ver), "verified_safe": safe}
        summary[method] = {"plotted_benchmarks": counts(benches),
                           "all_benchmarks": counts(sorted(rows))}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--stat", choices=["median", "mean", "both"], default="both")
    ap.add_argument("--result-dir", default=RESULT_DIR)
    ap.add_argument("--main-lambda", type=float, default=0.001,
                    help="lambda shown as 'MILP' in the main (method comparison) figure")
    ap.add_argument("--with-l1", action="store_true", help="also show MILP-L1 in both figures")
    ap.add_argument("--no-pt", action="store_true", help="drop the penalty-trees column")
    ap.add_argument("--font-scale", type=float, default=1.0, help="multiply every font size")
    ap.add_argument("--pt-table", default=None, help="LaTeX performance table with the PT column (default results/<split>.tex)")
    ap.add_argument("--pt-failures", default=None, help="previous figure's failures.json for PT NA counts")
    ap.add_argument("--pt-outcomes", default=os.path.join("results", "loop_statistics_sources.json"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "plots"))
    args = ap.parse_args()

    set_style(args.font_scale)

    files = discover(args.result_dir)
    data = load(files, args.split)
    benches = select_benchmarks(data)
    n_total = len(benches)
    lambda_methods = [m for m in data if m.startswith("MILP λ")]
    main_milp = f"MILP λ={args.main_lambda:g}"
    if main_milp not in data:
        ap.error(f"{main_milp} not found; available: {lambda_methods}")

    suffix = f"{args.split}_{args.stat}"
    out_main = os.path.join(args.out_dir, f"rainclouds_neurips_{suffix}.pdf")
    out_lambda = os.path.join(args.out_dir, f"rainclouds_neurips_lambda_{suffix}.pdf")

    # (1) method comparison
    methods = [m for m in (["MILP-L1"] if args.with_l1 else []) + [main_milp, "MH", "FALoss"] if m in data]
    series = series_for(data, benches, methods)
    for metric in series:   # relabel the chosen lambda as plain "MILP"
        series[metric] = OrderedDict((("MILP" if k == main_milp else k), v) for k, v in series[metric].items())
    colors = [COLOR_ORIGINAL] + [{"MILP-L1": COLOR_L1, main_milp: COLOR_MILP, "MH": COLOR_MH, "FALoss": COLOR_FALOSS}[m]
                                 for m in methods]
    na_main = na_counts_for(data, benches, methods)
    na_main = {("MILP" if k == main_milp else k): v for k, v in na_main.items()}
    pt = None
    if not args.no_pt:
        pt = load_pt(data, benches, args.split,
                     args.pt_table or os.path.join("results", f"{args.split}.tex"),
                     args.pt_failures or os.path.join(args.out_dir, f"rainclouds_{args.split}_both.failures.json"),
                     args.pt_outcomes)
        if pt is None:
            print("(!) no stored PT table found; PT column skipped")
        else:
            if pt["missing"]:
                print(f"(!) PT table has no row for: {pt['missing']}")
            series["safe"]["PT"] = pt["safe"]
            series["goal"]["PT"] = pt["goal"]
            colors.append(COLOR_PT)
            na_main["PT"] = pt["na"]
    stats_main = draw(series, colors, n_total, args.stat, out_main, na_counts=na_main)

    # (2) lambda sweep
    methods_l = [m for m in (["MILP-L1"] if args.with_l1 else []) if m in data] + lambda_methods
    series_l = series_for(data, benches, methods_l)
    colors_l = [COLOR_ORIGINAL] + ([COLOR_L1] if "MILP-L1" in methods_l else []) + lambda_ramp(len(lambda_methods))
    ticks = ["Original"] + [lambda_label(m) if m.startswith("MILP λ") else m for m in methods_l]
    stats_lambda = draw(series_l, colors_l, n_total, args.stat, out_lambda,
                        na_counts=na_counts_for(data, benches, methods_l), tick_labels=ticks)

    summary = {
        "split": args.split, "stat": args.stat, "result_dir": os.path.abspath(args.result_dir),
        "benchmarks": [pretty_name(next(rows[b] for rows in data.values() if b in rows)) for b in benches],
        "main_lambda": args.main_lambda,
        "stats_main": stats_main, "stats_lambda": stats_lambda,
        "status": status_summary(data, benches),
    }
    if pt is not None:
        summary["status"]["PT"] = {
            "source": pt["table"], "n": len(benches) - len(pt["missing"]),
            "verification": dict(pt["verification"]), "verified_safe": pt["verified_safe"],
            "na_benchmarks": pt["na_benchmarks"]}
    out_json = os.path.join(args.out_dir, f"rainclouds_neurips_summary_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"n = {n_total} benchmarks; wrote {os.path.abspath(out_main)}, {os.path.abspath(out_lambda)}, {os.path.abspath(out_json)}")
    print(f"\n{'method':16s} {'n':>3s} {'NA':>3s} | {'%Safe med':>9s} {'Δ':>6s} {'mean':>6s} {'Δ':>6s} | {'%Goal med':>9s} {'Δ':>6s} {'mean':>6s} {'Δ':>6s} | verified SAFE (plotted set / all)")
    for label in series_l["safe"]:
        s, g = stats_lambda["safe"][label], stats_lambda["goal"][label]
        key = label
        vs = summary["status"].get(key, {})
        ver = f"{len(vs.get('plotted_benchmarks', {}).get('verified_safe', []))} / {len(vs.get('all_benchmarks', {}).get('verified_safe', []))}" if vs else "-"
        print(f"{label:16s} {s['n']:3d} {s['na']:3d} | {s['median']:9.1f} {s['d_median']:+6.1f} {s['mean']:6.1f} {s['d_mean']:+6.1f} | "
              f"{g['median']:9.1f} {g['d_median']:+6.1f} {g['mean']:6.1f} {g['d_mean']:+6.1f} | {ver}")
    for label in ("MH", "FALoss", "PT"):
        if label in stats_main["safe"]:
            s, g = stats_main["safe"][label], stats_main["goal"][label]
            vs = summary["status"][label]
            ver = (f"{len(vs['plotted_benchmarks']['verified_safe'])} / {len(vs['all_benchmarks']['verified_safe'])}"
                   if "plotted_benchmarks" in vs else f"{len(vs['verified_safe'])} (stored)")
            print(f"{label:16s} {s['n']:3d} {s['na']:3d} | {s['median']:9.1f} {s['d_median']:+6.1f} {s['mean']:6.1f} {s['d_mean']:+6.1f} | "
                  f"{g['median']:9.1f} {g['d_median']:+6.1f} {g['mean']:6.1f} {g['d_mean']:+6.1f} | {ver}")
    print("\nVerification status per method (plotted benchmark set):")
    for method, st in summary["status"].items():
        if "plotted_benchmarks" in st:
            print(f"  {method:16s} {st['plotted_benchmarks']['verification']}  SAFE: {st['plotted_benchmarks']['verified_safe']}")
        else:
            print(f"  {method:16s} {st['verification']}  SAFE: {st['verified_safe']}  (stored loop outcomes)")


if __name__ == "__main__":
    main()
