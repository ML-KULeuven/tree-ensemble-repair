#!/usr/bin/env python3
"""Summarize fairness-repair results across datasets and folds.

Merges, per (dataset, fold), the result files that ``main.py -e <exp> --cache``
writes to ``{CACHE_DIR}/results/`` (orig, lop, milp, fatt, budro; see
load_results) and emits terminal summary tables, the LaTeX tables used in the
paper, and the summary figures.

Fold set: folds whose LOP-compressed model was already fair are dropped, for
every method. There the MILP exits with 0 iterations and returns the model
untouched, so its row would be LOP's row under MILP's name; dropping the fold
for all methods keeps every column and every box on the same instances. 

Averaging convention: every AVERAGE row is a **macro** average, i.e. the mean of
the per-dataset means, with the +- being the spread *across datasets*. 
Datasets therefore weigh the same regardless of how many usable folds they contributed.
The runtime tables are the exception: their summary row is a MEDIAN row, the
median of the per-dataset means (no +-), and all times are in seconds.
The figures instead show one point per fold, pooled across datasets, so a box's
median will not match the corresponding AVERAGE row.
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_style import (C_BOX_LINE, C_BUDRO, C_FATT, C_LOP, C_MILP, C_REF, ANNOT_SIZE,
                        save, set_style, style_axes)
from utils import CACHE_DIR

RESULTS_DIR = f"{CACHE_DIR}/results"
FIGURES_DIR = "figures"

# Datasets dropped from every table and figure.
EXCLUDE_DATASETS = {"Adult", "ACSMobility", "ACSHealthInsurance", "GermanCredit"}

# Row order of the LaTeX tables; datasets with no results are skipped.
DATASET_ORDER = [
    "ACSIncome", "ACSEmployment", "ACSMobility", "ACSPublicCoverage",
    "ACSTravelTime", "BankMarketingFull", "CompasTwoYears",
    "DefaultCreditCardClients", "DutchCensus", "LawSchool", "MEPSPanel21",
]

# Short row labels for the LaTeX tables (the terminal tables keep the full
# names). Anything not listed here is printed as-is.
DATASET_LABELS = {
    "ACSEmployment": "ACSEmploy",
    "ACSPublicCoverage": "ACSPublic",
    "ACSTravelTime": "ACSTravel",
    "BankMarketingFull": "Bank",
    "CompasTwoYears": "Compas",
    "DefaultCreditCardClients": "Credit",
    "MEPSPanel21": "MEPS",
}

# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Method:
    """One column of the summary: where to read it and how to label it."""

    name: str           # internal key, also the column prefix
    block: str          # JSON key of the block it is read from
    label: str          # display name ({mt} is replaced by the model type)
    kind: str           # "reference" | "lop" | "milp" | "baseline"
    eligibility: str    # which folds count: "compressed" | "enumerable" | "present"
    color: str = "black"
    short: str = ""             # short prefix, used by the model-size table
    label_long: str = ""        # fair-count table header; falls back to `label`
    label_plot: str = ""        # figure axis label; falls back to `label`
    rt_labels: tuple = ()       # runtime-table headers (MILP variants only)

    def head(self, mt):
        return self.label.format(mt=mt)

    def head_long(self, mt):
        return (self.label_long or self.label).format(mt=mt)

    @property
    def plot_label(self):
        return self.label_plot or self.label


# Eligibility rules:
#   "compressed" - the fold counts iff veritas could enumerate the compressed
#                  model's OC space (see `compressed_ok` in build_frame).
#   "enumerable" - the fold counts iff this method's own block is present and
#                  was enumerable.
METHODS = {
    m.name: m for m in [
        Method("original", "original_at", "Original {mt}", "reference", "compressed"),
        Method("lop", "compressed_at", "LOP", "lop", "compressed",
               color=C_LOP, label_long="{mt}+LOP"),
        Method("milp_composite", "composite_repaired_at", "MILP", "milp", "compressed",
               color=C_MILP, short="milp", label_plot="MILP Repair",
               rt_labels=("MILP Iters", "MILP Time", "Veritas Time")),
        Method("fatt", "fatt", "FATT", "baseline", "enumerable", color=C_FATT),
        Method("budro", "budro", "BuDRO", "baseline", "enumerable", color=C_BUDRO),
    ]
}


def select_methods():
    """Active methods, in the order they appear as columns."""
    return list(METHODS.values())


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def fmt_float(decimals):
    return lambda v: f"{v:.{decimals}f}"


def fmt_percent(decimals):
    """Value is a fraction in [0, 1]; print it as a percentage."""
    return lambda v: f"{v * 100:.{decimals}f}%"


def fmt_points(decimals, signed=False, suffix=""):
    """Value is already in percent / percentage points."""
    spec = f"{'+' if signed else ''}.{decimals}f"
    return lambda v: f"{v:{spec}}{suffix}"


def fmt_runtime(seconds):
    """Seconds as X.Yh / X.Ym / X.Ys ('---' when missing)."""
    if seconds is None or pd.isna(seconds):
        return "---"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def fmt_seconds(seconds):
    """Seconds with one decimal, for the runtime tables ('---' when missing)."""
    if seconds is None or pd.isna(seconds):
        return "---"
    return f"{seconds:.1f}"


def fmt_pm(mean, std, fmt):
    """LaTeX 'mean $\\pm$ std' cell."""
    if mean is None:
        return "---"
    return f"{fmt(mean)} $\\pm$ {fmt(std if std is not None else 0.0)}"


def mean_std(values):
    """Mean and (sample) std of `values`, with std 0.0 for a single value."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def median(values):
    """Median of `values`, ignoring missing entries (None when nothing is left)."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return statistics.median(vals) if vals else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

# {ds}_{model}_{fold}_{exp}.json, or {ds}_{fold}_{exp}.json for the
# model-independent baselines (dataset names contain no underscore)
FILENAME_RE = re.compile(r"^([^_]+)_(?:(xgb|rf)_)?(\d+)_([A-Za-z-]+)\.json$")

# experiment (main.py --exp) -> block read from its result file
EXP_BLOCKS = {
    "orig": "original_at",
    "lop": "compressed_at",
    "milp": "composite_repaired_at",
    "fatt": "fatt",
    "budro": "budro",
}
# baselines with their own training procedure: no model type in their file
# name, so the xgb and rf summaries report the same numbers
MODEL_INDEPENDENT = ("fatt", "budro")


def is_enumerable(block):
    """True unless the block explicitly reports that veritas ran out of time."""
    return (isinstance(block, dict)
            and block.get("enumerable") is not False
            and block.get("status") != "not_enumerable")


def load_results(results_dir, model_type):
    """-> {(dataset, fold): {block: dict}} for the requested model type,
    merging the per-experiment result files of each fold.

    Only the method blocks are merged: every file also carries its own
    top-level `runtime`, and those would overwrite each other.
    """
    results = defaultdict(dict)
    for filename in sorted(os.listdir(results_dir)):
        match = FILENAME_RE.match(filename)
        if not match:
            continue
        dataset, model, fold, exp = (match.group(1), match.group(2),
                                     int(match.group(3)), match.group(4))
        if exp not in EXP_BLOCKS or dataset in EXCLUDE_DATASETS:
            continue
        # model-independent baselines have no model type in their file name
        if (exp in MODEL_INDEPENDENT) != (model is None):
            continue
        if model is not None and model != model_type:
            continue

        with open(os.path.join(results_dir, filename)) as f:
            block = json.load(f).get(EXP_BLOCKS[exp])
        if block is not None:
            results[(dataset, fold)][EXP_BLOCKS[exp]] = block

    # keep the folds that have results for this model type
    return {k: v for k, v in results.items()
            if "original_at" in v or "compressed_at" in v}


def build_frame(results, methods):
    """Flatten the result files into one row per (dataset, fold, method).

    Columns:
      eligible       - the fold counts for this method (see Method.eligibility)
      repair_needed  - eligible, and this fold's compressed model was still
                       unfair (these are the folds we repair and report on)
      enumerable     - this method's own block was verifiable (fair-count table)
      already_fair   - the compressed model of this fold had 0 unfair boxes
      compressed_ok  - veritas could enumerate the compressed model of this fold
    """
    rows = []
    for (dataset, fold), result in sorted(results.items()):
        compressed = result.get("compressed_at") or {}
        composite = result.get("composite_repaired_at") or {}
        # compressed_ok: veritas could enumerate the compressed model's OC-space
        compressed_ok = is_enumerable(composite) and is_enumerable(compressed)
        already_fair = compressed_ok and compressed.get("#unfair_boxes") == 0

        for method in methods:
            block = result.get(method.block) or {}
            enumerable = is_enumerable(block) and bool(block)
            if method.eligibility == "compressed":
                eligible = compressed_ok
            elif method.eligibility == "enumerable":
                eligible = enumerable
            else:
                eligible = bool(block)

            rows.append({
                "dataset": dataset,
                "fold": fold,
                "method": method.name,
                "compressed_ok": compressed_ok,
                "already_fair": already_fair,
                "enumerable": is_enumerable(block),
                "eligible": eligible,
                "repair_needed": eligible and not already_fair,
                "unfair_boxes": block.get("#unfair_boxes"),
                "emp_fairness": block.get("emp_fairness"),
                "test_acc": block.get("test_acc"),
                "test_bacc": block.get("test_bacc"),
                "num_leaves": None if method.kind == "milp" else block.get("num_leaves"),
                # main.py logs MILP solves; +1 counts the final verification pass
                "iterations": block["iterations"] + 1 if block.get("iterations") is not None else None,
                # MILP variants are timed by their solve time, 
                # baselines by their training time.
                "time": block.get("milp_time") if method.kind == "milp" else block.get("runtime"),
                "veritas_time": block.get("fairness_time"),
                "leaves_total": block.get("leaves_total"),
                "leaves_encoded": block.get("leaves_encoded"),
                "leaves_modified": block.get("leaves_modified"),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def fold_values(df, column, mask="eligible"):
    """-> {method: {dataset: [per-fold values]}}, dropping missing entries."""
    out = defaultdict(lambda: defaultdict(list))
    sub = df[df[mask]]
    for method, dataset, value in zip(sub["method"], sub["dataset"], sub[column]):
        if value is not None and not pd.isna(value):
            out[method][dataset].append(value)
    return out


def dataset_means(values, method, datasets):
    """Per-dataset mean of `values[method]`, aligned to `datasets`."""
    per_dataset = values.get(method, {})
    return [statistics.mean(per_dataset[d]) if per_dataset.get(d) else None
            for d in datasets]


def dataset_stats(values, method, datasets):
    """Per-dataset (mean, std) of `values[method]`, aligned to `datasets`."""
    per_dataset = values.get(method, {})
    return [mean_std(per_dataset.get(d, [])) for d in datasets]


def pooled(values, method):
    """All per-fold values of one method, across datasets."""
    return [v for vals in values.get(method, {}).values() for v in vals]


def summary_table(datasets, columns, use_median=False):
    """Terminal table: one row per dataset plus a macro-averaged AVERAGE row
    (with `use_median`, a MEDIAN row: the median of the per-dataset values).

    `columns` is a list of (header, per-dataset values, formatter).
    """
    records = []
    for i, dataset in enumerate(datasets):
        record = {"dataset": dataset}
        for header, values, fmt in columns:
            record[header] = "---" if values[i] is None else fmt(values[i])
        records.append(record)

    average = {"dataset": "MEDIAN" if use_median else "AVERAGE"}
    for header, values, fmt in columns:
        center = median(values) if use_median else mean_std(values)[0]
        average[header] = "---" if center is None else fmt(center)
    records.append(average)
    return pd.DataFrame.from_records(records)


def latex_table(caption, headers, rows):
    print("\n\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print("\\begin{tabular}{l" + "c" * len(headers) + "}")
    print("\\toprule")
    print("Dataset & " + " & ".join(headers) + " \\\\")
    print("\\midrule")
    for label, cells in rows:
        # Row labels are dataset names (plus TOTAL / AVERAGE), abbreviated to
        # keep the tables inside the page width.
        print(DATASET_LABELS.get(label, label) + " & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# --------------------------------------------------------------------------- #
# Accuracy change vs. the original model (terminal + LaTeX table 2 + figure 2)
# --------------------------------------------------------------------------- #

REFERENCE = "original"


def fold_scores(df, column):
    """-> {method: {dataset: {fold: value}}} over the folds each method is
    eligible for."""
    scores = defaultdict(lambda: defaultdict(dict))
    sub = df[df["eligible"]]
    for method, dataset, fold, value in zip(
            sub["method"], sub["dataset"], sub["fold"], sub[column]):
        if value is not None and not pd.isna(value):
            scores[method][dataset][fold] = value
    return scores


def already_fair_folds(df):
    """-> {dataset: {folds whose compressed model already had 0 unfair boxes}}."""
    folds = defaultdict(set)
    fair = df[df["already_fair"]]
    for dataset, fold in zip(fair["dataset"], fair["fold"]):
        folds[dataset].add(fold)
    return folds


def fold_deltas(scores, already_fair, method, dataset, skip_already_fair=True):
    """Per-fold change vs. the original model, in percentage points.

    Negative means the method scores worse than the model it repairs. Fold
    matched: a fold contributes only when both models have a score for it. Folds
    where the compressed model was already fair are skipped by default -- no
    method had to intervene there, so charging one for it would compare it on a
    fold its competitors never faced.
    """
    reference = scores[REFERENCE].get(dataset, {})
    return [(value - reference[fold]) * 100.0
            for fold, value in scores[method].get(dataset, {}).items()
            if fold in reference and not (skip_already_fair and fold in already_fair[dataset])]


def reference_level(scores, already_fair, dataset):
    """Per-fold score of the original model on the same folds, in percent."""
    return [value * 100.0 for fold, value in scores[REFERENCE].get(dataset, {}).items()
            if fold not in already_fair[dataset]]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def print_fair_counts(df, methods, datasets):
    """Folds that were enumerable and verified fair, per method."""
    fairness_methods = [m for m in methods if m.kind != "reference"]

    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # method -> dataset -> [fair, total]
    verified = df[df["enumerable"] & df["unfair_boxes"].notna()]
    for method, dataset, boxes in zip(verified["method"], verified["dataset"], verified["unfair_boxes"]):
        cell = counts[method][dataset]
        cell[1] += 1
        cell[0] += boxes == 0

    records, totals = [], {}
    for dataset in datasets:
        record = {"dataset": dataset}
        for method in fairness_methods:
            fair, total = counts[method.name][dataset]
            record[f"{method.name}_fair"] = f"{fair}/{total}"
        records.append(record)

    total_record = {"dataset": "TOTAL"}
    for method in fairness_methods:
        fair = sum(counts[method.name][d][0] for d in datasets)
        total = sum(counts[method.name][d][1] for d in datasets)
        totals[method.name] = (fair, total)
        pct = (fair / total * 100.0) if total else 0.0
        total_record[f"{method.name}_fair"] = f"{fair}/{total} ({pct:.1f}%)"
    records.append(total_record)

    table = pd.DataFrame.from_records(records)
    print("\nFair fold counts (enumerable + #unfair_boxes=0)")
    print(table.to_string(index=False))
    return table, totals


def print_fairness_metrics(df, methods, datasets):
    # NOTE: all 50 folds, already-fair ones included. 
    # Every other table uses only the folds that needed repair, so the numbers are not comparable.
    fairness_methods = [m for m in methods if m.kind != "reference"]
    boxes = fold_values(df, "unfair_boxes")
    emp = fold_values(df, "emp_fairness")

    columns = [(f"{m.name}_unfair", dataset_means(boxes, m.name, datasets), fmt_float(2))
               for m in fairness_methods]
    columns += [(f"{m.name}_emp_fair", dataset_means(emp, m.name, datasets), fmt_float(4))
                for m in fairness_methods]
    print("\nFairness metrics (mean)")
    print(summary_table(datasets, columns).to_string(index=False))


def print_accuracy_metrics(scores, already_fair, methods, datasets):
    """Accuracy and balanced accuracy, as the change vs. the original model.

    Same convention as LaTeX table 2: the reference column is the absolute score
    of the model being repaired, every other column is the fold-matched change
    in percentage points.
    """
    reference = METHODS[REFERENCE]
    columns = []
    for metric in ("acc", "bacc"):
        by_fold = scores[metric]
        for method in methods:
            if method is reference:
                values = [mean_std(reference_level(by_fold, already_fair, d))[0]
                          for d in datasets]
                columns.append((f"{method.name}_{metric}", values, fmt_points(2, suffix="%")))
            else:
                values = [mean_std(fold_deltas(by_fold, already_fair, method.name, d))[0]
                          for d in datasets]
                columns.append((f"{method.name}_Δ{metric}", values, fmt_points(2, signed=True)))

    print("\nAccuracy metrics (mean; Δ = change vs. original, pct. points)")
    print(summary_table(datasets, columns).to_string(index=False))


def print_model_sizes(df, methods, datasets):
    # NOTE: mixed basis. num_leaves = all 50 folds; enc%/mod% = the 36 repaired.
    leaves = fold_values(df, "num_leaves")
    encoded = fold_values(df, "leaves_encoded", mask="repair_needed")
    modified = fold_values(df, "leaves_modified", mask="repair_needed")
    total = fold_values(df, "leaves_total", mask="repair_needed")

    columns = [(m.name, dataset_means(leaves, m.name, datasets), fmt_float(1))
               for m in methods if m.kind != "milp"]
    # Encoded/modified leaves are reported as a fraction of the model's leaves.
    for method in (m for m in methods if m.kind == "milp"):
        for values, suffix in ((encoded, "enc%"), (modified, "mod%")):
            fractions = []
            for dataset in datasets:
                counts = values.get(method.name, {}).get(dataset, [])
                totals = total.get(method.name, {}).get(dataset, [])
                pairs = [c / t for c, t in zip(counts, totals) if t > 0]
                fractions.append(statistics.mean(pairs) if len(counts) == len(totals) and pairs else None)
            columns.append((f"{method.short}_{suffix}", fractions, fmt_percent(1)))

    print("\nModel size (mean leaves)")
    print(summary_table(datasets, columns).to_string(index=False))


def with_total_time(df):
    """`df` plus a `total_time` column = MILP solve time + veritas time.

    Summed per fold (not as a sum of per-dataset means) so the column is the
    mean end-to-end repair time. NaN only when both terms are missing.
    """
    parts = df[["time", "veritas_time"]].apply(pd.to_numeric, errors="coerce")
    return df.assign(total_time=parts.sum(axis=1, min_count=1))


def total_time_label(time_label):
    """'MILP Time' -> 'MILP Total'; falls back to appending ' Total'."""
    return time_label.replace("Time", "Total") if "Time" in time_label \
        else f"{time_label} Total"


# Previous version: baseline training times as extra columns, no total column.
# def print_runtimes(df, methods, datasets):
#     milp_methods = [m for m in methods if m.kind == "milp"]
#     baselines = [m for m in methods if m.kind == "baseline"]
#     iterations = fold_values(df, "iterations", mask="repair_needed")
#     times = fold_values(df, "time", mask="repair_needed")
#     veritas = fold_values(df, "veritas_time", mask="repair_needed")
#
#     columns = [(f"{m.name}_iters", dataset_means(iterations, m.name, datasets), fmt_float(1))
#                for m in milp_methods]
#     for method in milp_methods:
#         columns.append((f"{method.name}_time", dataset_means(times, method.name, datasets), fmt_runtime))
#         columns.append((f"{method.name}_fairness_time",
#                         dataset_means(veritas, method.name, datasets), fmt_runtime))
#     columns += [(f"{m.name}_time", dataset_means(times, m.name, datasets), fmt_runtime)
#                 for m in baselines]
#
#     print("\nRuntime and MILP iterations")
#     print(summary_table(datasets, columns).to_string(index=False))
#     return times
def print_runtimes(df, methods, datasets):
    """MILP variants only: iterations, MILP time, veritas time and their sum.

    Baseline training times are still returned (figure 3 plots them), they are
    just no longer printed here.
    """
    milp_methods = [m for m in methods if m.kind == "milp"]
    iterations = fold_values(df, "iterations", mask="repair_needed")
    times = fold_values(df, "time", mask="repair_needed")
    veritas = fold_values(df, "veritas_time", mask="repair_needed")
    totals = fold_values(with_total_time(df), "total_time", mask="repair_needed")

    columns = [(f"{m.name}_iters", dataset_means(iterations, m.name, datasets), fmt_float(1))
               for m in milp_methods]
    for method in milp_methods:
        columns.append((f"{method.name}_fairness_time",
                        dataset_means(veritas, method.name, datasets), fmt_seconds))
        columns.append((f"{method.name}_time", dataset_means(times, method.name, datasets), fmt_seconds))
        columns.append((f"{method.name}_total_time",
                        dataset_means(totals, method.name, datasets), fmt_seconds))

    print("\nRuntime (s) and MILP iterations (per-dataset mean; last row: median across datasets)")
    print(summary_table(datasets, columns, use_median=True).to_string(index=False))
    return times

# Runtimes including the pipeline costs (training + LOP compression)
def load_pipeline_costs(results):
    """-> {(dataset, fold): (train_time, lop_time)}, as logged by the orig and
    lop steps (None when a step reused a cached model without a recorded time)."""
    return {key: ((result.get("original_at") or {}).get("train_time"),
                  (result.get("compressed_at") or {}).get("lop_time"))
            for key, result in results.items()}


def with_pipeline_times(df, costs):
    """`df` plus `train_time`, `lop_time` and `pipeline_total_time`.

    The pipeline total is the per-fold sum of training, LOP compression, the
    veritas checks and the MILP solves. Folds without a recorded training / LOP
    time get NaN for those columns, which drops them from those cells (and from
    the total).
    """
    keys = list(zip(df["dataset"], df["fold"]))
    train = pd.Series([costs.get(k, (None, None))[0] for k in keys], index=df.index)
    lop = pd.Series([costs.get(k, (None, None))[1] for k in keys], index=df.index)
    out = df.assign(train_time=pd.to_numeric(train, errors="coerce"),
                    lop_time=pd.to_numeric(lop, errors="coerce"))
    parts = out[["time", "veritas_time", "train_time", "lop_time"]].apply(
        pd.to_numeric, errors="coerce")
    return out.assign(pipeline_total_time=parts.sum(axis=1, min_count=1))

def report_missing_costs(df, costs, datasets):
    """Warn about eligible folds that have no recorded training / LOP time."""
    folds = df[df["repair_needed"]][["dataset", "fold"]].drop_duplicates()
    missing = sorted({(d, f) for d, f in zip(folds["dataset"], folds["fold"])
                      if costs.get((d, f), (None, None))[0] is None
                      or costs.get((d, f), (None, None))[1] is None})
    if missing:
        print(f"  [!] {len(missing)} fold(s) without pipeline costs: "
              + ", ".join(f"{d}/{f}" for d, f in missing))
    return missing

def pipeline_columns(df, methods, datasets, costs, mt, stats=False, time_fmt=fmt_seconds):
    """Shared column spec of the two pipeline-cost tables.

    `stats=False` gives per-dataset means (terminal table), `stats=True` gives
    per-dataset (mean, std) pairs (LaTeX table). Columns follow the pipeline
    order: training, LOP compression, verification, MILP repair, total.
    Training and LOP compression are per-fold costs shared by every method, so
    they appear once rather than per MILP variant. `time_fmt` formats the time
    cells: plain seconds (`fmt_seconds`) or X.Ys / X.Ym / X.Yh (`fmt_runtime`).
    """
    aggregate = dataset_stats if stats else dataset_means
    milp_methods = [m for m in methods if m.kind == "milp"]
    full = with_pipeline_times(df, costs)

    iterations = fold_values(full, "iterations", mask="repair_needed")
    times = fold_values(full, "time", mask="repair_needed")
    veritas = fold_values(full, "veritas_time", mask="repair_needed")
    train = fold_values(full, "train_time", mask="repair_needed")
    lop = fold_values(full, "lop_time", mask="repair_needed")
    totals = fold_values(full, "pipeline_total_time", mask="repair_needed")

    columns = [(label, aggregate(iterations, m.name, datasets), fmt_float(1))
               for m in milp_methods for label in [m.rt_labels[0]]]

    # Same for every method: read off the first MILP variant.
    reference = milp_methods[0].name
    columns.append((f"{mt} Training", aggregate(train, reference, datasets), time_fmt))
    columns.append(("LOP Compression", aggregate(lop, reference, datasets), time_fmt))

    for method in milp_methods:
        _, time_label, veritas_label = method.rt_labels
        columns.append((veritas_label, aggregate(veritas, method.name, datasets), time_fmt))
        columns.append((time_label, aggregate(times, method.name, datasets), time_fmt))
        columns.append((total_time_label(time_label),
                        aggregate(totals, method.name, datasets), time_fmt))
    return columns

def print_runtimes_with_pipeline(df, methods, datasets, costs, mt, time_fmt=fmt_seconds):
    """Second runtime table: `print_runtimes` plus training and LOP times.

    The total column is the full end-to-end cost (training + compression +
    veritas + MILP), not just the repair. `time_fmt` as in `pipeline_columns`.
    """
    unit = "s" if time_fmt is fmt_seconds else "s/m/h"
    print(f"\nRuntime ({unit}) including pipeline costs (training + LOP compression); "
          "last row: median across datasets")
    if not costs:
        print("  (no training / LOP times recorded)")
        return
    report_missing_costs(df, costs, datasets)
    columns = pipeline_columns(df, methods, datasets, costs, mt, stats=False, time_fmt=time_fmt)
    print(summary_table(datasets, columns, use_median=True).to_string(index=False))


# LATEX tables below

def latex_fair_counts(fair_counts_df, methods, latex_datasets, mt):
    fairness_methods = [m for m in methods if m.kind != "reference"]
    by_dataset = fair_counts_df.set_index("dataset")
    rows = [(d, [by_dataset.at[d, f"{m.name}_fair"] for m in fairness_methods])
            for d in latex_datasets + ["TOTAL"]]
    latex_table("Fair fold counts (enumerable + unfair\\_boxes=0)",
                [m.head_long(mt) for m in fairness_methods], rows)


def latex_bacc_loss(bacc, already_fair, methods, latex_datasets, mt):
    """Balanced-accuracy change (Δ BAcc) vs. the original model.

    The reference column shows the absolute score; every other column shows the
    (signed) change. Folds where the compressed model was already fair are
    dropped for all methods, so all columns are compared on the same folds.
    """
    reference = METHODS[REFERENCE]

    def cell(method, dataset):
        if method is reference:
            return mean_std(reference_level(bacc, already_fair, dataset))
        return mean_std(fold_deltas(bacc, already_fair, method.name, dataset))

    per_dataset = {m.name: [] for m in methods}  # dataset means, for the AVERAGE row
    rows = []
    for dataset in latex_datasets:
        cells = []
        for method in methods:
            mean, std = cell(method, dataset)
            if mean is not None:
                per_dataset[method.name].append(mean)
            cells.append(fmt_pm(mean, std, fmt_float(2)))
        rows.append((dataset, cells))

    rows.append(("AVERAGE", [fmt_pm(*mean_std(per_dataset[m.name]), fmt_float(2))
                             for m in methods]))

    latex_table(f"$\\Delta$ balanced accuracy relative to original {mt} (\\%)",
                [m.head(mt) for m in methods], rows)


# Previous version: baseline training times as extra columns, no total column.
# def latex_runtimes(df, methods, latex_datasets):
#     """Runtime and MILP iterations. The AVERAGE row is the mean of the
#     per-dataset means (see the module docstring)."""
#     milp_methods = [m for m in methods if m.kind == "milp"]
#     baselines = [m for m in methods if m.kind == "baseline"]
#     iterations = fold_values(df, "iterations", mask="repair_needed")
#     times = fold_values(df, "time", mask="repair_needed")
#     veritas = fold_values(df, "veritas_time", mask="repair_needed")
#
#     # (header, per-dataset (mean, std), cell formatter)
#     columns = []
#     for method in milp_methods:
#         iter_label, time_label, veritas_label = method.rt_labels
#         columns.append((iter_label, dataset_stats(iterations, method.name, latex_datasets), fmt_float(1)))
#         columns.append((time_label, dataset_stats(times, method.name, latex_datasets), fmt_runtime))
#         columns.append((veritas_label, dataset_stats(veritas, method.name, latex_datasets), fmt_runtime))
#     for method in baselines:
#         columns.append((f"{method.label} Time", dataset_stats(times, method.name, latex_datasets), fmt_runtime))
#
#     rows = [(dataset, [fmt_pm(stats[i][0], stats[i][1], fmt) for _, stats, fmt in columns])
#             for i, dataset in enumerate(latex_datasets)]
#     average = []
#     for _, stats, fmt in columns:
#         mean, std = mean_std([m for m, _ in stats])
#         average.append(fmt_pm(mean, std, fmt))
#     rows.append(("AVERAGE", average))
#
#     latex_table("Training/repair runtime (seconds) and MILP iterations",
#                 [header for header, _, _ in columns], rows)


def runtime_rows(datasets, columns):
    """LaTeX rows of the runtime tables: per-dataset 'mean $\\pm$ std' cells
    plus a MEDIAN row, the median of the per-dataset means (no $\\pm$).
    `columns` is a list of (header, per-dataset (mean, std), formatter)."""
    rows = [(dataset, [fmt_pm(stats[i][0], stats[i][1], fmt) for _, stats, fmt in columns])
            for i, dataset in enumerate(datasets)]
    summary = []
    for _, stats, fmt in columns:
        center = median([m for m, _ in stats])
        summary.append("---" if center is None else fmt(center))
    rows.append(("MEDIAN", summary))
    # a bare '<' renders as an inverted exclamation mark in LaTeX text mode
    return [(label, [cell.replace("<", "$<$") for cell in cells]) for label, cells in rows]


def latex_runtimes(df, methods, latex_datasets):
    """MILP repair runtime and iterations: no baseline columns, and a last
    column with the per-fold sum of the MILP and veritas times. Cells are
    per-dataset mean $\\pm$ std; the MEDIAN row is the median of the
    per-dataset means."""
    milp_methods = [m for m in methods if m.kind == "milp"]
    iterations = fold_values(df, "iterations", mask="repair_needed")
    times = fold_values(df, "time", mask="repair_needed")
    veritas = fold_values(df, "veritas_time", mask="repair_needed")
    totals = fold_values(with_total_time(df), "total_time", mask="repair_needed")

    # (header, per-dataset (mean, std), cell formatter)
    columns = []
    for method in milp_methods:
        iter_label, time_label, veritas_label = method.rt_labels
        columns.append((iter_label, dataset_stats(iterations, method.name, latex_datasets), fmt_float(1)))
        columns.append((veritas_label, dataset_stats(veritas, method.name, latex_datasets), fmt_seconds))
        columns.append((time_label, dataset_stats(times, method.name, latex_datasets), fmt_seconds))
        columns.append((total_time_label(time_label),
                        dataset_stats(totals, method.name, latex_datasets), fmt_seconds))

    latex_table("MILP repair runtime (seconds) and MILP iterations",
                [header for header, _, _ in columns], runtime_rows(latex_datasets, columns))


def print_pooled_statistics(df, methods):
    """Per-fold (not per-dataset) counts, for the prose in the paper."""
    iterations = fold_values(df, "iterations", mask="repair_needed")
    times = fold_values(df, "time", mask="repair_needed")
    totals = fold_values(with_total_time(df), "total_time", mask="repair_needed")

    def print_buckets(label, values):
        print(f"{label}: {sum(t < 60 for t in values)}/{len(values)} < 1min, "
              f"{sum(t < 600 for t in values)}/{len(values)} < 10min, "
              f"median = {fmt_runtime(statistics.median(values))}")

    print("\n=== Runtime Statistics (per fold, folds that needed a repair) ===")
    for method in methods:
        if method.kind not in ("milp", "baseline"):
            continue
        label, name = method.label, method.name
        is_milp = method.kind == "milp"
        values = pooled(times, name)
        if values:
            print_buckets(f"{label} (solve only)" if is_milp else label, values)
        # The line above is the Gurobi time alone; the repair loop also pays for
        # a veritas enumeration at every iteration.
        if is_milp:
            loop = pooled(totals, name)
            if loop:
                print_buckets(f"{label} (loop: veritas + MILP)", loop)

    milp_iters = pooled(iterations, "milp_composite")
    if milp_iters:
        n = len(milp_iters)
        print(f"\nMILP Iterations: {sum(i > 2 for i in milp_iters)}/{n} > 2 iters, "
              f"{sum(i > 5 for i in milp_iters)}/{n} > 5 iters, "
              f"{sum(i > 10 for i in milp_iters)}/{n} > 10 iters, max = {max(milp_iters):g}")
        # mean alongside the median of the runtime tables: macro = over the per-dataset
        # means (as in the tables), pooled = over all folds
        ds_means = [statistics.mean(v) for v in iterations.get("milp_composite", {}).values() if v]
        print(f"MILP Iterations: mean = {statistics.mean(ds_means):.2f} (macro, over datasets; "
              f"median {statistics.median(ds_means):.2f}), "
              f"{statistics.mean(milp_iters):.2f} (pooled, over {n} folds; "
              f"median {statistics.median(milp_iters):g})")


def latex_runtimes_with_pipeline(df, methods, latex_datasets, costs, mt, time_fmt=fmt_seconds):
    """LaTeX version of the table above (mean $\\pm$ std across folds, MEDIAN row)."""
    if not costs:
        return
    columns = pipeline_columns(df, methods, latex_datasets, costs, mt, stats=True,
                               time_fmt=time_fmt)
    unit = "seconds" if time_fmt is fmt_seconds else "s/m/h"
    latex_table(f"End-to-end runtime ({unit}): model training, LOP compression, "
                "fairness verification and MILP repair",
                [header for header, _, _ in columns], runtime_rows(latex_datasets, columns))

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def whisker_range(series):
    """Span covered by the whiskers of every box (matplotlib's 1.5*IQR rule).

    Fliers are hidden, so the axis only has to reach the whisker ends.
    """
    lows, highs = [], []
    for values in series:
        s = pd.Series(values)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lows.append(s[s >= q1 - 1.5 * iqr].min())
        highs.append(s[s <= q3 + 1.5 * iqr].max())
    return min(lows), max(highs)


def make_figures(df, methods, datasets, fair_totals, times, bacc, already_fair,
                 model_type, suffix):
    """Summary figures: verified-fair share and Δ balanced accuracy
    (the repair/training runtime figure is disabled)."""
    plot_methods = [m for m in methods if m.kind != "reference"]
    n_folds = df[["dataset", "fold"]].drop_duplicates().shape[0]

    set_style()

    # -- Figure 1: share of models that could be verified fair ---------------
    # Denominator is every fold, including the ones veritas could not enumerate.
    fig1, ax = plt.subplots(figsize=(6.5, 5))
    labels = [m.plot_label for m in plot_methods]
    counts = [fair_totals[m.name][0] for m in plot_methods]
    percents = [c / max(n_folds, 1) * 100.0 for c in counts]
    bars = ax.bar(labels, percents, color=[m.color for m in plot_methods],
                  edgecolor="none", linewidth=0)
    y_top = max(100.0, (max(percents) if percents else 0.0) + 16.0)
    ax.set_ylim(0, y_top)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("% models verified fair")
    ax.set_title("Provably fair models")
    for rect, count, pct in zip(bars, counts, percents):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + y_top * 0.03,
                f"{count}/{n_folds}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=ANNOT_SIZE)
    ax.axhline(100.0, color=C_REF, linestyle="--", linewidth=0.8)
    style_axes(ax, grid="y")
    fig1.tight_layout(pad=0.8)

    # -- Figure 2: per-fold balanced-accuracy change vs. the original ---------
    # One point per fold, pooled across datasets (the tables average per dataset
    # first, then across datasets).
    fig2, ax = plt.subplots(figsize=(6.5, 5))
    box_cols, palette = {}, []
    for method in plot_methods:
        deltas = [d for dataset in datasets
                  for d in fold_deltas(bacc, already_fair, method.name, dataset)]
        if deltas:
            box_cols[method.plot_label] = deltas
            palette.append(method.color)
    if box_cols:
        sns.boxplot(data=pd.DataFrame({k: pd.Series(v) for k, v in box_cols.items()}),
                    palette=palette, saturation=1, linecolor=C_BOX_LINE,
                    ax=ax, showfliers=False)
        ax.axhline(0, color=C_REF, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Δ balanced accuracy")
        ax.set_title(f"Δ balanced accuracy vs {model_type.upper()}")
        # Fit the axis to the whiskers actually drawn: xgb reaches ~-22 but rf
        # only ~-16, so a shared limit would waste a third of the rf panel.
        lo, hi = whisker_range(box_cols.values())
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([t for t in (0, -2.5, -5, -10, -20) if lo - pad <= t <= hi + pad])
    style_axes(ax, grid="y")
    fig2.tight_layout(pad=0.8)

    # -- Figure 3: per-fold repair/training runtime --------------------------
    # fig3, ax = plt.subplots(figsize=(6.5, 5))
    # box_cols, palette = {}, []
    # for method in plot_methods:
    #     if method.kind == "lop":  # LOP has no repair/training runtime of its own
    #         continue
    #     values = pooled(times, method.name)
    #     if values:
    #         box_cols[method.plot_label] = values
    #         palette.append(method.color)
    # if box_cols:
    #     sns.boxplot(data=pd.DataFrame({k: pd.Series(v) for k, v in box_cols.items()}),
    #                 palette=palette, saturation=1, ax=ax)
    #     ax.set_yscale("log")
    #     ax.set_ylabel("Runtime (seconds)", fontsize=18)
    #     ax.set_title("Repair/Training Runtime", pad=12, fontweight="bold", fontsize=17)
    #     ax.tick_params(axis="both", labelsize=15)
    # style_axes(ax)
    # fig3.tight_layout(pad=0.8)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fig, name in [(fig1, "summary_plot1_fair_folds"),
                      (fig2, "summary_plot2_bacc_loss")]:
                    #   (fig3, "summary_plot3_runtime")]:
        # side-by-side panels in the paper: no tight bbox (see plot_style.save)
        for path in save(fig, f"{FIGURES_DIR}/{name}_{model_type}{suffix}", tight=False):
            print(f"Saved summary plot to: {path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize fairness repair results across datasets and folds.")
    parser.add_argument("--model_type", "-m", default="xgb", choices=["xgb", "rf"],
                        help="Model type to summarize.")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Skip the summary figures.")
    return parser.parse_args()


def report_missing_baselines(results, methods):
    """Warn about folds without a result file for a baseline."""
    for method in (m for m in methods if m.kind == "baseline"):
        missing = sorted(k for k, r in results.items() if method.block not in r)
        if missing:
            print(f"  [!] {method.label}: no results for {len(missing)} fold(s): "
                  + ", ".join(f"{d}/{f}" for d, f in missing))


def main():
    args = parse_args()
    model_type = args.model_type
    mt = model_type.upper()
    methods = select_methods()
    suffix = ""

    print(f"\n{'=' * 80}")
    print(f"Summarizing {mt} model results")
    print(f"{'=' * 80}\n")

    results = load_results(RESULTS_DIR, model_type)
    if not results:
        print(f"No results found in {RESULTS_DIR}.")
        return
    report_missing_baselines(results, methods)

    df = build_frame(results, methods)
    datasets = sorted({dataset for dataset, _ in results})
    latex_datasets = [d for d in DATASET_ORDER if d in datasets]

    # Fold-matched scores, shared by the accuracy table, LaTeX table 2 and fig. 2.
    already_fair = already_fair_folds(df)
    scores = {"acc": fold_scores(df, "test_acc"), "bacc": fold_scores(df, "test_bacc")}
    bacc = scores["bacc"]

    fair_counts_df, fair_totals = print_fair_counts(df, methods, datasets)
    print_fairness_metrics(df, methods, datasets)
    print_accuracy_metrics(scores, already_fair, methods, datasets)
    print_model_sizes(df, methods, datasets)
    times = print_runtimes(df, methods, datasets)
    pipeline_costs = load_pipeline_costs(results)
    print_runtimes_with_pipeline(df, methods, datasets, pipeline_costs, mt)
    print_runtimes_with_pipeline(df, methods, datasets, pipeline_costs, mt, time_fmt=fmt_runtime)

    print("\n" + "=" * 80)
    print("LATEX TABLES")
    print("=" * 80)

    latex_fair_counts(fair_counts_df, methods, latex_datasets, mt)
    latex_bacc_loss(bacc, already_fair, methods, latex_datasets, mt)
    latex_runtimes(df, methods, latex_datasets)
    latex_runtimes_with_pipeline(df, methods, latex_datasets, pipeline_costs, mt)
    latex_runtimes_with_pipeline(df, methods, latex_datasets, pipeline_costs, mt,
                                 time_fmt=fmt_runtime)
    print_pooled_statistics(df, methods)

    print()
    if args.plot:
        try:
            make_figures(df, methods, datasets, fair_totals, times, bacc, already_fair,
                         model_type, suffix)
        except Exception as exc:
            print(f"\nCould not create plots: {exc}")

    # Folds veritas could not enumerate are excluded from every MILP-based number.
    skipped = (df[~df["compressed_ok"]][["dataset", "fold"]].drop_duplicates()
               .groupby("dataset").size())
    if skipped.sum() > 0:
        print("\nSkipped folds (not enumerable)")
        for dataset, count in skipped.items():
            print(f"  {dataset}: {count}")
        print(f"Total skipped folds: {skipped.sum()}")


if __name__ == "__main__":
    main()
