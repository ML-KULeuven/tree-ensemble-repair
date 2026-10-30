"""
Generalization study for the MILP fairness repair.

Question: when the MILP is given only a SUBSET of the counterexamples, how many of
the counterexamples it never saw are repaired as a side effect?

Protocol (per dataset x fold):
  1. Load the LOP-compressed model of the main experiment from the cache
     (run `main.py -e orig --cache` and `main.py -e lop --cache` first).
  2. Enumerate ALL unfair boxes with veritas -> the full counterexample set C.
  3. For an increasing fraction p (10%, 25%, 50%, 75%, 100%):
       - split C into a repair set R (|R| = p|C|) and a held-out set H = C \\ R,
       - run ONE MILP solve on R only (no iterative loop: re-verifying would
         rediscover the held-out counterexamples and contaminate the measurement),
       - measure how many pairs in H are fixed by the repaired model.
     Splits are nested: R(10%) subset of R(25%) subset of ... (one random
     permutation per fold, seeded with SEED + fold).
     The 100% point has no held-out set: it anchors the runtime and the accuracy
     cost of the complete repair.
  4. Also report the accuracy cost of each partial repair.

Every unfair box veritas returns is a distinct OC pair, so no held-out pair is
fixed for free by sharing its OC pair with a repaired one (checked at startup).

The MILP uses the composite objective with the main experiment's LAMBDA_L1
(i.e., the same repair as `main.py -e milp`), with a lower per-solve time limit.

Results -> {CACHE_DIR}/results/generalization/{dname}_{model_type}_{fold}.json

Usage:
    python main_generalization.py <dataset> --fold <0-4> -m xgb
"""

import os
import json
import time

import click
import numpy as np
import veritas

from utils import (
    SEED,
    SENSITIVE_COLS,
    CACHE_DIR,
    get_dataset,
    get_model_path,
    remap_sensitive_to_01,
    get_ocs_batch,
    collect_unique_OCs,
    collect_unique_OCs_pairs,
    extract_pairs_batch,
    evaluate_accuracies,
)
from fairness import run_fairness_check, evaluate_empirical_fairness
from leaf_refinement_fairness import fix_fairness
from leaf_refinement_core import build_W, _collect_touched_leaves, LAMBDA_L1

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# repair-set fractions
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
OUT_DIR = f"{CACHE_DIR}/results/generalization"

# Folds with fewer counterexamples than this are skipped
MIN_COUNTEREXAMPLES = 10

# Per-solve Gurobi cap, lower than leaf_refinement_core.GUROBI_TIMEOUT: 
# this study runs one solve per fraction, and the largest datasets (ACSIncome,
# ACSPublicCoverage, MEPSPanel21) otherwise burn hours on a single fraction.
MILP_TIMEOUT = 600


# ----------------------------- helpers -----------------------------

def _save_json(log, path):
    """Write the result file atomically.

    json.dump straight into `path` truncates it on open, so a full disk or a killed
    job leaves a 0-byte result behind (and destroys any previous one). Dumping to a
    temp file in the same directory and renaming makes the update all-or-nothing.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _agreement_mask(at, X_pairs):
    """For X_pairs = [inst0_0, inst1_0, inst0_1, inst1_1, ...], return a boolean
    array over pairs: True if both instances get the same predicted label."""
    scores = np.asarray(at.eval(X_pairs)).ravel()
    labels = scores >= 0.0
    return labels[0::2] == labels[1::2]


# ----------------------------- experiment -----------------------------

def run_split(
    at_compr, X_pairs, pairs, repair_idx, holdout_idx,
    train_info, dtrain, dtest, sensitive_col, W_original,
):
    """One MILP solve on `repair_idx`, then evaluate on `holdout_idx`."""
    repair_pairs = [pairs[i] for i in repair_idx]

    rec = {
        "n_repair": int(len(repair_idx)),
        "n_holdout": int(len(holdout_idx)),
    }

    t0 = time.time()
    status, at_repaired, __ = fix_fairness(
        at_compr.copy(), repair_pairs, train_info,
        objective="composite", lambda_l1=LAMBDA_L1, verbose=True,
    )
    rec["milp_time"] = time.time() - t0
    rec["status"] = status
    print(f"\n  Optimization status: {status} (took {rec['milp_time']:.3f} s)")

    if at_repaired is None:
        print(f"  !! MILP {status}: no repaired model for this split.")
        return rec, None

    # 1. which counterexample pairs are fixed?
    fixed = _agreement_mask(at_repaired, X_pairs)
    rec["repair_fixed"] = int(fixed[repair_idx].sum())
    rec["holdout_fixed"] = int(fixed[holdout_idx].sum()) if len(holdout_idx) else 0
    rec["holdout_fixed_frac"] = (
        rec["holdout_fixed"] / len(holdout_idx) if len(holdout_idx) else None
    )
    if rec["repair_fixed"] != len(repair_idx):
        # hard constraints -> should never happen; keep the run alive but flag it
        print(f"  (!) WARNING: only {rec['repair_fixed']}/{len(repair_idx)} "
              f"repair-set pairs are fixed!")

    # 2. accuracy cost of the partial repair
    acc_train, acc_test, bacc_train, bacc_test = evaluate_accuracies(
        at_repaired, dtrain, dtest, label=f"Repaired ({len(repair_idx)} pairs)"
    )
    rec.update({
        "train_acc": acc_train, "test_acc": acc_test,
        "train_bacc": bacc_train, "test_bacc": bacc_test,
        "emp_fairness": evaluate_empirical_fairness(at_repaired, dtest, sensitive_col),
    })

    # 3. how much of the model was touched / changed
    W_final = build_W(at_repaired)
    rec["leaves_total"] = int(W_final.shape[1])
    rec["leaves_encoded"] = int(len(_collect_touched_leaves(repair_pairs)))
    rec["leaves_modified"] = int(np.sum(np.abs(W_final[0, :] - W_original[0, :]) > 1e-8))

    print(f"\n  >> Repair set:  {rec['repair_fixed']}/{rec['n_repair']} fixed")
    if len(holdout_idx):
        print(f"  >> Held out:    {rec['holdout_fixed']}/{rec['n_holdout']} fixed "
              f"({100 * rec['holdout_fixed_frac']:.1f}%)")

    return rec, at_repaired


@click.command()
@click.argument("dname")
@click.option("--fold", default=0, type=int)
@click.option("--model_type", "-m", default="xgb", type=click.Choice(["xgb", "rf"]))
def main(dname, fold, model_type):

    t_start_script = time.time()

    # _solve_linear_problem reads GUROBI_TIMEOUT at call time, so overriding the
    # module attribute caps every solve of this run
    # (a capped solve keeps its best feasible solution, status 'suboptimal',
    #  which still satisfies every repair-set constraint)
    import leaf_refinement_core
    leaf_refinement_core.GUROBI_TIMEOUT = MILP_TIMEOUT
    print(f" * Gurobi time limit per solve: {MILP_TIMEOUT:.0f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    res_file = f"{OUT_DIR}/{dname}_{model_type}_{fold}.json"

    # ------------------------- data + model -------------------------
    d, dtrain, dvalid, dtest = get_dataset(dname, SEED, fold=fold, silent=True)

    sensitive_col = SENSITIVE_COLS[dname]
    if isinstance(sensitive_col, list):
        ordinal_cols = [d.X.columns.get_loc(c) for c in sensitive_col]
    else:
        ordinal_cols = [d.X.columns.get_loc(sensitive_col)]
    remap_sensitive_to_01(sensitive_col, d, dtrain, dvalid, dtest)

    log = {
        "dataset": dname, "fold": fold, "model_type": model_type,
        "lambda_l1": LAMBDA_L1, "fractions": list(FRACTIONS),
        "split_seed": SEED + fold, "milp_timeout": MILP_TIMEOUT,
    }

    # reuse the original and LOP-compressed models' metrics from the main experiment
    for step, block in (("orig", "original_at"), ("lop", "compressed_at")):
        step_file = f"{CACHE_DIR}/results/{dname}_{model_type}_{fold}_{step}.json"
        if not os.path.exists(step_file):
            raise SystemExit(f"{step_file} not found -- run `main.py {dname} --fold {fold} "
                             f"-m {model_type} -e {step} --cache` first.")
        with open(step_file) as f:
            log[block] = json.load(f)[block]

    at_compr = veritas.AddTree.read(
        get_model_path(dname, fold, mtype=model_type, compressed=True)
    )

    W_original = build_W(at_compr).copy()
    print(f"\n >> Model size: {len(at_compr)} trees, {W_original.shape[1]} leaves")
    evaluate_accuracies(at_compr, dtrain, dtest, label="Initial")

    # training-set OCs: needed by the accuracy term of the MILP objective
    t_oc = time.time()
    OCs_train = collect_unique_OCs(get_ocs_batch(dtrain.X, at_compr), dtrain.y)
    print(f"\n >> Extracted {len(OCs_train)} unique OCs from the "
          f"{len(dtrain.X)} training examples in {time.time() - t_oc:.3f} s.")
    train_info = {"X_train": dtrain.X, "y_train": dtrain.y, "OCs_train": OCs_train}

    # --------------- full counterexample set (veritas) ---------------
    print("\n ================ Enumerating all counterexamples ================ \n")
    t_fair = time.time()
    solutions, feat_map, emp_fairness_original, enumerable = run_fairness_check(
        at_compr, d, dtest, sensitive_col, ordinal_cols
    )
    log["fairness_time"] = time.time() - t_fair
    log["compressed_at"]["#unfair_boxes"] = len(solutions)
    log["compressed_at"]["emp_fairness"] = emp_fairness_original
    log["compressed_at"]["enumerable"] = enumerable
    print(f"\n * Found {len(solutions)} unfair boxes "
          f"(enumerable: {enumerable}) in {log['fairness_time']:.1f}s")

    if not enumerable or len(solutions) < MIN_COUNTEREXAMPLES:
        # incomplete counterexample set, or too few counterexamples to split
        log["n_counterexamples"] = len(solutions)
        log["status"] = ("not_enumerable" if not enumerable
                         else "fair" if len(solutions) == 0 else "too_few_counterexamples")
        print(f"\n * {log['status']} ({len(solutions)} counterexamples): "
              f"nothing meaningful to split. Exiting.")
        log["runtime"] = time.time() - t_start_script
        _save_json(log, res_file)
        print(f"\n✓ Results saved to {res_file}")
        return

    # counterexample pairs -> concrete instances -> OCs
    X_pairs = extract_pairs_batch(solutions, feat_map, d.X.columns, ordinal_cols)
    phi_pairs = get_ocs_batch(X_pairs, at_compr)
    pairs = [(phi_pairs[2 * i, :], phi_pairs[2 * i + 1, :]) for i in range(len(solutions))]

    n = len(pairs)
    log["n_counterexamples"] = n
    # veritas returns disjoint boxes of the contrast tree, i.e. distinct OC pairs
    # (the >0 output filter also fixes their orientation), so no held-out pair can
    # be repaired "for free" by sharing its OC pair with a repair-set one
    log["n_unique_oc_pairs"] = len(collect_unique_OCs_pairs(pairs))
    if log["n_unique_oc_pairs"] != n:
        print(f"\n (!) WARNING: {n} boxes map to only {log['n_unique_oc_pairs']} unique "
              f"OC pairs — held-out pairs sharing an OC pair with the repair set are "
              f"fixed by construction, so the reported percentages are optimistic.")
    # sanity check: on the original model every pair must be a genuine counterexample
    still_unfair = int((~_agreement_mask(at_compr, X_pairs)).sum())
    log["n_unfair_before"] = still_unfair
    print(f"\n >> {n} counterexample pairs ({log['n_unique_oc_pairs']} unique OC pairs); "
          f"{still_unfair}/{n} confirmed unfair on the compressed model.")

    # ----------------------- generalization sweep -----------------------
    rng = np.random.default_rng(SEED + fold)
    perm = rng.permutation(n)          # one permutation -> nested splits
    records = []
    for frac in FRACTIONS:
        k = int(np.clip(round(frac * n), 1, n))
        repair_idx, holdout_idx = perm[:k], perm[k:]
        print(f"\n{'=' * 78}\n"
              f" Repair fraction {frac:.0%}: "
              f"{k} repair / {n - k} held-out counterexamples\n{'=' * 78}")

        rec, __ = run_split(
            at_compr, X_pairs, pairs, repair_idx, holdout_idx,
            train_info, dtrain, dtest, sensitive_col, W_original,
        )
        rec["fraction"] = frac
        records.append(rec)

    log["generalization"] = records
    log["runtime"] = time.time() - t_start_script

    _save_json(log, res_file)
    print(f"\n✓ Results saved to {res_file}")
    print(f"✓ Total runtime: {log['runtime']:.1f}s\n")

    print_summary(log)


def print_summary(log):
    """Per-fraction table."""
    records = log.get("generalization", [])
    if not records:
        return
    n = log["n_counterexamples"]
    base_bacc = log.get("compressed_at", {}).get("test_bacc", float("nan"))

    print(f" ===== GENERALIZATION SUMMARY: {log['dataset']}, fold {log['fold']} ===== ")
    print(f" {n} counterexamples ({log['n_unique_oc_pairs']} unique OC pairs), "
          f"lambda={log['lambda_l1']}")
    print(f"\n {'frac':>6} {'|R|':>7} {'|H|':>7} {'H fixed':>16} "
          f"{'Δ test bacc':>12} {'MILP (s)':>10} {'status':>12}")
    print(" " + "-" * 78)

    for frac in log["fractions"]:
        recs = [r for r in records if r["fraction"] == frac]
        if not recs:
            continue
        ok = [r for r in recs if r.get("holdout_fixed") is not None]
        n_rep = np.mean([r["n_repair"] for r in recs])
        n_hold = np.mean([r["n_holdout"] for r in recs])

        def _pct(key_num, key_den):
            vals = [r[key_num] / r[key_den] for r in ok if r.get(key_den)]
            return f"{100 * np.mean(vals):.1f}%" if vals else "-"

        hold_s = _pct("holdout_fixed", "n_holdout") if n_hold else "-"
        dbacc = ([r["test_bacc"] - base_bacc for r in ok if "test_bacc" in r] or [float("nan")])
        milp = np.mean([r["milp_time"] for r in recs])
        statuses = ",".join(sorted({r["status"] for r in recs}))

        print(f" {frac:>6.0%} {n_rep:>7.0f} {n_hold:>7.0f} {hold_s:>16} "
              f"{100 * np.mean(dbacc):>11.2f}pp {milp:>10.1f} {statuses:>12}")
    print(" " + "=" * 78)


if __name__ == "__main__":
    main()
