"""
Ablation studies for the MILP fairness repair loop.

Experiment 1 (--experiment 1): sensitivity of composite objective wrt lambda (i.e., regularization strength)
  composite_lam000001 : lambda_l1 = 0.00001
  composite_lam00001  : lambda_l1 = 0.0001
  composite_lam0001   : lambda_l1 = 0.001
  composite_lam001    : lambda_l1 = 0.01
  composite_lam01     : lambda_l1 = 0.1
  composite_lam1      : lambda_l1 = 1.0
  composite_lam10     : lambda_l1 = 10.0
  -> {CACHE_DIR}/results/ablation1/{dname}_{model_type}_{fold}.json

Experiment 2 (--experiment 2): compare composite objective with accuracy-only and L1-only variants
  composite_acc_only : lambda_l1 = 0.0        (composite, accuracy term only)
  composite          : lambda_l1 = LAMBDA_L1  (full composite, same as main experiment)
  l1                 : pure L1 objective
  -> {CACHE_DIR}/results/ablation2/{dname}_{model_type}_{fold}.json

Both reuse the original and LOP-compressed models of the main experiment:
run `main.py -e orig --cache` and `main.py -e lop --cache` first
for the same dataset, fold and model type.
"""

import os
import click
import json
import time

import veritas
from utils import (
    get_dataset, SEED, SENSITIVE_COLS, CACHE_DIR, remap_sensitive_to_01, get_model_path,
)
from leaf_refinement_core import LAMBDA_L1
from main import fairness_repair_loop


EXPERIMENTS = {
    1: [
        (1e-5,  "composite", "composite_lam000001"),
        (1e-4,  "composite", "composite_lam00001"),
        (0.001, "composite", "composite_lam0001"),
        (0.01,  "composite", "composite_lam001"),
        (0.1,   "composite", "composite_lam01"),
        (1.0,   "composite", "composite_lam1"),
        (10.0,  "composite", "composite_lam10"),
    ],
    2: [
        (0.0,       "composite", "composite_acc_only"),
        (LAMBDA_L1, "composite", "composite"),
        (None,      "l1",        "l1"),
    ],
}


def _run_variant(d, dtrain, dtest, at_compr, sensitive_col, ordinal_cols, lambda_l1, objective):
    loop_log = {}
    kwargs = dict(objective=objective, log=loop_log)
    if objective == "composite":
        kwargs["lambda_l1"] = lambda_l1
    fairness_repair_loop(d, dtrain, dtest, at_compr.copy(), sensitive_col, ordinal_cols, **kwargs)
    return loop_log.get(f"{objective}_repaired_at", {})


@click.command()
@click.argument("dname")
@click.option("--fold", default=0, type=int)
@click.option("--model_type", "-m", default="xgb", type=click.Choice(["xgb", "rf"]))
@click.option("--experiment", "-e", default=1, type=click.Choice(["1", "2"]))
@click.option("--cache", default=False, is_flag=True)
def main(dname, fold, model_type, experiment, cache):

    exp = int(experiment)
    t_start_script = time.time()

    d, dtrain, dvalid, dtest = get_dataset(dname, SEED, fold=fold, silent=True)

    out_dir = f"{CACHE_DIR}/results/ablation{exp}"
    os.makedirs(out_dir, exist_ok=True)
    res_file = f"{out_dir}/{dname}_{model_type}_{fold}.json"

    sensitive_col = SENSITIVE_COLS[dname]
    ordinal_cols = [d.X.columns.get_loc(sensitive_col)]
    remap_sensitive_to_01(sensitive_col, d, dtrain, dvalid, dtest)

    # reuse the original and LOP-compressed models' metrics from the main experiment
    log = {}
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

    # skip the (expensive) repair variants if there is nothing to repair
    compr = log["compressed_at"]
    if compr.get("enumerable") is False or compr.get("#unfair_boxes") == 0:
        reason = "not enumerable" if compr.get("enumerable") is False else "already fair"
        print(f"\n * Compressed model is {reason}: skipping ablation {exp}.")
    else:
        for lambda_l1, objective, variant_key in EXPERIMENTS[exp]:
            print(f"\n ===== Ablation {exp}: {variant_key} (objective={objective}, "
                  f"lambda_l1={lambda_l1}) ===== \n")
            log[variant_key] = _run_variant(
                d, dtrain, dtest, at_compr, sensitive_col, ordinal_cols, lambda_l1, objective,
            )

    log["runtime"] = time.time() - t_start_script
    if cache:
        with open(res_file, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n✓ Results saved to {res_file}")
    print(f"✓ Total runtime: {log['runtime']:.1f}s\n")

    print(f" ===== ABLATION {exp} SUMMARY: {dname}, fold {fold} ===== ")
    print(f"Compressed AT: {compr.get('#unfair_boxes', 'N/A')} unfair boxes")
    for _, _, variant_key in EXPERIMENTS[exp]:
        v = log.get(variant_key, {})
        print(f"\n  [{variant_key}]")
        print(f"    Status:             {v.get('status', 'N/A')}")
        print(f"    Iterations:         {v.get('iterations', 'N/A')}")
        print(f"    Test bacc:          {v.get('test_bacc', float('nan')):.4f}")
        print(f"    Unfair boxes:       {v.get('#unfair_boxes', 'N/A')}")
        print(f"    Empirical fairness: {v.get('emp_fairness', float('nan')):.4f}")
        print(f"    MILP time (s):      {v.get('milp_time', float('nan')):.1f}")
    print(" ==============================================")


if __name__ == "__main__":
    main()
