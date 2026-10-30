import click
import json
import numpy as np
import os
import sys
import time

from utils import * 
from fairness import run_fairness_check
from leaf_refinement_fairness import LAMBDA_L1

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Try to generate a tree ensemble that satisfies 
# individual fairness constraints.
#
# Methods:
#
# 1. [Ours] MILP repair: 
#   - train XGB/RF ensemble, 
#   - compress it with LOP (to ensure enumerable OC-space), 
#   - run iterative MILP repair loop to fix unfair pairs of OCs
#     (composite objective: accuracy preservation + L1 penalty)
#
# 2. [Baseline 1] FATT (Ranzato & Zanella, CIKM 2021): 
#    Individual fairness-aware single decision tree training
#
# 3. [Baseline 2] BuDRO (Vargo et al., ICLR 2021):
#    Individual fairness-aware GBDT training with distributionally
#    robust optimization (DRO).
#
# 4. [Baseline 3] FairGBM (Cruz et al., ICLR 2023): 
#    Group fairness-aware GBDT training with FPR/FNR constraints.
#    [dropped: does not really address individual fairness]
# ---------------------------------------------------------------------------
# Further baseline ideas:
#
# 5. Rectify with PyXAI 
#     Note: only supports RF, no GBDT yet. 
#     Problems:
#        - we cannot compress a RF with LOP and then go back to sklearn 
#          (which we need to use PyXAI)
#        - even if we could:   
#              * fixes only one input at a time
#              * requires specifying the 'correct' class of a pair 
#                (which is unknown)
# ---------------------------------------------------------------------------


@click.command()
@click.argument("dname")
@click.option("--fold", "-f", default=0, type=int)
@click.option("--model_type", "-m", default="xgb", type=click.Choice(["xgb", "rf"]))
@click.option(
    "--exp", "-e", default="milp",
    type=click.Choice(["orig", "lop", "milp", "milp-b", "fatt", "budro"]) # "fairgbm"
)
@click.option("--verify", default=False, is_flag=True) # try verifying original XGB/RF
@click.option("--cache", default=False, is_flag=True)
def main(dname, fold, model_type, exp, verify, cache):
    """
    Wrapper to run each method separately.

    Pipeline order: orig -> lop -> milp / milp-b (-> orig --verify).
    fatt and budro are independent baselines that train a model from scratch.
    """
    if verify and exp != "orig":
        raise click.UsageError("--verify is only supported with -e orig")

    # Load & transform data
    t_start_script = time.time()
    d, dtrain, dvalid, dtest = get_dataset(dname, SEED, fold=fold, silent=True)
    # extract protected attribute
    sensitive_col = SENSITIVE_COLS[dname]
    # extract index of sensitive column (for veritas featmap)
    sensitive_col_idx = [d.X.columns.get_loc(sensitive_col)]
    remap_sensitive_to_01(sensitive_col, d, dtrain, dvalid, dtest)

    mt = f"_{model_type}" if exp in ("orig", "lop", "milp", "milp-b") else ""
    res_file = f"{CACHE_DIR}/results/{dname}{mt}_{fold}_{exp}.json"
    log = {}

    if exp == "orig" and not verify:
        # 0. Train original XGB/RF model, exporting it to cache
        at_orig = train_model(
            dname, dtrain, dtest, model_type, fold, cache=cache, log=log
        )

    elif exp == "orig" and verify:
        # 0b. (final pass) try to verify the (cached) original model:
        #     only adds verification info to its existing results
        if not os.path.exists(res_file):
            raise SystemExit(f"{res_file} not found -- run `-e orig --cache` first.")
        with open(res_file) as f:
            log = json.load(f)
        at_orig = veritas.AddTree.read(
            get_model_path(dname, fold, mtype=model_type, compressed=False)
        )

        t_verify = time.time()
        sols, __, ef, enumerable = run_fairness_check(
            at_orig, d, dtest, sensitive_col, sensitive_col_idx
        )
        log["original_at"].update({
            "#unfair_boxes": len(sols),
            "emp_fairness": ef,
            "enumerable": enumerable,
            "verify_time": time.time() - t_verify,
        })

    elif exp == "lop":
        # 1. Compress the cached original model with LOP, exporting it to cache
        orig_path = get_model_path(dname, fold, mtype=model_type, compressed=False)
        if not os.path.exists(orig_path):
            raise SystemExit(f"{orig_path} not found -- run `-e orig --cache` first.")
        at_orig = veritas.AddTree.read(orig_path)

        at_compr = lop_compress(
            dname, fold, dtrain, dvalid, dtest, at_orig, model_type, 
            cache=cache, log=log
        )

        # check if LOP-compressed model is enumerable
        sols, __, ef, enumerable = run_fairness_check(
            at_compr, d, dtest, sensitive_col, sensitive_col_idx
        )

        if not enumerable:
            print("⚠️  [LOP-FAIL] LOP-compressed model is not enumerable.")
        
        # add fairness info to the metrics already logged by lop_compress
        log["compressed_at"].update({
            "#unfair_boxes": len(sols),
            "emp_fairness": ef,
            "enumerable": enumerable,
        })
        

    elif exp == "milp":
        # 2. Run MILP repair loop on LOP-compressed model

        # load LOP-compressed model from cache
        at_compr = veritas.AddTree.read(
            get_model_path(dname, fold, mtype=model_type, compressed=True)
        )

        # run our fairness repair loop
        at_repaired = fairness_repair_loop(
            d, dtrain, dtest, at_compr.copy(), sensitive_col, 
            sensitive_col_idx, objective="composite", log=log
        )

    elif exp == "milp-b":
        # 2b. Run MILP repair loop with hard accuracy budget on LOP-compressed model

        # load LOP-compressed model from cache
        at_compr = veritas.AddTree.read(
            get_model_path(dname, fold, mtype=model_type, compressed=True)
        )

        # run our fairness repair loop with hard accuracy budget
        at_repaired_b = fairness_repair_loop(
            d, dtrain, dtest, at_compr.copy(), sensitive_col, 
            sensitive_col_idx, log=log,
            accuracy_budget=0.02  # (!) budget currently hardcoded to 2% bacc. loss
        )

    elif exp == "fatt":
        # 3. Train FATT model
        at_fatt = train_fatt(
            d, fold, dtrain, dvalid, dtest, sensitive_col, 
            sensitive_col_idx, log=log
        )

    elif exp == "budro":
        # 4. Train BuDRO model, reading config from budro/budro_configs.json
        #    (need to first run BuDRO tuning through ./run_budro_tuning.sh)
        at_budro = train_budro(
            dname, fold, dtrain, dtest, sensitive_col,
            sensitive_col_idx, d, log=log
        )

    # elif exp == "fairgbm":
    #     # 5. Train FairGBM model
    #     at_fairgbm = train_fairgbm(
    #         dname, dtrain, dtest, fold, sensitive_col, 
    #         sensitive_col_idx, d, cache=cache, log=log
    #     )
  
    print_summary(log, dname, fold, exp)

    # save results to JSON file
    # (the verify pass keeps the runtime of the original training run)
    runtime_key = "verify_runtime" if verify else "runtime"
    log[runtime_key] = time.time() - t_start_script
    if cache:
        with open(res_file, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n✓ Results saved to {res_file}")
    print(f"✓ Total runtime: {log[runtime_key]:.1f}s\n")

    

# ------- Functions to perform fairness repair or fairness-aware training  ----------

# 1. our MILP repair loop
def fairness_repair_loop(
        d, dtrain, dtest, at, sensitive_col, sensitive_col_idx, 
        objective="composite", log={}, lambda_l1=LAMBDA_L1, accuracy_budget=None, 
    ):

    from fairness import check_fixes_hold
    from leaf_refinement_fairness import (
        build_W,
        get_ocs_batch, 
        collect_unique_OCs,
        extract_pairs_batch,
        fix_fairness,
    )

    print(" \n ===================== Running fairness repair loop ==================== \n")

    # print model size
    W_original = build_W(at).copy()  # save before we modify addtree
    print(f"\n >> Model size: {len(at)} trees, {W_original.shape[1]} leaves")

    # check original accuracy & balanced accuracy
    acc_train, acc_test, bacc_train, bacc_test = evaluate_accuracies(
        at, dtrain, dtest, label="Initial"
    )

    # extract and cache training set OCs for original model structure
    t_oc = time.time()
    Phi_train = get_ocs_batch(dtrain.X, at)
    OCs_train = collect_unique_OCs(Phi_train, dtrain.y)
    print(f"\n >> Extracted {len(OCs_train)} unique OCs from the {len(dtrain.X)} training examples in {time.time() - t_oc:.3f} s.")
    train_info = {"X_train": dtrain.X, "y_train": dtrain.y, "OCs_train": OCs_train}
    if accuracy_budget is not None:
        min_bacc, old_tp, old_tn, P, N = compute_allowed_misclassifications(
            at, dtrain.X, dtrain.y, accuracy_budget
        )
        train_info["min_bacc"] = min_bacc
        train_info["old_tp"] = old_tp
        train_info["old_tn"] = old_tn
        train_info["P"] = P
        train_info["N"] = N

    # run initial fairness check
    t_fair = time.time()
    solutions, feat_map, emp_fairness_original, enumerable = \
        run_fairness_check(at, d, dtest, sensitive_col, sensitive_col_idx)
    total_fairness_time = time.time() - t_fair

    print(f"\n * Found {len(solutions)} unfair boxes!")

    if len(solutions) == 0 or not enumerable:
        log[f"{objective}_repaired_at"] = {
            "iterations": 0,
            "status": "fair" if enumerable else "not_enumerable",
            "train_acc": acc_train,
            "test_acc": acc_test,
            "train_bacc": bacc_train,
            "test_bacc": bacc_test,
            "#unfair_boxes": len(solutions),
            "emp_fairness": emp_fairness_original,
            "milp_time": 0,
        }
        return at

    print(f"\n --------------------------------------------------------------------------")
    at_current = at.copy()
    all_pairs = []                      # accumulated across iterations
    initial_unfair = len(solutions)
    total_milp_time = 0.0
    status = "unknown"
    # final_weights_info = None

    for iteration in range(1, MAX_IT + 1):
        print(f"\n  -- Iteration {iteration}/{MAX_IT} ({len(solutions)} unfair boxes) -- \n")

        # 2a. Create pairs from current unfair boxes
        t_pairs = time.time()
        X_batch = extract_pairs_batch(solutions, feat_map, d.X.columns, sensitive_col_idx)
        phi_batch = get_ocs_batch(X_batch, at_current)
        new_pairs = [(phi_batch[2 * i, :], phi_batch[2 * i + 1, :]) for i in range(len(solutions))]
        print(f"  >> Extracted {len(new_pairs)} unfair pairs in {time.time() - t_pairs:.3f} seconds.")
        all_pairs.extend(new_pairs)
        print(f"  >> New pairs: {len(new_pairs)}. Total pairs: {len(all_pairs)}.")

        # 2b. Fix unfair pairs with MILP + constraint on train accuracy
        t_milp_start = time.time()
        status, at_repaired, __ = fix_fairness(
            at_current, all_pairs, train_info, objective,
            accuracy_budget=accuracy_budget,
            lambda_l1=lambda_l1, verbose=True,
        )
        t_milp_end = time.time()
        total_milp_time += t_milp_end - t_milp_start
        print(f"\n  Optimization status: {status} (took {t_milp_end - t_milp_start:.3f} s)")

        if at_repaired is None:
            if status not in ("infeasible", "timeout", "fair"):
                raise ValueError(f"MILP repair failed with status {status}.")
            print(f"\n  !! MILP {status}. Exiting repair loop.")
            log[f"{objective}_repaired_at"] = {
                "iterations": iteration,
                "status": status,
                "milp_time": total_milp_time,
                "fairness_time": total_fairness_time,
            }
            return

        # keep weights for final diagnostics
        # final_weights_info = weights_info

        # 2c. Verify original unfair pairs are fixed
        fixed_count = check_fixes_hold(at_repaired, solutions, feat_map, d, sensitive_col_idx)
        assert fixed_count == len(solutions), "Not all unfair pairs were fixed!"

        # 2d. Check repaired model accuracy
        acc_train_repaired, acc_test_repaired, bacc_train_repaired, bacc_test_repaired = \
            evaluate_accuracies(at_repaired, dtrain, dtest, label="Repaired")

        # 2e. Re-check fairness on the repaired model
        t_fair = time.time()
        solutions_new, __, emp_fairness, enumerable_new = run_fairness_check(
            at_repaired, d, dtest, sensitive_col, sensitive_col_idx
        )
        total_fairness_time += time.time() - t_fair
        # check that repaired model is still enumerable
        if not enumerable_new:
            raise SystemExit(
                f"ERROR: repaired model not enumerable anymore??"
            )
        remaining = len(solutions_new)
        print(f"\n  >> Unfair boxes remaining: {remaining}.")

        at_current = at_repaired
        solutions = solutions_new

        if remaining == 0:
            print(f"\n  *** Converged! Fairness reached after {iteration} iteration(s). ***")
            break

    if remaining > 0:
        # hit MAX_IT without converging: model is still unfair
        print(f"\n  !! Reached MAX_IT={MAX_IT} with {remaining} unfair boxes left.")
        status = "max_iterations"

    # check how many leaves have been encoded and modified by the MILP repair
    from leaf_refinement_fairness import build_W as _build_W, _collect_touched_leaves
    W_final = _build_W(at_current)
    tot_leaves = W_final.shape[1]
    leaves_encoded = len(_collect_touched_leaves(all_pairs))
    leaves_modified = float(
        np.sum(np.abs(W_final[0, :] - W_original[0, :]) > 1e-8)
    )
    
    
    print(f" \n----------------------------------------------------------------------")
    print(f" [Summary of fairness repair loop] ")
    # print(f"    - Train accuracy:          {acc_train:.4f} --> {acc_train_repaired:.4f} (Δ {acc_train_repaired - acc_train:.4f})")
    # print(f"    - Train balanced accuracy: {bacc_train:.4f} --> {bacc_train_repaired:.4f} (Δ {bacc_train_repaired - bacc_train:.4f})")
    print(f"    - Test accuracy:            {acc_test:.4f} --> {acc_test_repaired:.4f} (Δ {acc_test_repaired - acc_test:.4f})")
    print(f"    - Test balanced accuracy:   {bacc_test:.4f} --> {bacc_test_repaired:.4f} (Δ {bacc_test_repaired - bacc_test:.4f})")
    print(f"    - Unfair boxes:             {initial_unfair} --> {remaining}")
    print(f"    - Empirical fairness:       {emp_fairness_original:.4f} --> {emp_fairness:.4f}")
    print(f"    - Total veritas time:       {total_fairness_time:.1f} seconds")
    print(f"    - Total MILP time:          {total_milp_time:.1f} seconds")
    print(f"    - % leaves encoded in MILP: {leaves_encoded/tot_leaves*100:.1f}%")
    print(f"    - % leaves changed by MILP: {leaves_modified/tot_leaves*100:.1f}%")

    # if final_weights_info is not None:
    #     from repair_diagnostics import run_diagnostics
    #     W_final = build_W(at_current)
    #     run_diagnostics(at, at_current, all_pairs, train_info["OCs_train"],
    #                 W_original, W_final, len(dtrain.X))

    log[f"{objective}_repaired_at"] = {
        "iterations": iteration,
        "status": status,
        "train_acc": acc_train_repaired,
        "test_acc": acc_test_repaired,
        "train_bacc": bacc_train_repaired,
        "test_bacc": bacc_test_repaired,
        "#unfair_boxes": remaining,
        "emp_fairness": emp_fairness,
        "milp_time": total_milp_time,
        "fairness_time": total_fairness_time,
        "leaves_total": tot_leaves,
        "leaves_encoded": leaves_encoded,
        "leaves_modified": leaves_modified,
    }

    return at_current

# 2. Train FATT decision tree (Ranzato & Zanella, CIKM 2021)
def train_fatt(d, fold, dtrain, dvalid, dtest, sensitive_col, ordinal_cols, log={}):

    from fatt_baseline import (
        export_dataset_fatt,
        train_fatt,
        # tune_fatt_config,
        silva_predict,
        fatt_to_addtree,
        DEFAULT_FATT_CONFIG,
    )

    print(" \n ===================== Training FATT tree ==================== \n")

    # 1. export dataset in FATT format
    train_csv  = f"{FATT_CACHE}/{d.name()}_fold{fold}.csv"
    silva_path = f"{FATT_CACHE}/{d.name()}_fold{fold}.silva"
    export_dataset_fatt(dtrain.X, dtrain.y, train_csv)

    # 2. tune + train FATT
    base_config = {**DEFAULT_FATT_CONFIG, "seed": fold + 1}
    # best_config = tune_fatt_config(
    #     d, fold, dtrain, dvalid, base_config, sensitive_col, ordinal_cols
    # )
    # config = best_config or base_config
    config = base_config # for now, we just use FATT default params
    t_train = time.time()
    train_fatt(FATT_BIN, train_csv, silva_path, config)
    train_time = time.time() - t_train

    def _count_leaves(silva_path):
        with open(silva_path) as f:
            return sum(1 for line in f if line.startswith("LEAF"))

    n_leaves = _count_leaves(silva_path)
    print(f"\n  FATT training done in {train_time:.1f}s  ({n_leaves} leaves)")

    at_fatt = fatt_to_addtree(silva_path)

    # 3. check alignment btw .silva model and addtree
    preds_silva = silva_predict(SILVA_BIN, silva_path, dtest.X)
    # preds_silva_at = (at_fatt.predict(dtest.X.to_numpy()) >= 0.5).astype(int).flatten()
    # switched to at.eval() for better numerical stability
    preds_silva_at = (at_fatt.eval(dtest.X.to_numpy()) >= 0.0).astype(int).flatten()
    assert np.array_equal(preds_silva, preds_silva_at), \
        "Mismatch between Silva binary predictions and AddTree predictions!"

    # 4. evaluate FATT accuracy
    acc_train, acc, bacc_train, bacc = evaluate_accuracies(at_fatt, dtrain, dtest, label="FATT")

    # 5. check fairness with veritas
    solutions, __, emp_fairness, enumerable = run_fairness_check(at_fatt, d, dtest, sensitive_col, ordinal_cols)
    print(f"  * Found {len(solutions)} unfair boxes!")

    # 6. evaluate empirical fairness of FATT model
    # emp_fairness = evaluate_empirical_fairness(at_fatt, dtest, sensitive_col)
    print(f"  * Empirical fairness on test set: {emp_fairness:.4f}")

    log["fatt"] = {
        "train_acc": acc_train,
        "train_bacc": bacc_train,
        "test_acc": acc,
        "test_bacc": bacc,
        "runtime": train_time,
        "num_leaves": n_leaves,
        "emp_fairness": emp_fairness,
        "#unfair_boxes": len(solutions),
        "enumerable": enumerable,
        "config": config,
    }

    return at_fatt


# 4. Train BuDRO model (Vargo et al., ICLR 2021)
def train_budro(dname, fold, dtrain, dtest, sensitive_col, ordinal_cols, d, log={}):
    import tensorflow as tf
    import xgboost as xgb
    import random
    from budro.tuning import (load_config, subsample, xgb_params, TRAIN_SUBSAMPLE, LOWG, HIGHG,
                              SEED_BUDRO)

    print(" \n ===================== Training BuDRO ==================== \n")

    # Load per (dataset, fold) hyperparameters, previously tuned by budro/tuning.py.
    # `solver` routes the two BuDRO implementations:
    #   "sinkhorn"  small n: dense n x n C + entropic Sinkhorn (budro.fair_metric)
    #   "dual"      large n: matrix-free LazyCost + exact root-find eta (budro/tf_dual.py)
    bcfg = load_config(dname, fold)
    solver = bcfg["solver"]
    n_iter = int(bcfg["n_estimators"])
    eps = float(bcfg["eps"])
    gamma_reg = float(bcfg.get("gamma_reg", 0.00005))          # Sinkhorn only

    X = dtrain.X.to_numpy(dtype=np.float32)
    y = dtrain.y.to_numpy().astype(int)

    # BuDRO is O(n^2) per boosting step: 
    # ACSEmployment's training set is subsampled to 125k rows.
    n_full = X.shape[0]
    X, y = subsample(X, y, ordinal_cols[0], TRAIN_SUBSAMPLE.get(dname))
    if X.shape[0] < n_full:
        print(f"  !! SUBSAMPLED train {n_full:,} -> {X.shape[0]:,} rows (stratified on "
              f"label x sensitive); valid/test folds are FULL.")

    n = X.shape[0]
    param = xgb_params(n, y, bcfg["eta"], bcfg["max_depth"])
    print(f"  BuDRO [{solver}] config for {dname} fold {fold}: eta={param['eta']:g}, "
          f"max_depth={param['max_depth']}, n_estimators={n_iter}, eps={eps:g}, "
          f"lambda={param['lambda']:g}")

    import budro
    txl, _script, _ds = budro.import_budro()   # BuDRO's boosting modules; also patches tf_dual
    random.seed(SEED_BUDRO); np.random.seed(SEED_BUDRO); tf.random.set_seed(SEED_BUDRO)

    if solver == "sinkhorn":
        tC = time.time()
        C = budro.fair_metric(X, ordinal_cols, dataset=dname)          # dense n x n
        c_time = time.time() - tC
        print(f"  Computed dense fair-metric C in {c_time:.1f}s.")
        tB = time.time()
        bst, _ = txl.boost_sinkhorn_tf(
            X, y, tf.constant(C, dtype=tf.float32), n_iter,
            pred=None, eps=eps, gamma_reg=gamma_reg, param=param, bst=None,
            lowg=LOWG, highg=HIGHG, verbose=False)
        boost_time = time.time() - tB
    elif solver == "dual":
        tC = time.time()
        C = budro.fair_metric_lazy(X, ordinal_cols, use_proxy=True)       # matrix-free
        c_time = time.time() - tC
        print(f"  Built matrix-free C (LazyCost) in {c_time:.1f}s "
              f"[dense would be {n * n * 4 / 2**30:.1f}GB].")
        _ds.reset_state()
        tB = time.time()

        # --- boost_dual socket in budro/tf_dual.py ---
        bst, _, _ = txl.boost_dual(
            X, y, C, n_iter,
            X_test=None, y_test=None, pred=None, eps=eps,
            param=param, bst=None, verbose=False, verify=False,
            dtype=tf.float32)
        boost_time = time.time() - tB
    else:
        raise ValueError(f"unknown BuDRO solver {solver!r} in config for {dname}")

    train_time = c_time + boost_time
    print(f"  BuDRO training done in {train_time:.1f}s "
          f"(C matrix {c_time:.1f}s + boosting {boost_time:.1f}s)")

    # convert BuDRO model to veritas AddTree for evaluation
    at = veritas.get_addtree(bst)
    print(f"  BuDRO model has {len(at)} trees with max depth {at.max_depth()} ({at.num_leafs()} leaves).")

    preds_budro = (bst.predict(xgb.DMatrix(dtrain.X.to_numpy())) >= 0.5).astype(int)
    preds_veritas = (at.eval(dtrain.X.to_numpy()).flatten() >= 0.0).astype(int)
    agree = (preds_budro == preds_veritas).mean()
    # print(f"  BuDRO vs veritas prediction agreement: {agree:.4f}")
    assert agree > 0.999, f"Conversion altered predictions! Agreement: {agree:.4f}"

    acc_train, acc_test, bacc_train, bacc_test = evaluate_accuracies(at, dtrain, dtest, label="BuDRO")

    solutions, __, emp_fairness, enumerable = run_fairness_check(at, d, dtest, sensitive_col, ordinal_cols)
    print(f"  * Found {len(solutions)} unfair boxes!")
    print(f"  * Empirical fairness on test set: {emp_fairness:.4f}")

    # HACK: even if veritas could not enumerate the full OC space, the model is PROVABLY unfair
    #       if (a) a box was found before the timeout: run_fairness_check keeps a box only if
    #       its extracted point pair gets different classes, so it is a concrete witness; or
    #       (b) emp_fairness < 1: a test point flips class with the sensitive attribute.
    #       We log a lower bound on #unfair_boxes (the boxes found, at least 1) + enumerable=True,
    #       so this unsafe model is not dropped in the results.
    n_unfair_boxes = len(solutions)
    unfair_boxes_lower_bound = False
    if not enumerable and (len(solutions) > 0 or emp_fairness < 1.0):
        n_unfair_boxes = max(len(solutions), 1)
        print(f"  * (!) OC-space not enumerable, but the model is provably unfair "
              f"({len(solutions)} boxes found, emp_fairness={emp_fairness:.4f}) -> logging "
              f"#unfair_boxes={n_unfair_boxes} (lower bound), enumerable=True.")
        enumerable, unfair_boxes_lower_bound = True, True

    # UNKNOWN status: veritas timed out with 0 boxes found and emp_fairness == 1, so the model is
    # neither certified fair nor provably unfair. Do not save it but terminate with an error.
    # [this is just an additional check, it never happened in practice]
    if not enumerable:
        print(f"\n  ERROR: BuDRO model for {dname} fold {fold} has UNKNOWN fairness status: "
              f"veritas could not enumerate its OC space (0 unfair boxes found before the "
              f"timeout, emp_fairness={emp_fairness:.4f}).\n"
              f"  Config: eta={param['eta']:g}, max_depth={param['max_depth']}, "
              f"n_estimators={n_iter}, eps={eps:g} -> {len(at)} trees, {at.num_leafs()} leaves.\n"
              f"  Nothing saved. Pick another config for this fold in budro/budro_configs.json "
              f"and rerun.", flush=True)
        sys.exit(1)

    log["budro"] = {
        "train_acc": acc_train,
        "train_bacc": bacc_train,
        "test_acc": acc_test,
        "test_bacc": bacc_test,
        "runtime": train_time,
        "c_time": c_time,
        "boost_time": boost_time,
        "num_leaves": at.num_leafs(),
        "emp_fairness": emp_fairness,
        "#unfair_boxes": n_unfair_boxes,
        "unfair_boxes_lower_bound": unfair_boxes_lower_bound,   # True: search timed out, see above
        "enumerable": enumerable,
        "config": {"solver": solver, "eta": param["eta"], "max_depth": param["max_depth"],
                   "n_estimators": n_iter, "eps": eps,
                   "reg_lambda": param["lambda"], "gamma_reg": gamma_reg},
    }

    return at


# 5. Train FairGBM model (Cruz et al., ICLR 2023) - group fairness with FPR/FNR constraints
# def train_fairgbm(dname, dtrain, dtest, fold, sensitive_col, ordinal_cols, d, cache=True, log={}):
#     from fairgbm import FairGBMClassifier
#     # import os

#     # model_path = get_model_path(dname, fold, mtype="lgbm_fairgbm", cv=False, compressed=False)

#     print(" \n ===================== Training FairGBM ==================== \n")

#     # if cache and os.path.exists(model_path):
#     #     print(f" * Loading cached FairGBM model from {model_path}")
#     #     at = veritas.AddTree.read(model_path)
#     # else:
#     S_train = dtrain.X[sensitive_col].to_numpy().astype(int)
#     # Note: FairGBM requires sensitive column to be mapped to {0,1} 
#     #       --> sensitive column already remapped to {0,1} by remap_sensitive_to_01()

#     # print(f"  * Training FairGBM with {max(10, log['compressed_at']['num_trees'])} trees and max_depth {max(4,log['compressed_at']['max_depth'])}...")
#     model = FairGBMClassifier(
#         constraint_type="FPR,FNR",
#         # use same size of LOP-compressed model 
#         # (for comparability and OC enumerability)
#         n_estimators=max(10, log["compressed_at"]["num_trees"]), 
#         max_depth = max(4, log["compressed_at"]["max_depth"]),
#         # [alternative] fixed size FairGBM
#         # HACK: slightly smaller model for 2 datasets or veritas goes OOM
#         # n_estimators=25 if dname not in ["DefaultCreditCardClients", "MEPSPanel21"] else 20,
#         # max_depth=4,
#         random_state=SEED,
#         n_jobs=NJOBS,
#     )
#     t_train = time.time()
#     model.fit(dtrain.X.to_numpy(), dtrain.y.to_numpy(), constraint_group=S_train)
#     train_time = time.time() - t_train
#     print(f"  FairGBM training done in {train_time:.1f}s")
        
#     at = fairgbm_to_addtree(model)

#     # sanity check: veritas predictions must match FairGBM's own predictions
#     preds_fairgbm = (model.predict_proba(dtrain.X.to_numpy())[:, 1] >= 0.5).astype(int)
#     # preds_veritas = (at.predict(dtrain.X.to_numpy()).flatten() >= 0.5).astype(int)
#     # switched to at.eval() for better numerical stability
#     preds_veritas = (at.eval(dtrain.X.to_numpy()).flatten() >= 0.0).astype(int)
#     agree = (preds_fairgbm == preds_veritas).mean()
#     print(f"  FairGBM vs veritas prediction agreement: {agree:.4f}")
#     assert agree > 0.99, f"Conversion altered predictions! Agreement: {agree:.4f}"

#     # if cache:
#     #     with open(model_path, "w") as f:
#     #         f.write(at.to_json())
#     #         print(f" * Saved FairGBM model to {model_path}.")

#     acc_train, acc_test, bacc_train, bacc_test = evaluate_accuracies(at, dtrain, dtest, label="FairGBM")

#     solutions, __, emp_fairness, enumerable = run_fairness_check(at, d, dtest, sensitive_col, ordinal_cols)
#     print(f"  * Found {len(solutions)} unfair boxes!")

#     # emp_fairness = evaluate_empirical_fairness(at, dtest, sensitive_col)
#     print(f"  * Empirical fairness on test set: {emp_fairness:.4f}")

#     # HACK: emp_fairness < 1 certifies that the model is not fair,
#     #       even if veritas could not enumerate the full OC space.
#     #       We log a lower bound of 1 unfair boxes + enumerable=True,
#     #       so this unsafe model is not dropped in the results.
#     n_unfair_boxes = len(solutions)
#     if not enumerable and emp_fairness < 1.0:
#         print(f"  * (!) OC-space not enumerable, but emp_fairness={emp_fairness:.4f} < 1 proves "
#               f"the model is unfair -> logging #unfair_boxes=1 (lower bound), enumerable=True.")
#         n_unfair_boxes, enumerable = 1, True

#     log["fairgbm"] = {
#         "train_acc": acc_train,
#         "train_bacc": bacc_train,
#         "test_acc": acc_test,
#         "test_bacc": bacc_test,
#         "runtime": train_time,
#         "num_leaves": at.num_leafs(),
#         "emp_fairness": emp_fairness,
#         "#unfair_boxes": n_unfair_boxes,
#         "enumerable": enumerable,
#     }

#     return at


if __name__ == "__main__":
    main()