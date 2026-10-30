"""
budro/tuning.py -- per-fold BuDRO hyperparameter tuning, 
writing the resulting configurations for the final training (main.py -e budro).

SETUP: One (dataset, fold) configuration, tuned on each fold's own validation split. 
       One job = one grid config on one (dataset, fold). 
       Boosting is incremental, so one fit logs validation metrics every OUT_EVERY rounds 
       and n_estimators is read off that trajectory (as the BuDRO authors do). 

HYPERPARAMS SELECTION:
    Highest validation balanced accuracy among the (config, round) snapshots with validation
    empirical fairness == 1.0. Ties -> fewer rounds. If no snapshot reaches 1.0: highest
    empirical fairness, then highest balanced accuracy.

Note: ALWAYS READ eff_eps. It is the budget the adversary ACTUALLY spent, and it is what
      certifies that a row labelled eps=X was really run at eps=X.

USAGE (from the repo root):
    1. ./run_budro_tuning.sh [-j JOBS] [datasets...]
           Run the whole tuning: every (dataset, fold, config) job, JOBS at a time.
           Rerunning it only runs the jobs that have no result yet (e.g. after a crash).
    2. python -m budro.tuning select [datasets...]
           When all jobs are done: apply the selection rule above to every (dataset, fold)
           and write the chosen configs to budro/budro_configs.json (read by main.py -e budro).

    Used internally by run_budro_tuning.sh (only needed by hand to rerun one failed job):
    python -m budro.tuning list [datasets...]
           Print the command of every job that has no result yet.
    python -m budro.tuning run <dataset> --fold F --config I
           Run ONE job: grid config number I (0..47, an index into GRID = one
           (eta, depth, eps) combination) on fold F of <dataset>.
"""

import os
import sys
import json
import time

import numpy as np


# ==========================================================================================
# 1) Settings shared by tuning and final training
# ==========================================================================================

SEED_BUDRO = 6

# ---- solver switch ----------------------------------------------------------------------
# 12k: LawSchool (11,216) stays on Sinkhorn; DutchCensus (36,252) goes dual.
# Below this, dense C is <= ~0.5 GB and Sinkhorn is cheaper per round than dual.
SINKHORN_MAX_N = 12_000

def pick_solver(n):
    return "sinkhorn" if n <= SINKHORN_MAX_N else "dual"


# ---- fixed hyperparameters ----------------------------------------------------------------
LAMBDA = 1e-4           # their winners span 1.0/1e-4/1e-8; their notes call it "pretty random"
MCW_NUM = 0.1           # min_child_weight = MCW_NUM / n  (Adult and COMPAS both 0.1/n)
GAMMA_REG = 5e-5        # Sinkhorn only. The ONLY value in every grid they ran.
LOWG, HIGHG = 0.0, 10.0  # Sinkhorn root bracket

XGB_NTHREAD = 2

def xgb_params(n, y, eta, max_depth):
    """xgboost parameters of one BuDRO run on n training points (same in tuning and training)."""
    y = np.asarray(y).astype(int)
    num0, num1 = int((y == 0).sum()), int((y == 1).sum())
    return {
        "objective": "binary:logistic",
        "nthread": XGB_NTHREAD,
        "max_depth": int(max_depth),
        "eta": float(eta),
        "min_child_weight": MCW_NUM / n,    # tied to ~1/n (OT weights sum to 1)
        "lambda": LAMBDA,
        "scale_pos_weight": num0 / num1,
        "seed": SEED_BUDRO,
    }


# ---- subsampling --------------------------------------------------------------------------
# Final training: BuDRO's adversary maximises over every PAIR of training points, so cost is
# O(n^2 d): at ACSEmployment's n=227k a single run is ~5 DAYS. We subsample it to 100k.
TRAIN_SUBSAMPLE = {"ACSEmployment": 125_000}

# Tuning uses the full training set for all datasets (tried to subsample, didn't work well)
TUNE_MAX_N = 125_000 

def stratified_subsample(y, s, n_sub, seed=SEED_BUDRO):
    """Indices of an n_sub-row sample keeping the (label, sensitive) cell proportions."""
    rng = np.random.default_rng(seed)
    keys = (np.asarray(y).astype(int) * 2 + np.asarray(s).astype(int)).astype(int)
    take = []
    n = len(keys)
    for k in np.unique(keys):
        idx = np.nonzero(keys == k)[0]
        m = max(1, int(round(n_sub * idx.size / n)))
        take.append(rng.choice(idx, size=min(m, idx.size), replace=False))
    out = np.sort(np.concatenate(take))
    return out[rng.permutation(out.size)][:n_sub] if out.size > n_sub else out


def subsample(X, y, prot_idx, n_max):
    """(X, y) capped at n_max rows (stratified); returned unchanged if already small enough."""
    if n_max is None or X.shape[0] <= n_max:
        return X, y
    keep = stratified_subsample(y, X[:, prot_idx], n_max)
    return X[keep], y[keep]


# ---- selected configs (written by `select`, read by main.py -e budro) ---------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "budro_configs.json")


def load_config(dname, fold):
    """The selected config of ONE (dataset, fold); was one config per dataset."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"{CONFIG_PATH} not found -- run the BuDRO tuning first "
                                f"(./run_budro_tuning.sh, then python -m budro.tuning select).")
    with open(CONFIG_PATH) as f:
        cfgs = json.load(f)
    try:
        return cfgs[dname][str(fold)]
    except KeyError:
        raise KeyError(f"No BuDRO config for {dname} fold {fold} in {CONFIG_PATH}.")


def _save_configs(selected):
    """Merge {dname: {fold: entry}} into CONFIG_PATH."""
    cfgs = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfgs = json.load(f)
    for dname, per_fold in selected.items():
        cfgs.setdefault(dname, {}).update({str(k): v for k, v in per_fold.items()})
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfgs, f, indent=2)


# ==========================================================================================
# 2) Tuning
# ==========================================================================================

# Hyperparameter grid: eta, max_depth, eps. 
ETA_GRID = [1e-4, 0.005, 0.02, 0.05]
DEPTH_GRID = [2, 4, 6]
EPS_GRID = [0.01, 0.05, 0.5, 1.0]
GRID = [(e, dp, ep) for e in ETA_GRID for dp in DEPTH_GRID for ep in EPS_GRID]

N_ITER = 200            # rounds per fit; n_estimators is picked off the trajectory
OUT_EVERY = 10          # validation metrics every OUT_EVERY rounds

DATASETS = [            # cheapest first, so complete results for the small datasets come early
    "CompasTwoYears", "BankMarketingFull", "DefaultCreditCardClients", "MEPSPanel21",
    "LawSchool", "DutchCensus", "ACSPublicCoverage", "ACSEmployment", "ACSTravelTime",
    "ACSIncome",
]
FOLDS = [0, 1, 2, 3, 4]


def _results_dir():
    from utils import CACHE_DIR
    return f"{CACHE_DIR}/results/budro_tuning"   


def _partial_path(dname, fold, cfg_idx):
    return f"{_results_dir()}/{dname}_fold{fold}_cfg{cfg_idx:02d}.json"


# ---- fold loading -------------------------------------------------------------------------

def load_fold(dname, fold):
    """Load one CV fold + its fair metric. The metric representation depends on the
    solver: dense ndarray for Sinkhorn, LazyCost for dual."""
    import io
    import contextlib
    from utils import get_dataset, remap_sensitive_to_01, SENSITIVE_COLS, SEED

    with contextlib.redirect_stdout(io.StringIO()):
        d, dtrain, dvalid, dtest = get_dataset(dname, SEED, fold=fold, silent=True)
        sensitive_col = SENSITIVE_COLS[dname]
        remap_sensitive_to_01(sensitive_col, d, dtrain, dvalid, dtest)
        prot_idx = d.X.columns.get_loc(sensitive_col)
        Xtr = dtrain.X.to_numpy(dtype=np.float32); ytr = dtrain.y.to_numpy().astype(int)
        Xva = dvalid.X.to_numpy(dtype=np.float32); yva = dvalid.y.to_numpy().astype(int)

        n_full = Xtr.shape[0]
        n_max = min(TUNE_MAX_N, TRAIN_SUBSAMPLE.get(dname, n_full)) # subsample datasets for tuning
        Xtr, ytr = subsample(Xtr, ytr, prot_idx, n_max)

        solver = pick_solver(Xtr.shape[0])

        import budro
        if solver == "sinkhorn":
            C = budro.fair_metric(Xtr, [prot_idx], use_proxy=True, dataset=dname)
        else:
            C = budro.fair_metric_lazy(Xtr, [prot_idx], use_proxy=True)

    if Xtr.shape[0] < n_full:
        print(f"  !! SUBSAMPLED train {n_full:,} -> {Xtr.shape[0]:,} rows (stratified on "
              f"label x {sensitive_col}).", flush=True)

    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, C=C, solver=solver,
                dvalid=dvalid, sensitive_col=sensitive_col,
                prot_idx=prot_idx, n_full=n_full, n_used=Xtr.shape[0])


def group_gap(scores, X, y_true, prot_idx):
    """GROUP fairness GapRMS/GapMax via BuDRO's own script.group_metrics. Logged for
    context only; selection is on (bacc, emp_fairness)."""
    import script
    y_pred = (scores >= 0.0).astype(int)
    r = script.group_metrics(y_true, y_pred, X[:, prot_idx],
                             label_protected=0, label_good=1, verbose=False)
    return float(r[4]), float(r[5])


# ---- one fit -> the whole boosting trajectory ---------------------------------------------

def fit_trajectory(txl, tf, fd, param, n_iter, eps, out_every):
    """One BuDRO fit, evaluating VALIDATION metrics every `out_every` boosting rounds via
    a callback the solver runs after each round. Boosting is incremental, so a single run
    yields the whole trajectory and n_estimators becomes a free, post-hoc dimension
    (as the authors do: their 68/90/180 are read off the curve, not gridded).

    On the dual path boost_dual's callback passes no slack, so eff_eps (the eta*-convergence
    diagnostic) is read from the solver's spend_log() -- the mean distance it spent that round.

    Returns (recs, eval_s, probes). recs = [{round, bacc, efair, grms, gmax, slack, eff_eps},...]
    sampled every out_every rounds; `probes` is the dual solver's argmax passes for EVERY
    round (the dual runtime is probes x one O(n^2) pass, so it is the number to watch).
    """
    import io
    import contextlib
    import veritas
    from sklearn.metrics import balanced_accuracy_score
    from fairness import evaluate_empirical_fairness

    Xva, yva = fd["Xva"], fd["yva"]
    recs, eval_s, probes = [], [0.0], []
    t_start = time.perf_counter()
    _ds = None
    real_stdout = sys.stdout      # boost_* is wrapped in redirect_stdout; progress must escape

    def _eval_cb(bst, dtrain, dtest, max_loss, it, slack=None):
        # slack=None on the Sinkhorn path (boost_sinkhorn_tf doesn't pass it); its root
        # solve enforces <C,Pi> == eps by construction, so eff_eps == eps there.
        t = time.perf_counter()
        at = veritas.get_addtree(bst, silent=True)
        sva = at.eval(Xva).flatten()
        gr, gm = group_gap(sva, Xva, yva, fd["prot_idx"])
        _eff = _ds.spend_log()[it] if (_ds is not None and it < len(_ds.spend_log())) else eps
        recs.append({
            "round": it + 1,
            "bacc": balanced_accuracy_score(yva, (sva >= 0.0).astype(int)),
            "efair": evaluate_empirical_fairness(at, fd["dvalid"], fd["sensitive_col"]),
            "grms": gr, "gmax": gm,
            "slack": eps - _eff,
            "eff_eps": _eff,
        })
        eval_s[0] += time.perf_counter() - t

        # print progress
        if _ds is not None:
            pl = _ds.probe_log()
            el = time.perf_counter() - t_start
            done = it + 1
            eff = recs[-1]["eff_eps"] / eps if eps else float("nan")
            print(f"    [{done:>3}/{n_iter}] eff_eps={eff:>4.0%}  probes/rnd={np.mean(pl):>5.1f}"
                  f"  elapsed={el/60:>5.1f}m  projected={el/done*n_iter/60:>5.1f}m",
                  file=real_stdout, flush=True)

    # ---- DIVERGENCE HANDLING -------------------------------------------------------------
    # BuDRO's loss is -log(1 - pred) (tf_xgboost_log.py:148). 
    # Once the model is so confident that pred == 1.0, the loss is inf, 
    # the transport weights NaN, and xgboost rejects them ("Weights must be positive values"). 
    # It is THEIR code, so we do not patch it: the run stops there, 
    # and every snapshot recorded before the crash is a valid model (the final run
    # reproduces it up to that round). Any other xgboost error is re-raised.
    import xgboost as xgb
    diverged = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            if fd["solver"] == "sinkhorn":
                txl.boost_sinkhorn_tf(
                    fd["Xtr"], fd["ytr"], tf.constant(fd["C"], tf.float32), n_iter,
                    X_test=Xva, y_test=yva, pred=None,
                    eps=eps, gamma_reg=GAMMA_REG, param=param, bst=None,
                    lowg=LOWG, highg=HIGHG, verbose=False,
                    outinterval=out_every, outfunc=_eval_cb)
            else:
                # boost_dual socket (root-find find_wts in budro/tf_dual.py). Its callback passes
                # no slack, so eff_eps is read from _ds.spend_log() in _eval_cb.
                import budro
                _ds = budro._install_patched_tf_dual()
                _ds.reset_state()          # per-run: forget the previous run's eta
                txl.boost_dual(
                    fd["Xtr"], fd["ytr"], fd["C"], n_iter,
                    X_test=Xva, y_test=yva, pred=None, eps=eps,
                    param=param, bst=None, verbose=False, verify=False,
                    dtype=tf.float32,
                    outinterval=out_every, outfunc=_eval_cb)
        except xgb.core.XGBoostError as err:
            if "Weights must be positive" not in str(err):
                raise
            diverged = str(err).splitlines()[0]
        if _ds is not None:
            probes = _ds.probe_log()
    # ---- end DIVERGENCE HANDLING ------------------------------------------------------------

    return recs, eval_s[0], probes, diverged


def run_job(dname, fold, cfg_idx, n_iter=N_ITER, out_every=OUT_EVERY):
    """One grid config on one (dataset, fold) -> one partial file."""
    import random
    import budro
    txl, _script, _ds = budro.import_budro()       # patches tf_dual (root-find + LazyCost)
    import tensorflow as tf

    # Limit how many CPU threads TensorFlow uses, so parallel jobs don't fight over cores.
    # Set by run_budro_tuning.sh (before TF does any work).
    _intra = int(os.environ.get("TF_NUM_INTRAOP_THREADS", "0"))
    if _intra > 0:
        tf.config.threading.set_intra_op_parallelism_threads(_intra)
        tf.config.threading.set_inter_op_parallelism_threads(
            int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")) or 1)

    random.seed(SEED_BUDRO); np.random.seed(SEED_BUDRO); tf.random.set_seed(SEED_BUDRO)
    eta, depth, eps = GRID[cfg_idx]

    t_setup = time.perf_counter()
    fd = load_fold(dname, fold)
    n = fd["Xtr"].shape[0]
    print(f"---- {dname} fold {fold} cfg {cfg_idx}: eta={eta:g} depth={depth} eps={eps:g} | "
          f"solver={fd['solver']} train {fd['Xtr'].shape} valid {fd['Xva'].shape} "
          f"[setup {time.perf_counter() - t_setup:.1f}s] ----", flush=True)

    param = xgb_params(n, fd["ytr"], eta, depth)
    t0 = time.perf_counter()
    recs, eval_s, probes, diverged = fit_trajectory(txl, tf, fd, param, n_iter, eps, out_every)
    fit_s = time.perf_counter() - t0

    # A failed run is saved with the snapshots recorded before the crash (possibly none)
    # and "diverged" set; selection just treats it as a run with fewer snapshots.
    if diverged:
        last = recs[-1]["round"] if recs else 0
        print(f"  !! DIVERGED after round {last} ({diverged}): saving the {len(recs)} "
              f"snapshots before it", flush=True)

    if recs:
        # eff_eps is the budget the adversary ACTUALLY spent. Report the RANGE, not the mean.
        # The solver enforces mean_dist <= eps every round, so ratio > 1 should be impossible.
        e = np.array([r["eff_eps"] for r in recs]) / eps
        flag = ""
        if e.max() > 1.02:
            flag = "  <-- OVER BUDGET: not a valid eps run"
        elif e.min() < 0.9:
            flag = "  <-- underspent: adversary could not spend the budget"
        fair = [r for r in recs if r["efair"] >= 1.0]
        best = max(fair or recs, key=lambda r: r["bacc"])
        print(f"  best{'(fair)' if fair else '(NOT fair)'} BAcc@{best['round']}={best['bacc']:.4f} "
              f"emp_fair={best['efair']:.4f}  eff_eps={e.mean():.0%}[{e.min():.0%}-{e.max():.0%}]"
              f"  [fit {fit_s - eval_s:.1f}s | eval {eval_s:.1f}s]{flag}", flush=True)

    out = _partial_path(dname, fold, cfg_idx)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = {"dataset": dname, "fold": fold, "config": cfg_idx,
           "eta": eta, "depth": depth, "eps": eps,
           "solver": fd["solver"], "n_train_used": fd["n_used"], "n_train_full": fd["n_full"],
           "n_iter": n_iter, "out_every": out_every,
           "fit_s": fit_s, "eval_s": eval_s,
           "probes_mean": float(np.mean(probes)) if probes else None,
           "diverged": diverged,      # None for a normal run, else save that it failed
           "records": recs}
    tmp = f"{out}.tmp"                 # atomic write: a killed job leaves no half-written file
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    print(f"  saved {out}", flush=True)


# ---- job list -------------------------------------------------------------------------------

def list_jobs(datasets):
    """Commands of all (dataset, fold, config) jobs whose partial is not there yet."""
    for dname in datasets:
        for fold in FOLDS:
            for i in range(len(GRID)):
                if not os.path.exists(_partial_path(dname, fold, i)):
                    print(f"python -m budro.tuning run {dname} --fold {fold} --config {i}")


# ---- selection ------------------------------------------------------------------------------

def select_fold(partials):
    """Per-fold selection over all (config, round) snapshots of one fold:
    highest validation BAcc among snapshots with validation emp. fairness == 1.0 (ties -> fewer
    rounds); if none reaches 1.0, highest emp. fairness, then highest BAcc."""
    snaps = [(p, r) for p in partials for r in p["records"]]
    fair = [(p, r) for p, r in snaps if r["efair"] >= 1.0]
    if fair:
        p, r = max(fair, key=lambda pr: (pr[1]["bacc"], -pr[1]["round"]))
        rule = "max_bacc_emp_fair_1"
    else:
        p, r = max(snaps, key=lambda pr: (pr[1]["efair"], pr[1]["bacc"]))
        rule = "fallback_max_emp_fair"
    entry = {"solver": p["solver"], "eta": p["eta"], "max_depth": p["depth"], "eps": p["eps"],
             "n_estimators": r["round"], "reg_lambda": LAMBDA, "mcw_num": MCW_NUM,
             # the selected snapshot, for the record
             "val_bacc": r["bacc"], "val_emp_fair": r["efair"], "eff_eps": r["eff_eps"],
             "rule": rule, "n_train_tune": p["n_train_used"]}
    if p["solver"] == "sinkhorn":
        entry["gamma_reg"] = GAMMA_REG
    return entry


def select(datasets):
    selected = {}
    for dname in datasets:
        for fold in FOLDS:
            partials, missing = [], []
            for i in range(len(GRID)):
                path = _partial_path(dname, fold, i)
                if os.path.exists(path):
                    with open(path) as f:
                        partials.append(json.load(f))
                else:
                    missing.append(i)
            if missing:
                print(f"  [!] {dname} fold {fold}: {len(missing)}/{len(GRID)} configs missing "
                      f"-- NOT selected (rerun ./run_budro_tuning.sh {dname})")
                continue
            entry = select_fold(partials)
            selected.setdefault(dname, {})[fold] = entry
            ratio = entry["eff_eps"] / entry["eps"]
            flag = "" if 0.9 <= ratio <= 1.02 else f"  <-- eff_eps {ratio:.0%} of eps"
            print(f"  {dname:26s} fold {fold}: eta={entry['eta']:<6g} depth={entry['max_depth']} "
                  f"eps={entry['eps']:<5g} rounds={entry['n_estimators']:<4} "
                  f"val_bacc={entry['val_bacc']:.4f} val_emp_fair={entry['val_emp_fair']:.4f} "
                  f"[{entry['rule']}]{flag}")
    if selected:
        _save_configs(selected)
        print(f"\nSaved to {CONFIG_PATH}")


# ---- CLI ------------------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Per-fold BuDRO tuning.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run one (dataset, fold, config) job")
    p.add_argument("dataset")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--config", type=int, required=True, help=f"grid index, 0..{len(GRID) - 1}")

    p = sub.add_parser("list", help="print the jobs not done yet")
    p.add_argument("datasets", nargs="*", default=DATASETS)

    p = sub.add_parser("select", help="per-fold selection -> budro/budro_configs.json")
    p.add_argument("datasets", nargs="*", default=DATASETS)

    args = ap.parse_args()
    if args.cmd == "run":
        if not 0 <= args.config < len(GRID):
            raise SystemExit(f"--config must be in 0..{len(GRID) - 1}")
        run_job(args.dataset, args.fold, args.config)
    elif args.cmd == "list":
        list_jobs(args.datasets)
    else:
        select(args.datasets)


if __name__ == "__main__":
    main()
