"""Reproduce BuDRO (Vargo et al., ICLR 2021) on the authors' own COMPAS and German data.

An independent check that our pipeline matches the published method: it reproduces the paper's
COMPAS (Table 3) and German (Table 1) results by running BuDRO end-to-end through their code:
  - their data prep + train/test split (compas_proc / german_proc, via aif360),
  - their fair-metric cost matrix C,
  - their boosting (scripts/boost_sinkhorn_tf, entropic/Sinkhorn solver),
  - their Table 4/8 hyperparameters,
  - their OWN evaluation functions: consistency (sex_cons/race_cons, status_cons)
    and group GapMax/GapRMS (script.group_metrics).

For each dataset we sweep the paper's eps grid and average every point over SEEDS random 80/20
splits (the paper uses 10), reporting mean +/- std beside the paper's numbers. The fair-metric C
does not depend on eps, so it is built once per split and reused across the grid.

Scope: COMPAS and German only -- both small enough for the dense n x n Sinkhorn solver. Adult
uses the large-n dual path (see budro/tf_dual.py) and is out of scope; a fair-metric matrix too
large for memory is reported as a MemoryError and skipped.

Usage:
    python budro/budro_iclr21.py [compas|german]      # default: compas
"""

import sys
import io
import random
import contextlib
import numpy as np
import scipy.spatial.distance as ssd

import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)   # aif360 logs "No module named 'fairlearn'/'inFairness'" via logging

import budro
txl, script, _ = budro.import_budro()   # BuDRO's boost_sinkhorn_tf + group_metrics
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# compas_proc / german_proc live in these per-dataset dirs of the BuDRO clone, not in scripts/.
for _sub in ("compas/hyper-p_selection/BuDRO", "german"):
    _dir = f"{budro.BUDRO_DIR}/{_sub}"
    if _dir not in sys.path:
        sys.path.append(_dir)

GAMMA_REG = 0.00005                 # entropic regularization (all BuDRO experiments)
SEEDS = [6, 10, 20, 30, 42]         # 5 random 80/20 splits (paper averages 10)

# BuDRO's reported values (Table 1 / Table 3) at their optimal eps, for reference.
PAPER = {
    "compas": (0.12, {"Acc": 0.652, "G-cons": 1.000, "R-cons": 1.000,
                      "sexMax": 0.124, "sexRMS": 0.099, "raceMax": 0.145, "raceRMS": 0.125}),
    "german": (1.0, {"BAcc": 0.715, "S-cons": 0.974, "ageMax": 0.185, "ageRMS": 0.151}),
}


def group_gap(y_true, y_pred, prot):
    """GapRMS, GapMax via BuDRO's own script.group_metrics (label_good=1)."""
    r = script.group_metrics(y_true, y_pred, prot,
                             label_protected=0, label_good=1, verbose=False)
    return float(r[4]), float(r[5])  # rms, max


def fit_sinkhorn(Xtr, ytr, Ctf, param, n_iter, eps, bracket):
    """One BuDRO run with the Sinkhorn solver -> trained xgboost booster.
    boost_sinkhorn_tf prints 'Iter {t}:' every step unconditionally, so we
    swallow its stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        bst, _ = txl.boost_sinkhorn_tf(
            Xtr, ytr, Ctf, n_iter, pred=None, eps=eps, gamma_reg=GAMMA_REG,
            param=param, bst=None, lowg=bracket[0], highg=bracket[1], verbose=False)
    return bst


# ------------------------------------------------------------------ builders
# build_<name>(seed) loads ONE split + its fair-metric C and returns the config
# (params/n_iter/bracket/eps_grid) plus an eval(bst)->metrics closure. C is
# eps-independent, so the caller reuses it across the eps grid.

def build_compas(seed):
    import compas_proc
    (Xtr, Xte, ytr, yte, _ysx_tr, ysx_te, _yr_tr, yr_te, fn, Corig,
     *_rest) = compas_proc.compas_setup(False, False, nsteps=1, baseline=True,
                                        seed=seed, param=None, pct=0.8, loadData=False)
    sind, rind = fn.index("sex"), fn.index("race")
    ytr, yte = ytr.astype(int), yte.astype(int)
    n, nte = Xtr.shape[0], Xte.shape[0]
    num0, num1 = int((ytr == 0).sum()), int((ytr == 1).sum())

    def evaluate(bst):
        pred = (bst.predict(xgb.DMatrix(Xte)) >= 0.5).astype(int)
        s_rms, s_max = group_gap(yte, pred, ysx_te)
        r_rms, r_max = group_gap(yte, pred, yr_te)
        return {"Acc": accuracy_score(yte, pred), "BAcc": balanced_accuracy_score(yte, pred),
                "G-cons": compas_proc.sex_cons(bst, Xte, sind, rind) / nte,
                "R-cons": compas_proc.race_cons(bst, Xte, sind, rind) / nte,
                "sexMax": s_max, "sexRMS": s_rms, "raceMax": r_max, "raceRMS": r_rms}

    # Table 8 (COMPAS): depth 2, eta 1.5e-5, min_weight 0.1/n, lambda 1e-8,
    # scale_pos_weight = #neg/#pos; 68 steps; bracket [-10, 10].
    param = {"objective": "binary:logistic", "nthread": 8, "max_depth": 2,
             "eta": 1.5e-5, "min_child_weight": 0.1 / n, "lambda": 1e-8,
             "scale_pos_weight": num0 / num1, "seed": seed}
    return dict(Xtr=Xtr, ytr=ytr, Corig=Corig, param=param, n_iter=68,
                bracket=(-10.0, 10.0), eps_grid=[0.01, 0.05, 0.075, 0.1, 0.12, 0.15, 0.2, 0.5],
                evaluate=evaluate, prot_idx={"sex": sind, "race": rind})


def build_german(seed):
    import german_proc
    orig = german_proc.get_german_data()
    (Xtr, Xte, ytr, yte, yage_tr, _yage_te, fn, _tri, _tei) = \
        german_proc.get_german_train_test_age(orig, pct=0.8, removeProt=False, seed=seed)
    aind, abind = fn.index("age"), fn.index("age_bin")
    age_grp_te = Xte[:, abind].copy()              # binary age group, for group gaps
    Xtr[:, abind] = 0; Xte[:, abind] = 0           # eliminate binary age (their driver)
    status_inds = [i for i, nm in enumerate(fn) if "personal_status" in nm]

    # fair metric: ridge on age, project out the age subspace, sq-euclidean distances
    RCV = german_proc.train_ridge(Xtr, yage_tr, aind=aind, abind=None)
    proj = german_proc.german_proj_mat(RCV, aind)
    Corig = ssd.squareform(ssd.pdist(Xtr @ proj, metric="sqeuclidean"))

    ytr, yte = ytr.astype(int), yte.astype(int)
    n = Xtr.shape[0]
    num0, num1 = int((ytr == 0).sum()), int((ytr == 1).sum())

    def evaluate(bst):
        pred = (bst.predict(xgb.DMatrix(Xte)) >= 0.5).astype(int)
        a_rms, a_max = group_gap(yte, pred, age_grp_te)
        return {"Acc": accuracy_score(yte, pred), "BAcc": balanced_accuracy_score(yte, pred),
                "S-cons": float(german_proc.status_cons(bst, Xte, status_inds).mean()),
                "ageMax": a_max, "ageRMS": a_rms}

    # Table 4 (German): depth 4, eta 0.005, min_weight 1/80 (=10/n), lambda 1.0,
    # scale_pos_weight = #neg/#pos; 90 steps; bracket [0, 5].
    param = {"objective": "binary:logistic", "nthread": 8, "max_depth": 4,
             "eta": 0.005, "min_child_weight": 10.0 / n, "lambda": 1.0,
             "scale_pos_weight": num0 / num1, "seed": seed}
    return dict(Xtr=Xtr, ytr=ytr, Corig=Corig, param=param, n_iter=90,
                bracket=(0.0, 5.0), eps_grid=[0.5, 1.0, 1.5, 2.0, 2.5], evaluate=evaluate,
                prot_idx={"age": aind})   # age_bin (abind) is zeroed out, can't be split on


BUILDERS = {"compas": build_compas, "german": build_german}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "compas"
    if name not in BUILDERS:
        sys.exit(f"unknown dataset '{name}'; choose from {list(BUILDERS)}")
    build = BUILDERS[name]
    print(f"==== BuDRO ICLR'21 reproduction: {name}, eps sweep x {len(SEEDS)} splits ====", flush=True)

    # per_eps[eps] = list of metric dicts, one per split
    per_eps, eps_grid, metrics = {}, None, None
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        print(f"\n---------------- seed {seed} ----------------------", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # silence BuDRO's load-time prints
                ctx = build(seed)
        except MemoryError:
            sys.exit(f"MemoryError: {name} fair-metric matrix too large for the "
                     f"Sinkhorn solver -- needs the dual-SGD path (out of scope).")
        Ctf = tf.constant(ctx["Corig"], tf.float32)
        eps_grid = ctx["eps_grid"]
        for eps in eps_grid:
            print(f"\n[seed {seed} | eps {eps:g}]", flush=True)
            bst = fit_sinkhorn(ctx["Xtr"], ctx["ytr"], Ctf, ctx["param"],
                               ctx["n_iter"], eps, ctx["bracket"])
            m = ctx["evaluate"](bst)
            metrics = list(m)
            per_eps.setdefault(eps, []).append(m)
            print("  -> " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)

    # ---- aggregate: per-eps mean +/- std over splits ----
    opt_eps, paper = PAPER[name]
    print(f"\n===== {name}: mean +/- std over {len(SEEDS)} splits (paper-optimal eps={opt_eps}*) =====")
    print(f"{'eps':>6} | " + " ".join(f"{m:>15}" for m in metrics))
    print("-" * (9 + 16 * len(metrics)))
    for eps in eps_grid:
        rows = per_eps[eps]
        cells = []
        for m in metrics:
            v = np.array([r[m] for r in rows])
            cells.append(f"{v.mean():.3f}+/-{v.std():.3f}")
        mark = "*" if eps == opt_eps else " "
        print(f"{eps:>5g}{mark}| " + " ".join(f"{c:>15}" for c in cells))
    print("\npaper @ optimal eps: " + "  ".join(f"{k}={v}" for k, v in paper.items()))


if __name__ == "__main__":
    main()
