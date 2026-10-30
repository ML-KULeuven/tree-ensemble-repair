from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.utils import *
from policy.tree_ensemble_helpers import *

MAX_FAULT_WEIGHT = 1e3

# stable softmax to avoid NaNs
def _softmax_stable(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    denom = Z.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0  # guard (should not happen after stabilization)
    Z /= denom
    return Z

def _argmax_applicable(row_logits: np.ndarray, app: Optional[List[int]]) -> int:
    if app is None or len(app) == 0:
        return int(np.argmax(row_logits))
    subidx = int(np.argmax(row_logits[app]))
    return int(app[subidx])

def _make_ce_plus_nl_obj(K: int, fault_mask_train: np.ndarray, lam: float, hessian_mode: str):
        """
        Create a custom XGBoost objective function that combines:
        - Cross-Entropy (CE) loss on normal rows
        - Negative Loss (NL) on fault rows
        
        CE:  L_CE = -log(P(y_true|x))
        NL:  L_NL = -log(1 - P(forbidden_class|x))
        """
        eps = 1e-12
        fault_mask_train = np.asarray(fault_mask_train, dtype=bool)

        def obj(y_pred: np.ndarray, dmat: xgb.DMatrix):
            y_true = dmat.get_label().astype(int)
            w      = dmat.get_weight()
            if w is None or w.size == 0:
                w = np.ones_like(y_true, dtype=float)

            Nrows = y_true.shape[0]
            y_pred = np.asarray(y_pred)
            Z = y_pred.reshape(Nrows, K) if y_pred.ndim == 1 else y_pred

            P = _softmax_stable(Z.copy())
            G = np.zeros_like(P)
            H = np.zeros_like(P)

            # CE on normal rows
            normal_rows = ~fault_mask_train
            G[normal_rows, :] = P[normal_rows, :]
            G[normal_rows, y_true[normal_rows]] -= 1.0
            H[normal_rows, :] = P[normal_rows, :] * (1.0 - P[normal_rows, :])

            # NL on fault rows
            rows = np.flatnonzero(fault_mask_train)
            if rows.size > 0:
                i_forbid = y_true[rows]
                Pi = P[rows, i_forbid]
                denom = np.maximum(1.0 - Pi, eps)

                if hessian_mode == "safe":
                    H[rows, :] += P[rows, :] * (1.0 - P[rows, :])
                else:
                    raise ValueError("hessian_mode must be 'safe'.")

                P_rows = P[rows, :]
                P_mask = P_rows.copy()
                P_mask[np.arange(rows.size), i_forbid] = 0.0
                G[rows, :]         += -lam * (Pi[:, None] * P_mask) / denom[:, None]
                G[rows, i_forbid]  +=  lam * Pi

            # apply weights
            G *= w[:, None]
            H *= w[:, None]
            return G.reshape(-1), H.reshape(-1)

        return obj


def retrain_ensemble(
    df_clean: pd.DataFrame,                     # normal training data
    y_clean: pd.Series,                         # normal labels
    Fx: pd.DataFrame,                           # fault states
    Fy: pd.Series,                              # forbidden labels
    Aa: List[List[int]],                        # applicable actions per fault
    fault_weights: np.ndarray,                        # fault weights multipliers (len == len(Fx))
    model_type: str = "xgb",
    params: Optional[Dict[str, Any]] = None,    # params for xgb
    # prev_bst: Optional[xgb.Booster] = None,   # (optional) previous model to continue training from
    num_boosting_rounds: Optional[int] = None, 
    retrain_method: str = "retraining-loss",    # "retraining-loss" or "retraining-augment"
    lam: float = 1.0,                           # lambda for NL loss
    violator_boost: float = 2.0,                # multiplier for faults that still violate
    hessian_mode: str = "safe",                 # hessian mode for custom loss
    benchmark: Optional[str] = None,            # used to round split thresholds in discrete domains
):
    """
    Retrain a multiclass XGBoost model to avoid faults using either:
    - Custom loss (CE + NL) where:
        * Normals use standard cross-entropy (CE) against the teacher label.
        * Faults use Negative Loss (NL) against the forbidden class.
    - Augmented training set where fault states are paired with NN-corrected labels

    Each fault's weight is multiplied by a `violator_boost` if it still violates after retraining.

    Returns:
        AddTree, Booster, still_unsafe (bool)
    """

    assert model_type == "xgb", "Retraining is implemented for XGBoost only."
    assert len(Fx) == len(Fy) == len(Aa), "Fault states, labels, and applicable actions must align."
    assert len(Fx) > 0, "No faults provided for retraining."
    # assert len(fault_weights) == len(Fx), "Weights array length mismatch."

    # 1. prepare data
    K = int((params or {}).get("num_class", int(max(y_clean.max(), Fy.max())) + 1)) # num classes

    w_norm = np.ones(len(df_clean), dtype=float)
    w_fault = np.array([fault_weights[tuple(row)] for row in Fx.to_numpy() ], dtype=float)
   
    # stack together normals + faults
    X_aug = np.vstack([df_clean.to_numpy(), Fx.to_numpy()])
    y_aug = np.concatenate([y_clean, Fy])
    w_aug = np.concatenate([w_norm, w_fault])
    is_fault = np.array([False]*len(df_clean) + [True]*len(Fx.to_numpy()))

    # split: ALL faults -> train; normals split into train/val
    idx_norm = np.flatnonzero(~is_fault)
    idx_fault = np.flatnonzero(is_fault)
    idx_norm_tr, idx_norm_va = train_test_split(idx_norm, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    train_idx = np.concatenate([idx_norm_tr, idx_fault], axis=0)
    val_idx = idx_norm_va   

    Xtrain, Xval = X_aug[train_idx], X_aug[val_idx]
    ytrain, yval = y_aug[train_idx], y_aug[val_idx]
    w_train = w_aug[train_idx]
    ftr = is_fault[train_idx]
    fault_train_positions = np.flatnonzero(ftr)
    # guard weights
    # w_train = np.clip(w_train, 1e-6, np.inf)

    # map applicable actions to the training rows (for fault rows only)
    app_actions_train = [None] * Xtrain.shape[0]
    for k, pos in enumerate(fault_train_positions):
        app_actions_train[pos] = list(map(int, Aa[k]))  

    # set XGB parameters 
    def _build_params(base: Optional[Dict[str, Any]], K: int) -> Tuple[Dict[str, Any], int]:
        p = dict(base or {})
        p.update(get_params(model_type, y_aug))
        for k in ("ablation", "lr", "model_type"):
            p.pop(k, None)
        p["num_class"]   = K
        p["objective"]   = "multi:softprob"
        p["eval_metric"] = "mlogloss"
        p.setdefault("tree_method", "hist")
        p.setdefault("random_state", SEED)
        # Regularization (tmp off)
        # p.setdefault("reg_lambda", 1.0)
        # p.setdefault("max_delta_step", 4.0)

        rounds = p.pop("num_trees", 20)
        return p, rounds
    params_xgb, default_rounds = _build_params(params, K)
    if num_boosting_rounds is None:
        num_boosting_rounds = default_rounds

    # build dmatrix
    dtrain = xgb.DMatrix(Xtrain, label=ytrain, weight=w_train)
    dval = xgb.DMatrix(Xval, label=yval)

    # 2. retrain using selected method
    t0 = time.time()
    if retrain_method == "retraining-loss":
        # craft custom objective for XGB: CE (normal) + NL (fault)
        obj = _make_ce_plus_nl_obj(K, ftr, lam=lam, hessian_mode=hessian_mode)

        bst = xgb.train(
            params_xgb, 
            dtrain, 
            num_boost_round=num_boosting_rounds, 
            obj=obj,
            evals=[(dval, "val")], # only show logloss on val (train has faults, so not comparable)
            # xgb_model=prev_bst, 
            verbose_eval=False, # True,
        )
    elif retrain_method == "retraining-augment":
        bst = xgb.train(
            params_xgb, 
            dtrain, 
            num_boost_round=num_boosting_rounds, 
            evals=[(dval, "val")], 
            verbose_eval=False
        )
    else:
        raise ValueError(f"Unknown retrain method: {retrain_method}")
    print(f"\n\t- xgb ensemble retrained in {time.time() - t0:.3f} seconds.")

    # 3. count faults that have not been avoided even after retraining
    raw = bst.predict(dtrain, output_margin=True).reshape(-1, K)
    yhat_app = np.fromiter((_argmax_applicable(raw[i], app_actions_train[i]) for i in range(raw.shape[0])),
                           dtype=int, count=raw.shape[0])
    if retrain_method == "retraining-loss":
        viol_mask_train = ftr & (yhat_app == ytrain)
    elif retrain_method == "retraining-augment":
        # NOTE: any action different from teacher is a violation here (even if safe)
        viol_mask_train = ftr & (yhat_app != ytrain)
        # @TODO: use a lighter violation definition?
        #        --> violation: predicted class is unsafe?
        # ...
    num_viol = int(viol_mask_train.sum())
    # viol_fault_mask = viol_mask_train[fault_train_positions]    
    print(f"\t--> after re-train: train-fault violations = {num_viol}")
    # update weights for these violators
    # weights[viol_fault_mask] *= violator_boost
    # for i in np.flatnonzero(viol_mask_train):
    #     state_key = tuple(Xtrain[i])
    #     fault_weights[state_key] *= violator_boost
    # print output per class, sleected class, applicable actions and selected action per each violator
    # for i in np.flatnonzero(viol_mask_train):
    #     state_key = tuple(Xtrain[i])
    #     print(f"\t [violator] state: {state_key}, forbidden: {ytrain[i]}, all predicted: {raw[i]}, selecyed action: {yhat_app[i]}, applicable: {app_actions_train[i]}")
    #     break
    
    # TMP cap max weights to avoid overflow at 
    for i in np.flatnonzero(viol_mask_train):
        state_key = tuple(Xtrain[i])
        fault_weights[state_key] = min(fault_weights[state_key] * violator_boost,
                                    MAX_FAULT_WEIGHT)

    # 4. report quick metrics about how retrained model is doing
    def _predict_proba(model: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
        raw = model.predict(dmat, output_margin=True)
        return _softmax_stable(raw)
    probs_tr = _predict_proba(bst, dtrain)
    probs_va = _predict_proba(bst, dval)
    yhat_tr = np.fromiter(
        (_argmax_applicable(probs_tr[i], app_actions_train[i]) for i in range(probs_tr.shape[0])),
        dtype=int, count=probs_tr.shape[0]
    )
    yhat_va = np.fromiter(
        (_argmax_applicable(probs_va[i], None) for i in range(probs_va.shape[0])),
        dtype=int, count=probs_va.shape[0]
    )   
    normal_tr_mask = ~ftr
    acc_tr_normal = (
        accuracy_score(ytrain[normal_tr_mask], yhat_tr[normal_tr_mask])
        if np.any(normal_tr_mask) else float("nan")
    )
    acc_va_normal = accuracy_score(yval, yhat_va) if yval.size else float("nan") # val has only normal rows
    
    if retrain_method == "retraining-loss":
        avoid_tr = (
            float(np.mean(yhat_tr[ftr] != ytrain[ftr]))
            if np.any(ftr) else float("nan")
        )
    elif retrain_method == "retraining-augment":
        # see comment when defining violations 
        # this is currently indicating how many faults match NN-corrected label
        # (i.e., if we chose a suboptimal safe action, this is still counted as not avoided)
        avoid_tr = (
            float(np.mean(yhat_tr[ftr] == ytrain[ftr]))
            if np.any(ftr) else float("nan")
        )
    print(
        "\n\t- retrained ensemble metrics:\n"
        f"\t * train normal acc: {acc_tr_normal:.3f}, "
        f"valid normal acc: {acc_va_normal:.3f}\n"
        f"\t * train fault avoidance: {avoid_tr:.3f}"
    )

    # 5. convert to AddTree, verify conversion, return
    at = veritas.get_addtree(bst, silent=True)
    try:
        ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-3)
    except Exception as e:
        ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-2)
    assert ok is True, "Something wrong with conversion to AddTree!"
    if benchmark is not None:
        at = round_splits_if_discrete(at, benchmark)
    return at, bst, (num_viol > 0)#, weights



# OLD VERSIONS

# AAAI26
# old function
# def retrain_ensemble(
#     df,                                # pandas DataFrame with training set + faults
#     y,                                 # labels (teacher on normal rows, forbidden class on faults)
#     is_fault,                          # boolean array masking fault rows
#     applicable_actions,                # list of actions allowed for each fault
#     model_type="xgb",
#     params=None,                          
#     # loss / optimization controls
#     lam: float = 1.0,                           # base NL penalty weight
#     hessian_mode: str = "safe",                 # {"safe", "exact_diag"}
#     fault_weight: float = 20.0,                 # base weight for all fault rows
#     # tighten_rounds: int = 10,                 # extra trees added per tighten pass
#     # tighten_max_passes: int = 10,             # max tighten passes
# ) -> Tuple[veritas.AddTree, xgb.Booster, bool]:
#     """
#     Retrain a multiclass XGBoost model with a custom CE+NL objective:
#       - Normal rows use standard cross-entropy (CE) against the teacher label.
#       - Fault rows encode the *forbidden* class id in `y`, and use Negative Loss:
#             L_NL = -log(1 - P(forbidden_class|x))

#     The function:
#       1) Puts *all* fault rows in the training split (hardening the current fault set).
#       2) Uses higher sample weights on fault rows.
#       3) Optionally re-trains with higher lambda/fault weight if any violations remain.

#     Returns:
#       AddTree     : converted tree ensemble for your pipeline
#       Booster     : trained XGBoost booster
#     """
#     assert model_type == "xgb", "Retraining is implemented for XGBoost only."
#     assert len(df) == len(y) == len(is_fault), "df, y, and is_fault must have the same length."

#     # 1. prepare data
#     X = np.asarray(df)
#     yy = np.asarray(y, dtype=int).reshape(-1)
#     fault_mask = np.asarray(is_fault, dtype=bool)
#     N = X.shape[0]
#     # K = int(np.max(yy)) + 1
#     K = int((params or {}).get("num_class", int(np.max(yy)) + 1))
#     # print(f"N = {N}  K = {K}")

#     # split: ALL faults -> train; normals split into train/val
#     idx_fault = np.flatnonzero(fault_mask)
#     idx_norm  = np.flatnonzero(~fault_mask)

#     idx_norm_tr, idx_norm_va = train_test_split(
#         idx_norm, test_size=TEST_SIZE, random_state=SEED, shuffle=True
#     )
#     train_idx = np.concatenate([idx_norm_tr, idx_fault], axis=0)
#     val_idx   = idx_norm_va

#     Xtrain, Xval = X[train_idx], X[val_idx]
#     ytrain, yval = yy[train_idx], yy[val_idx]
#     ftr = fault_mask[train_idx]  # training-fault mask

#     # map applicable actions to the training rows (for fault rows only)
#     app_actions_train = [None] * Xtrain.shape[0]
#     fault_app_list = [list(map(int, a)) for a in applicable_actions]
#     fault_train_positions = np.flatnonzero(ftr)
#     for k, pos in enumerate(fault_train_positions):
#         app_actions_train[pos] = fault_app_list[k]

#     # print(f"N_train = {Xtrain.shape[0]}  (faults in train: {int(ftr.sum())})")

#     # 2. craft custom objective for XGB: CE (normal) + NL (fault)
#     def _make_ce_plus_nl_obj(K: int, fault_mask_train: np.ndarray, lam: float, hessian_mode: str):
#         eps = 1e-12
#         fault_mask_train = np.asarray(fault_mask_train, dtype=bool)

#         def obj(y_pred: np.ndarray, dmat: xgb.DMatrix):
#             # extract true labels & weights
#             y_true = dmat.get_label().astype(int)
#             w = dmat.get_weight()
#             if w is None or w.size == 0:
#                 w = np.ones_like(y_true, dtype=float)

#             Nrows = y_true.shape[0] 
#             y_pred = np.asarray(y_pred)
#             if y_pred.ndim == 1:
#                 if y_pred.size != Nrows * K:
#                     raise ValueError(f"Expected {Nrows*K} margins, got {y_pred.size}.")
#                 Z = y_pred.reshape(Nrows, K)
#             elif y_pred.shape == (Nrows, K):
#                 Z = y_pred
#             else:
#                 raise ValueError(f"Bad margins shape {y_pred.shape}; expected ({Nrows}, {K}).")

#             P = _softmax_stable(Z.copy())  # (N, K) output class probabilities

#             G = np.zeros_like(P)    # gradients
#             H = np.zeros_like(P)    # hessians

#             # CE on normal rows
#             # gradient:  P - 1{y_true}
#             # hessian:   P * (1 - P)
#             normal_rows = ~fault_mask_train
#             assert normal_rows.size == Nrows, "normal_rows mask has wrong size??"
#             G[normal_rows, :] = P[normal_rows, :]
#             G[normal_rows, y_true[normal_rows]] -= 1.0
#             H[normal_rows, :] = P[normal_rows, :] * (1.0 - P[normal_rows, :])

#             # NL on fault rows: L = -log(1 - P_i) for forbidden class i
#             assert sum(fault_mask_train) >0, "No fault rows in training set??"
#             rows = np.flatnonzero(fault_mask_train)
#             i_forbid = y_true[rows]                     # all forbidden class indices (M,)
#             Pi = P[rows, i_forbid]                      # all forbidden class probs (M,)
#             denom = np.maximum(1.0 - Pi, eps)           # avoid div-by-0 when doing 1-Pi

#             # safe positive-diagonal Hessian
#             if hessian_mode == "safe":
#                 H[rows, :] += P[rows, :] * (1.0 - P[rows, :])
#             elif hessian_mode == "exact_diag":
#                 # TODO needs checking
#                 # H[rows, i_forbid] += Pi * (1.0 - Pi)
#                 # H[rows, :] += P[rows, :] * (1.0 - P[rows, :])
#                 raise NotImplementedError("exact_diag Hessian not implemented yet.")
#             else:
#                 raise ValueError("hessian_mode must be 'safe' or 'exact_diag'.")

#             # gradients
#             P_rows = P[rows, :]
#             P_mask = P_rows.copy()
#             P_mask[np.arange(rows.size), i_forbid] = 0.0
#             G[rows, :]           += -lam * (Pi[:, None] * P_mask) / denom[:, None]
#             G[rows, i_forbid]    +=  lam * Pi

#             # apply weights
#             G *= w[:, None]
#             H *= w[:, None]
#             return G.reshape(-1), H.reshape(-1)

#         return obj

#     # 3. set XGB parameters 
#     def _build_params(base: Optional[Dict[str, Any]], K: int) -> Tuple[Dict[str, Any], int]:
#         p = dict(base or {})
#         p.update(get_params(model_type, yy))  # your helper to get template params

#         # remove keys xgboost doesn't recognize in xgb.train
#         for k in ("ablation", "lr", "model_type"):
#             p.pop(k, None)

#         # force multiclass layout + robust metric
#         p["num_class"] = K
#         p["objective"] = "multi:softprob"
#         p["eval_metric"] = "mlogloss"            # avoid misleading 'merror' with faults in labels
#         p.setdefault("tree_method", "hist")
#         p.setdefault("random_state", SEED)

#         # keep margins in safe numeric range (--> prevents NaNs in downstream naive softmax)
#         # AAAI rebuttal: strong regularization 
#         # p.setdefault("reg_lambda", 6.0)          
#         # p.setdefault("max_delta_step", 1.0)      # limit per-tree logit jumps
#         # ICAPS26: moderate regularization
#         p.setdefault("reg_lambda", 1.0)          
#         p.setdefault("max_delta_step", 4.0)

#         # pull training rounds (from either naming convention)
#         # rounds = p.pop("n_estimators", None)
#         # if rounds is None:
#         #     rounds = p.pop("num_trees", 100)
#         rounds = p.pop("num_trees", 20)

#         return p, rounds

#     params_xgb, num_boost_rounds = _build_params(params, K)

#     # print("Using XGBoost params:")
#     # for k, v in params_xgb.items():
#     #     print(f"  {k}: {v}")

#     # 4. build DMatrix (with weights)
#     w_train = np.ones_like(ytrain, dtype=float)
#     w_train[ftr] = float(fault_weight)
#     dtrain = xgb.DMatrix(Xtrain, label=ytrain, weight=w_train)
#     dval = xgb.DMatrix(Xval, label=yval, weight=np.ones_like(yval, dtype=float))

#     # 5. train
#     obj = _make_ce_plus_nl_obj(K, ftr, lam=lam, hessian_mode=hessian_mode)
#     t0 = time.time()
#     print(f"\t... training new ensemble with {num_boost_rounds} trees ...")
#     bst = xgb.train(
#         params_xgb,
#         dtrain,
#         num_boost_round=num_boost_rounds,
#         obj=obj,
#         # evals=[(dtrain, "train"), (dval, "val")],  # val has only normal rows
#         evals=[(dval, "val")],  # only show logloss on val (train has faults, so not comparable)
#         # early_stopping_rounds=10,
#         verbose_eval=True,
#     )
#     print(f"\n\t- new xgb ensemble retrained (CE+NL) in {time.time() - t0:.3f} seconds.")

#     # 6. count faults that have not been avoided even after retraining
#     def _count_train_fault_violations(model: xgb.Booster) -> Tuple[int, np.ndarray]:
#         raw = model.predict(dtrain, output_margin=True).reshape(-1, K)
#         yhat_app = np.fromiter(
#             (_argmax_applicable(raw[i], app_actions_train[i]) for i in range(raw.shape[0])),
#             dtype=int, count=raw.shape[0]
#         )
#         viol_mask = ftr & (yhat_app == ytrain)
#         return int(viol_mask.sum()), viol_mask

#     num_viol, viol_mask = _count_train_fault_violations(bst)
#     print(f"\t--> after re-train: train-fault violations = {num_viol}")

#     # TMP: comment out tightening rounds, just one retrain per loop iteration
#     # 7. optional tightening if any training-fault violations remain
#     # # a) if any violations, do tightening passes where we:
#     # #       1. increase lambda
#     # #       2. upweight still-violating rows
#     # lam_local = float(lam)
#     # passes = 0
#     # # make weight increases cumulative across passes by updating w_train in-place
#     # while num_viol > 0 and passes < tighten_max_passes:
#     #     bst0 = bst.copy()
#     #     passes += 1
#     #     lam_local *= 2.0  # stronger NL
#     #     w_train[viol_mask] *= 2.0 # double weight of still-violating rows every time
#     #     w_step = w_train.copy()
#     #     # uncomment to only double weight once
#     #     # w_step[viol_mask] *= 2.0
#     #     dtrain.set_weight(w_step)

#     #     print(f"\n\t* tighten pass {passes}: lam -> {lam_local:.3f}, upweighted {int(viol_mask.sum())} violators")
#     #     obj = _make_ce_plus_nl_obj(K, ftr, lam=lam_local, hessian_mode=hessian_mode)
#     #     # print(f"\t... adding {tighten_rounds} more trees ...")
#     #     # full retrain from scratch 
#     #     bst = xgb.train(
#     #         params_xgb,
#     #         dtrain,
#     #         num_boost_round=num_boost_rounds,
#     #         obj=obj,
#     #         evals=[(dval, "val")],  # only show logloss on val (train has faults, so not comparable)
#     #         verbose_eval=False,
#     #     )
#     #     num_viol, viol_mask = _count_train_fault_violations(bst)
#     #     print(f"\t--> remaining train-fault violations = {num_viol}")

#     # # b) if there are STILL violations, repeat previous loop by adding more trees each pass
#     # if num_viol > 0:
#     #     print(f"\n\t* STILL {num_viol} train-fault violations after {passes} passes, "
#     #           f"repeating loop + adding {tighten_rounds} trees per pass.")
#     #     bst = bst0.copy()  # restart from first retrained model with lambda=1
#     #     passes, lam_local = 0, float(lam)
#     #     # reset w_train to original base weights before extra tighten passes
#     #     # so these extra passes start from the original weighting scheme
#     #     w_train = np.ones_like(ytrain, dtype=float)
#     #     w_train[ftr] = float(fault_weight)
#     #     while num_viol > 0 and passes < tighten_max_passes:
#     #         passes += 1
#     #         lam_local *= 2.0  # stronger NL
#     #         w_train[viol_mask] *= 2.0
#     #         w_step = w_train.copy()
#     #         # uncomment to only double weight once
#     #         # w_step[viol_mask] *= 2.0
#     #         dtrain.set_weight(w_step)

#     #         print(f"\n\t* extra tighten pass {passes}: lam -> {lam_local:.3f}, upweighted {int(viol_mask.sum())} violators")
#     #         obj = _make_ce_plus_nl_obj(K, ftr, lam=lam_local, hessian_mode=hessian_mode)
#     #         print(f"\t... adding {tighten_rounds} more trees ...")
#     #         bst = xgb.train(
#     #             params_xgb,
#     #             dtrain,
#     #             num_boost_round=tighten_rounds, # add tighten_rounds more trees
#     #             obj=obj,
#     #             evals=[(dval, "val")],  # only show logloss on val (train has faults, so not comparable)
#     #             xgb_model=bst,
#     #             verbose_eval=False,
#     #         )
#     #         num_viol, viol_mask = _count_train_fault_violations(bst)
#     #         print(f"\t--> remaining train-fault violations = {num_viol}")

#     # 8. report quick metrics
#     def _predict_proba(model: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
#         raw = model.predict(dmat, output_margin=True)
#         return _softmax_stable(raw)

#     # probs_tr = _predict_proba(bst, dtrain)
#     # probs_va = _predict_proba(bst, dval)
#     # yhat_tr = probs_tr.argmax(axis=1)
#     # yhat_va = probs_va.argmax(axis=1)

#     probs_tr = _predict_proba(bst, dtrain)
#     probs_va = _predict_proba(bst, dval)
#     yhat_tr = np.fromiter(
#         (_argmax_applicable(probs_tr[i], app_actions_train[i]) for i in range(probs_tr.shape[0])),
#         dtype=int, count=probs_tr.shape[0]
#     )
#     yhat_va = probs_va.argmax(axis=1)  # val has only normal rows


#     normal_tr_mask = ~ftr
#     acc_tr_normal = (
#         accuracy_score(ytrain[normal_tr_mask], yhat_tr[normal_tr_mask])
#         if np.any(normal_tr_mask) else float("nan")
#     )
#     acc_va_normal = accuracy_score(yval, yhat_va) if yval.size else float("nan")

#     fault_tr_mask = ftr
#     avoid_tr = (
#         float(np.mean(yhat_tr[fault_tr_mask] != ytrain[fault_tr_mask]))
#         if np.any(fault_tr_mask) else float("nan")
#     )

#     print(
#         "\n\t- retrained ensemble metrics:\n"
#         f"\t * train normal acc: {acc_tr_normal:.3f}, "
#         f"valid normal acc: {acc_va_normal:.3f}"
#     )
#     # print(f"\t * train fault avoidance: {avoid_tr:.3f}")
#     print(f"\t * train fault avoidance (applicable-argmax): {avoid_tr:.3f}")

#     # 9. convert to AddTree, verify conversion, return
#     at = veritas.get_addtree(bst, silent=True)
#     try:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-3, silent=True)
#     except Exception as e:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-2, silent=True)
#     assert ok is True, "Something wrong with conversion to AddTree!"
#     return at, bst, num_viol > 0

# earlier versions ICAPS26

# def retrain_ensemble_custom_loss(
#     df_clean: pd.DataFrame,                     # normal training data
#     y_clean: pd.Series,                         # normal labels
#     Fx: pd.DataFrame,                           # fault states
#     Fy: pd.Series,                              # forbidden labels
#     Aa: List[List[int]],                        # applicable actions per fault
#     fault_weight_multipliers: np.ndarray,       # fault weights multipliers (len == len(Fx))
#     model_type: str = "xgb",
#     params: Optional[Dict[str, Any]] = None,
#     # prev_bst: Optional[xgb.Booster] = None,
#     num_boosting_rounds: Optional[int] = None,
#     lam: float = 1.0,
#     violator_boost: float = 2.0,                # multiplier for faults that still violate
#     hessian_mode: str = "safe",
# ) -> Tuple[veritas.AddTree, xgb.Booster, bool, np.ndarray]:
#     """
#     Retrain a multiclass XGBoost model with a custom CE+NL objective:
#     - Normal rows use standard cross-entropy (CE) against the teacher label.
#     - Fault rows encode the *forbidden* class id in `y`, and use Negative Loss:
#         L_NL = -log(1 - P(forbidden_class|x))

#     - Normals weight = 1.0
#     - Faults weight = base_fault_weight * fault_weight_multipliers
#     - fault_weight_multipliers updated in-place for faults that still violate
#     Returns:
#         AddTree, Booster, still_unsafe (bool), updated fault_weight_multipliers
#     """
#     assert model_type == "xgb", "Retraining is implemented for XGBoost only."
#     assert len(Fx) == len(Fy) == len(Aa), "Fault states, labels, and applicable actions must align."
#     assert len(Fx) > 0, "No fault rows provided for retraining."
#     assert len(fault_weight_multipliers) == len(Fx), "Multiplier array length mismatch."

#     # prepare data
#     X_norm = df_clean.to_numpy()
#     y_norm = np.asarray(y_clean).reshape(-1)
#     X_fault = Fx.to_numpy()
#     y_fault = np.asarray(Fy).reshape(-1)

#     K = int((params or {}).get("num_class", int(max(y_norm.max(), y_fault.max())) + 1))

#     # compute weights
#     w_norm = np.ones(len(X_norm), dtype=float)
#     w_fault = violator_boost * fault_weight_multipliers 

#     # stack together normals + faults
#     X_aug = np.vstack([X_norm, X_fault]) 
#     y_aug = np.concatenate([y_norm, y_fault]) 
#     w_aug = np.concatenate([w_norm, w_fault]) 
#     is_fault_aug = np.array([False]*len(X_norm) + [True]*len(X_fault))

#     # split: ALL faults -> train; normals split into train/val
#     idx_norm = np.flatnonzero(~is_fault_aug)
#     idx_fault = np.flatnonzero(is_fault_aug)
#     idx_norm_tr, idx_norm_va = train_test_split(idx_norm, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
#     train_idx = np.concatenate([idx_norm_tr, idx_fault], axis=0)
#     val_idx = idx_norm_va

#     Xtrain, Xval = X_aug[train_idx], X_aug[val_idx]
#     ytrain, yval = y_aug[train_idx], y_aug[val_idx]
#     w_train = w_aug[train_idx]
#     ftr = is_fault_aug[train_idx]
#     fault_train_positions = np.flatnonzero(ftr)

#     # map applicable actions to the training rows (for fault rows only)
#     app_actions_train = [None] * Xtrain.shape[0]
#     for k, pos in enumerate(fault_train_positions):
#         app_actions_train[pos] = list(map(int, Aa[k]))

#     # craft custom objective for XGB: CE (normal) + NL (fault)
#     obj = _make_ce_plus_nl_obj(K, ftr, lam=lam, hessian_mode=hessian_mode)

#     # set XGB parameters 
#     def _build_params(base: Optional[Dict[str, Any]], K: int) -> Tuple[Dict[str, Any], int]:
#         p = dict(base or {})
#         p.update(get_params(model_type, y_aug))
#         for k in ("ablation", "lr", "model_type"):
#             p.pop(k, None)
#         p["num_class"]   = K
#         p["objective"]   = "multi:softprob"
#         p["eval_metric"] = "mlogloss"
#         p.setdefault("tree_method", "hist")
#         p.setdefault("random_state", SEED)
#         # Regularization 
#         p.setdefault("reg_lambda", 1.0)
#         p.setdefault("max_delta_step", 4.0)

#         rounds = p.pop("num_trees", 20)
#         return p, rounds
#     params_xgb, default_rounds = _build_params(params, K)
#     if num_boosting_rounds is None:
#         num_boosting_rounds = default_rounds

#     # guard weights
#     w_train = np.clip(w_train, 1e-6, np.inf)

#     # retrain
#     dtrain = xgb.DMatrix(Xtrain, label=ytrain, weight=w_train)
#     dval = xgb.DMatrix(Xval, label=yval)
#     t0 = time.time()
#     bst = xgb.train(
#         params_xgb, 
#         dtrain, 
#         num_boost_round=num_boosting_rounds, 
#         obj=obj,
#         evals=[(dval, "val")], # only show logloss on val (train has faults, so not comparable)
#         # xgb_model=prev_bst, 
#         verbose_eval=False, # True,
#     )
#     print(f"\n\t- new xgb ensemble retrained (CE+NL) in {time.time() - t0:.3f} seconds.")

#     # count faults that have not been avoided even after retraining
#     raw = bst.predict(dtrain, output_margin=True).reshape(-1, K)
#     yhat_app = np.fromiter((_argmax_applicable(raw[i], app_actions_train[i]) for i in range(raw.shape[0])),
#                            dtype=int, count=raw.shape[0])
#     viol_mask_train = ftr & (yhat_app == ytrain)
#     num_viol = int(viol_mask_train.sum())
#     viol_fault_mask = viol_mask_train[fault_train_positions]

#     print(f"\t--> after re-train: train-fault violations = {num_viol}")

#     # update multipliers cumulatively
#     fault_weight_multipliers[viol_fault_mask] *= violator_boost


#     # report quick metrics
#     def _predict_proba(model: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
#         raw = model.predict(dmat, output_margin=True)
#         return _softmax_stable(raw)
#     probs_tr = _predict_proba(bst, dtrain)
#     probs_va = _predict_proba(bst, dval)
#     yhat_tr = np.fromiter(
#         (_argmax_applicable(probs_tr[i], app_actions_train[i]) for i in range(probs_tr.shape[0])),
#         dtype=int, count=probs_tr.shape[0]
#     )
#     yhat_va = probs_va.argmax(axis=1)   # val has only normal rows
#     normal_tr_mask = ~ftr
#     acc_tr_normal = (
#         accuracy_score(ytrain[normal_tr_mask], yhat_tr[normal_tr_mask])
#         if np.any(normal_tr_mask) else float("nan")
#     )
#     acc_va_normal = accuracy_score(yval, yhat_va) if yval.size else float("nan")
#     avoid_tr = (
#         float(np.mean(yhat_tr[ftr] != ytrain[ftr]))
#         if np.any(ftr) else float("nan")
#     )
#     print(
#         "\n\t- retrained ensemble metrics:\n"
#         f"\t * train normal acc: {acc_tr_normal:.3f}, "
#         f"valid normal acc: {acc_va_normal:.3f}\n"
#         f"\t * train fault avoidance (applicable-argmax): {avoid_tr:.3f}"
#     )

#     # convert to AddTree, verify conversion, return
#     at = veritas.get_addtree(bst, silent=True)
#     try:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-3, silent=True)
#     except Exception as e:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-2, silent=True)
#     assert ok is True, "Something wrong with conversion to AddTree!"

#     return at, bst, (num_viol > 0), fault_weight_multipliers


# def retrain_ensemble_augment(
#     df_clean: pd.DataFrame,
#     y_clean: pd.Series,
#     Fx: pd.DataFrame,
#     Cy: pd.Series,  # corrected labels from NN
#     fault_weight_multipliers: np.ndarray,
#     Aa: List[List[int]],  # applicable actions for violation check
#     params: Dict[str, Any],
#     num_boosting_rounds: int,
#     violator_boost: float = 2.0,
# ) -> Tuple[veritas.AddTree, xgb.Booster, bool, np.ndarray, int]:
#     """
#     Retrain XGBoost on augmented data (normals + faults with NN labels).
#       - Progressive weight boosting for persistent violators.
#       - Optional model size increase when weights grow.
#     Returns:
#       AddTree, Booster, still_unsafe_flag, updated_weights, updated_num_trees
#     """
#     assert len(Fx) == len(Cy) == len(fault_weight_multipliers), "Fault arrays must align."

#     # Prepare augmented data
#     X_norm = df_clean.to_numpy()
#     y_norm = np.asarray(y_clean).reshape(-1)
#     X_fault = Fx.to_numpy()
#     y_fault = np.asarray(Cy).reshape(-1)

#     X_aug = np.vstack([X_norm, X_fault])
#     y_aug = np.concatenate([y_norm, y_fault])
#     w_norm = np.ones(len(X_norm), dtype=float)
#     w_fault = fault_weight_multipliers.astype(float)
#     w_aug = np.concatenate([w_norm, w_fault])

#     # Split: all faults -> train; normals split into train/val
#     is_fault = np.array([False]*len(X_norm) + [True]*len(X_fault))
#     idx_norm = np.flatnonzero(~is_fault)
#     idx_fault = np.flatnonzero(is_fault)
#     idx_norm_tr, idx_norm_va = train_test_split(idx_norm, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
#     train_idx = np.concatenate([idx_norm_tr, idx_fault])
#     val_idx = idx_norm_va

#     Xtrain, Xval = X_aug[train_idx], X_aug[val_idx]
#     ytrain, yval = y_aug[train_idx], y_aug[val_idx]
#     wtrain = np.clip(w_aug[train_idx], 1e-6, np.inf)

#     # Params
#     K = int(params.get("num_class", int(np.max(y_aug)) + 1))
#     p = dict(params)
#     p.update(get_params("xgb", y_aug))
#     for k in ("ablation", "lr", "model_type"): p.pop(k, None)
#     p["num_class"] = K
#     p["objective"] = "multi:softprob"
#     p["eval_metric"] = "mlogloss"
#     p.setdefault("tree_method", "hist")
#     p.setdefault("random_state", SEED)
#     p.setdefault("reg_lambda", 1.0)
#     p.setdefault("max_delta_step", 4.0)

#     # Train
#     dtrain = xgb.DMatrix(Xtrain, label=ytrain, weight=wtrain)
#     dval = xgb.DMatrix(Xval, label=yval)
#     t0 = time.time()
#     bst = xgb.train(
#         p, 
#         dtrain, 
#         num_boost_round=num_boosting_rounds, 
#         evals=[(dval, "val")], 
#         verbose_eval=False
#     )
#     print(f"\n\t- new xgb ensemble retrained in {time.time() - t0:.3f} seconds.")


#     # Map applicable actions to the training rows (for fault rows only)
#     app_actions_train = [None] * Xtrain.shape[0]
#     fault_train_positions = np.flatnonzero(is_fault[train_idx])
#     for k, pos in enumerate(fault_train_positions):
#         app_actions_train[pos] = list(map(int, Aa[k]))

#     # Check violations on fault set as in retraining-loss
#     raw = bst.predict(dtrain, output_margin=True).reshape(-1, K)
#     yhat_app = np.fromiter((_argmax_applicable(raw[i], app_actions_train[i]) for i in range(raw.shape[0])),
#                            dtype=int, count=raw.shape[0])
#     ftr = is_fault[train_idx]
#     viol_mask = ftr & (yhat_app == ytrain)
#     num_viol = int(viol_mask.sum())
#     print(f"\t--> after re-train: train-fault violations = {num_viol}")
#     still_unsafe = viol_mask.any()

#     # Boost weights for violators
#     fault_weight_multipliers[viol_mask] *= violator_boost


#     # report quick metrics
#     def _predict_proba(model: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
#         raw = model.predict(dmat, output_margin=True)
#         return _softmax_stable(raw)
#     probs_tr = _predict_proba(bst, dtrain)
#     probs_va = _predict_proba(bst, dval)
#     yhat_tr = np.fromiter(
#         (_argmax_applicable(probs_tr[i], app_actions_train[i]) for i in range(probs_tr.shape[0])),
#         dtype=int, count=probs_tr.shape[0]
#     )
#     yhat_va = probs_va.argmax(axis=1)   # val has only normal rows
#     normal_tr_mask = ~ftr
#     acc_tr_normal = (
#         accuracy_score(ytrain[normal_tr_mask], yhat_tr[normal_tr_mask])
#         if np.any(normal_tr_mask) else float("nan")
#     )
#     acc_va_normal = accuracy_score(yval, yhat_va) if yval.size else float("nan")
#     avoid_tr = (
#         float(np.mean(yhat_tr[ftr] != ytrain[ftr]))
#         if np.any(ftr) else float("nan")
#     )
#     print(
#         "\n\t- retrained ensemble metrics:\n"
#         f"\t * train normal acc: {acc_tr_normal:.3f}, "
#         f"valid normal acc: {acc_va_normal:.3f}\n"
#         f"\t * train fault avoidance (applicable-argmax): {avoid_tr:.3f}"
#     )

#     # Convert to AddTree
#     at = veritas.get_addtree(bst, silent=True)
#     try:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-3, silent=True)
#     except Exception as e:
#         ok = veritas.test_conversion(at, Xval, probs_va, single_rel_tol=1e-2, silent=True)
#     assert ok is True, "Something wrong with conversion to AddTree!"

#     return at, bst, still_unsafe, fault_weight_multipliers