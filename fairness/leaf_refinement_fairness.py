"""
MILP-based leaf-refinement repair for individual fairness.

Fairness is encoded as AGREEMENT between OC pairs that differ only on a protected
attribute: both inputs must receive the same predicted label. Neither input is
privileged, so the target side is a free binary (reduced via connected components).
Always feasible (a constant classifier trivially agrees), hence hard constraints only.

The shared MILP machinery (objective, balanced-accuracy budget, solve, utilities)
lives in leaf_refinement_core.py.
"""
import numpy as np
from scipy.sparse import vstack as sp_vstack
from gurobipy import Model, GRB
import time
from utils import *
from leaf_refinement_core import (
    LEAF_BOUND,
    EPS,
    LAMBDA_L1,
    _define_milp_objective,
    _add_accuracy_constraints,
    _solve_linear_problem,
    build_W,
    _collect_touched_leaves,
    extract_big_M_local,
    connected_components,
    _set_new_leaf_vals_mc,
    _check_modified_weights,
)


def fix_fairness(
    at,
    pairs,
    training_set_info,
    objective: str = "composite",
    accuracy_budget: float = None,
    lambda_l1: float = LAMBDA_L1,
    verbose: bool = False,
):
    """
    Fix counterexample pairs while trying to keep original model performance.

    Args:
    at:                 input AddTree (binary classification).
    pairs:              iterable of (phi1, phi2) sparse-row OC pairs that must agree.
    training_set_info:  dict with at least "OCs_train";
                        if accuracy_budget is set,
                        it must also contain "X_train", "y_train", "min_bacc",
                        "old_tp", "old_tn", "P", "N".
    objective:          "l1" | "composite" | "output" (see below)
    accuracy_budget:    if set, adds a HARD balanced-accuracy budget constraint;
                        otherwise, accuracy preservation is soft in the objective.
    lambda_l1:          weight on the L1 leaf-change penalty.

    Objective options:
    - l1: minimize leaf value changes
    - output: minimize change in raw model output on the training set
              (proxy for accuracy preservation)
    - composite: minimize composite loss on the training set, i.e.,
                    (a) minimize training misclassifications via per-OC slacks
                    (b) L1 regularization

    Returns: (status, at_refined, (W, opt_W))
    """
    # 1) Build W from the model
    W = build_W(at)

    # 2) Build the MILP
    m, delta, touched = _build_fairness_problem(
        at, W, pairs, training_set_info,
        objective=objective,
        accuracy_budget=accuracy_budget,
        lambda_l1=lambda_l1,
        verbose=verbose,
    )
    if m is None:
        print("  [Property] No violations to fix.")
        return "fair", None, None

    # 3) Solve the MILP
    status, opt_W = _solve_linear_problem(m, W, delta, touched, verbose=False)
    if opt_W is None:
        return status, None, None

    # 4) Update leaf values (keeping original base scores)
    base_scores = np.array(
        [at.get_base_score(c) for c in range(at.num_leaf_values())], dtype=np.float64
    )
    at_refined = _set_new_leaf_vals_mc(at, base_scores, opt_W)

    # 5) Log changed weights
    _check_modified_weights(W, opt_W, touched)
    return status, at_refined, (W, opt_W)


def _build_fairness_problem(
    at, W, pairs, training_set_info,
    objective="composite",
    accuracy_budget=None,
    lambda_l1=LAMBDA_L1,
    verbose=False,
):
    """
    (...)
    """
    from contextlib import redirect_stdout
    from io import StringIO

    t0 = time.time()

    # extract unique pairs of counterexamples to fix
    pairs = collect_unique_OCs_pairs(pairs)
    if not pairs:
        return None, None, None
    print(f"  [Counterexamples] {len(pairs)} unique OC pairs to fix.")

    with redirect_stdout(StringIO()):
        m = Model()

    num_classes, num_leaves = W.shape
    if num_classes > 1:
        raise NotImplementedError("Only binary classification is supported.")

    # 1. Distinguish between touched vs frozen leaves:
    #    - touched: leaves activated by at least one pair
    #    - frozen:  all remaining leaves (we won't touch these)
    touched = _collect_touched_leaves(pairs)
    touched_mask = np.zeros(num_leaves, dtype=bool)
    touched_mask[touched] = True

    # 2. Variables to optimize (for touched leaves only):
    #     delta[j] = new_leaf_value[j] - W[0, j]
    lb_delta = -LEAF_BOUND - W[0, touched]
    ub_delta =  LEAF_BOUND - W[0, touched]
    delta = m.addMVar(shape=(len(touched),), lb=lb_delta, ub=ub_delta, name="delta")
    # delta.start = np.zeros(len(touched)) # not useful as this is not a valid solution

    # 3. Define MILP objective
    _define_milp_objective(
        m, delta, touched, W, at, touched_mask, pairs, training_set_info,
        objective=objective, lambda_l1=lambda_l1,
        accuracy_budget=accuracy_budget, verbose=verbose,
    )

    # 4. Add hard property (agreement) constraints 
    __ = _add_property_constraints(
        m, delta, touched, W, pairs, at, verbose=verbose,
    )

    # 5. Optional HARD balanced-accuracy budget constraint
    if accuracy_budget is not None:
        OCs_train = training_set_info["OCs_train"]
        X_train = training_set_info["X_train"]
        y_train = training_set_info["y_train"]
        min_bacc = training_set_info["min_bacc"]
        old_tp   = training_set_info["old_tp"]
        old_tn   = training_set_info["old_tn"]
        P        = training_set_info["P"]
        N        = training_set_info["N"]

        if verbose:
            print(f"  [Hard-BAcc.] Accounting for {len(OCs_train)} unique OCs "
                  f"from the {len(X_train)} training examples.")

        if len(OCs_train) > 0 and accuracy_budget >= 0.0:
            _add_accuracy_constraints(
                m, delta, touched, W, OCs_train, at,
                accuracy_budget, verbose, min_bacc=min_bacc,
                old_tp=old_tp, old_tn=old_tn, P=P, N=N,
            )

    m.update()
    print(f"    - time to set up gurobi: {time.time() - t0:.3f} s.")
    return m, delta, touched


def _add_property_constraints(
    m, delta, touched, W, pairs, at, verbose=False,
):
    """Add fairness constraints: counterexample pairs must receive
    the same predicted label.

    Trick: reduce binary variables via connected-component analysis
    (see connected_components): one binary var per component, not per pair.
      b_view=1 → score >= EPS, b_view=0 → score <= -EPS
    """
    if len(pairs) == 0:
        print("  [Property] No pairs passed.")
        return 0.0

    base_score = float(at.get_base_score(0))
    M = extract_big_M_local(at)

    # 1. Connected components: OCs in the same component must share the same label.
    t_cc0 = time.time()
    oc_phis, oc_to_comp, n_components = connected_components(pairs)
    n_ocs = len(oc_phis)
    if verbose:
        print(f"  [Property/CC] {n_ocs} unique OCs, {n_components} components "
              f"from {len(pairs)} pairs ({time.time() - t_cc0:.3f} s)")

    # 2. Reconstruct each OC score as a function of the optimized leaf deltas.
    Phi_unique = sp_vstack(oc_phis)
    fixed_s_all = np.asarray(Phi_unique @ W[0, :]).flatten() + base_score
    Phi_unique_t = Phi_unique[:, touched]

    # one binary per component; b_view maps each OC to its component's binary
    b_comp = m.addMVar(shape=(n_components,), vtype=GRB.BINARY, name="b_comp")

    b_view = b_comp[oc_to_comp]
    score_expr = Phi_unique_t @ delta + fixed_s_all
    # b_view=1 → score >= EPS, b_view=0 → score <= -EPS
    m.addConstr(score_expr >= EPS - M * (1 - b_view), name="agree_ge")
    m.addConstr(score_expr <= -EPS + M * b_view, name="agree_le")
    print(f"  [Property] Added {n_ocs} hard agreement constraints "
          f"({n_components} components, {len(pairs)} pairs).")
    return 0.0
