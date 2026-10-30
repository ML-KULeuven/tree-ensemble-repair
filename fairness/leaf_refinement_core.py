import numpy as np
from scipy.sparse import vstack as sp_vstack
from gurobipy import GRB
from utils import *

# hardcoded bound for magnitude of optimized leaf values (to avoid overflows)
LEAF_BOUND = 1e3
# hardcoded timeout for gurobi solver
GUROBI_TIMEOUT = 3600 * 1  # 1h
# small safety margins in class comparisons
# (a) fairness EPS
# --> counterexamples score should be at least EPS away from the decision boundary
EPS = 1e-3
# (b) accuracy EPS
# --> correctly-classified OCs must score at least EPS_ACC past the decision boundary
EPS_ACC = 1e-4
# composite objective weight: trade off leaf-change vs training correctness.
LAMBDA_L1 = 1e-3 # old: 0.1      


def _define_milp_objective(
    m, delta, touched, W, at, touched_mask, pairs, training_set_info,
    objective="composite", lambda_l1=LAMBDA_L1, accuracy_budget=None, 
    verbose=False,
):
    obj = objective
    num_leaves = W.shape[1]

    # L1 regularizer: total change in leaf values
    abs_diff = m.addMVar(shape=(len(touched),), lb=0.0, name="abs_diff")
    m.addConstr(abs_diff >= delta, name="abs_pos")
    m.addConstr(abs_diff >= -delta, name="abs_neg")
    l1_term = abs_diff.sum()

    if obj == "l1":
        # exclusively minimize leaf-value changes
        m.setObjective(l1_term, GRB.MINIMIZE)

        if verbose:
            print(f"  [L1 obj] Minimizing leaf-value changes "
                  f"({len(touched)} touched leaves out of {num_leaves}).")
        return

    if accuracy_budget is not None:
        # if we set accuracy as hard constraints, we're done here
        m.setObjective(l1_term, GRB.MINIMIZE)
        return

    # composite/output obj: explicitly model training accuracy in objective

    # extract all training set (OC, label) pairs
    OCs_train = training_set_info.get("OCs_train")
    num_OCs = len(OCs_train)
    all_items = list(OCs_train.values())
    counts_all = np.array([g["count"] for g in all_items], dtype=np.float64)
    labels_all = np.array([g["label"] for g in all_items], dtype=int)

    # reconstruct model output for every training OC
    base_score = float(at.get_base_score(0))
    Phi = sp_vstack([g["phi"] for g in all_items])
    old_scores_all = np.asarray(Phi @ W[0, :]).flatten() + base_score

    # filter out training OCs that we do not need to model explicitly;
    # i.e., those that are (1) misclassified or (2) don't activate any touched leaf
    guard_indices = _get_guard_indices(
        OCs_train, W, Phi, old_scores_all, pairs, touched_mask, at, verbose=verbose
    )
    counts_guard = counts_all[guard_indices]
    fixed_scores_guard = old_scores_all[guard_indices]
    guard_pos = np.where(labels_all[guard_indices] == 1)[0]
    guard_neg = np.where(labels_all[guard_indices] == 0)[0]

    if verbose:
        print(f"  [Acc-aware obj] Guarding {len(guard_indices)}/{num_OCs} training OC groups "
              f"({len(guard_pos)} pos, {len(guard_neg)} neg)")

    Phi_guard_t = Phi[guard_indices][:, touched]
    scores_OC = m.addMVar(shape=(len(guard_indices),), lb=-GRB.INFINITY, name="scores_OC")
    m.addConstr(scores_OC == Phi_guard_t @ delta + fixed_scores_guard, name="scores_OC_def")

    # extract per-class weights (we minimize bacc. loss)
    P_full = counts_all[labels_all == 1].sum()   # tot positives in training set
    N_full = counts_all[labels_all == 0].sum()   # tot negatives in training set
    weights_guard = counts_guard.copy()
    scale = 0.5 * (P_full + N_full)
    weights_guard[guard_pos] *= scale / P_full
    weights_guard[guard_neg] *= scale / N_full

    # (a) minimize training misclassifications via per-OC slacks or exact model output
    # (b) L1 regularization
    if obj == "composite":
        # per-OC slack variables
        slacks = m.addMVar(shape=(len(guard_indices),), lb=0.0, name="slacks")
        if len(guard_pos) > 0:
            m.addConstr(slacks[guard_pos] >= EPS_ACC - scores_OC[guard_pos], name="slacks_pos")
        if len(guard_neg) > 0:
            m.addConstr(slacks[guard_neg] >= EPS_ACC + scores_OC[guard_neg], name="slacks_neg")
        acc_term = weights_guard @ slacks

    elif obj == "output":
        # minimize change in raw model output on the training set
        # (weaker proxy for accuracy preservation; not used in NeurIPS paper)
        score_delta = m.addMVar(shape=(len(guard_indices),), lb=0.0, name="score_delta")
        m.addConstr(score_delta >= scores_OC - fixed_scores_guard, name="delta_pos")
        m.addConstr(score_delta >= fixed_scores_guard - scores_OC, name="delta_neg")
        acc_term = weights_guard @ score_delta

    else:
        raise ValueError(f"Unknown objective: {obj}")

    w_mass = float(weights_guard.sum())
    m.setObjective(
        lambda_l1 * l1_term / len(touched) + acc_term / w_mass ,
        GRB.MINIMIZE
    )


# def get_training_correct_mask(OCs_train, W, at):
#     """Boolean mask over the training OCs:
#     we extract which ones are correctly classified by the model with leaf values W
#     (we call it using the repair loop's input model, i.e. the LOP-compressed model)."""
#     items = list(OCs_train.values())
#     scores = (np.asarray(sp_vstack([g["phi"] for g in items]) @ W[0, :]).flatten()
#               + float(at.get_base_score(0))) # original scores of each training OC
#     labels = np.array([g["label"] for g in items])
#     return np.where(labels == 1, scores >= 0.0, scores < 0.0)


def _get_guard_indices(
        OCs_train, W, Phi, old_scores, pairs, touched_mask, at, verbose=False):
    """
    Return the subset of (OC, label) pairs indices for which we want to retain
    the original model prediction through accuracy-preservation measures
    (soft constraints in objective via slack vars, or hard constraints via binary vars).

    Applies two filters in sequence:
      1. Correctly-classified only  — misclassified OCs shouldn't be kept fixed
      2. Touches at least one touched leaf — frozen OCs can't be affected
    """

    items = list(OCs_train.values())
    num_OCs = len(items)
    labels = np.array([g["label"] for g in items], dtype=int)

    # filter 1: only consider OCs that are correctly classified
    # --> if an OC is misclassified, we don't want that prediction to stay fixed
    all_pos = np.where(labels == 1)[0]
    all_neg = np.where(labels == 0)[0]
    correct_pos = all_pos[old_scores[all_pos] >= 0.0]
    correct_neg = all_neg[old_scores[all_neg] < 0.0]
    guard_indices = np.concatenate([correct_pos, correct_neg])
    n_misclassified = num_OCs - len(guard_indices)
    # edit below: we directly pass the training_correct_mask (correctly classified OCs) from main.py
    # guard_indices = np.where(training_correct_mask)[0]

    # filter 2: only track OCs that reach at least one touched leaf
    #           --> OCs that only activate frozen leaves can't be affected
    at_risk_mask = np.array(Phi[guard_indices] @ touched_mask.astype(float)).flatten() > 0
    n_frozen = int(np.sum(~at_risk_mask))
    guard_indices = guard_indices[at_risk_mask]

    if verbose:
        print(f"  [Training OCs] Discarding {n_misclassified} misclassified OCs")
        print(f"  [Training OCs] Discarding {n_frozen} OCs not touching any unfair leaf")

    return guard_indices


# @TODO unify with fairness constraints: either explicit big-M, or GenConstrIndicators
def _add_accuracy_constraints(
    m,
    delta,
    touched,
    W,
    OCs_train,
    at,
    max_acc_drop,
    verbose=False,
    min_bacc=None,          # balanced-accuracy budget: bound on TP/P + TN/N
    old_tp=None,
    old_tn=None,
    P=None,
    N=None,
):
    items = list(OCs_train.values())
    counts = np.array([g["count"] for g in items], dtype=np.float64)
    labels = np.array([g["label"] for g in items], dtype=int)
    base_score = float(at.get_base_score(0))

    if verbose:
        old_bacc = 0.5 * (old_tp / P + old_tn / N)
        print(
            f"  [Hard-BAcc.]:  max_bacc_drop={max_acc_drop},  old_bacc={old_bacc:.4f},  min_bacc={min_bacc:.4f}"
        )

    # Score expression: score_new = Phi_t @ delta + old_scores
    # old_scores is the full original score (baseline for the delta change)
    Phi = sp_vstack([g["phi"] for g in items])
    old_scores = np.asarray(Phi @ W[0, :]).flatten() + base_score

    # OCs reaching a touched leaf can change prediction; all others are frozen
    touched_mask = np.zeros(W.shape[1], dtype=bool)
    touched_mask[touched] = True
    var = np.asarray(Phi @ touched_mask.astype(float)).ravel() > 0
    frozen = ~var

    # frozen OCs: prediction is fixed -> constant contribution to TP / TN
    tp_frozen = counts[frozen & (labels == 1) & (old_scores >= 0.0)].sum()
    tn_frozen = counts[frozen & (labels == 0) & (old_scores < 0.0)].sum()

    # variable OCs: binary z_g = 1 iff OC-group g is correctly classified
    var_idx = np.where(var)[0]
    z = m.addMVar(shape=(len(var_idx),), vtype=GRB.BINARY, name="z_correct")
    v_pos = np.where(labels[var_idx] == 1)[0]
    v_neg = np.where(labels[var_idx] == 0)[0]
    # bacc = 0.5*(TP/P + TN/N) >= min_bacc  <==>  TP/P + TN/N >= 2*min_bacc
    tp_expr = tp_frozen + counts[var_idx][v_pos] @ z[v_pos]
    tn_expr = tn_frozen + counts[var_idx][v_neg] @ z[v_neg]
    m.addConstr(tp_expr / P + tn_expr / N >= 2.0 * float(min_bacc), name="min_bacc")

    scores_OC = m.addMVar(shape=(len(var_idx),), lb=-GRB.INFINITY, name="scores_OC")
    m.addConstr(scores_OC == Phi[var_idx][:, touched] @ delta + old_scores[var_idx], name="scores_OC_def")

    # z=1 iff OC correctly classified (bi-directional indicator constraints)
    for k in v_pos.tolist():
        # z=1 → score >= +EPS_ACC (confidently positive)
        m.addGenConstrIndicator(z[k], True, scores_OC[k] >= EPS_ACC, name=f"acc_pos_if_z1[{k}]")
        # z=0 → score <= 0
        m.addGenConstrIndicator(z[k], False, scores_OC[k] <= 0.0, name=f"acc_pos_if_z0[{k}]")
    for k in v_neg.tolist():
        # z=1 → score <= -EPS_ACC (confidently negative)
        m.addGenConstrIndicator(z[k], True, scores_OC[k] <= -EPS_ACC, name=f"acc_neg_if_z1[{k}]")
        # z=0 → score >= 0
        m.addGenConstrIndicator(z[k], False, scores_OC[k] >= 0.0, name=f"acc_neg_if_z0[{k}]")

    return


def _solve_linear_problem(m, W, delta, touched, verbose=False):
    m.setParam('OutputFlag', 0)                 # silent mode
    m.setParam('TimeLimit', GUROBI_TIMEOUT)     # set max time for gurobi
    m.setParam('Seed', SEED)
    # Keep feasibility tolerances comfortably smaller than EPS
    m.setParam('FeasibilityTol', 1e-9)
    m.setParam("OptimalityTol", 1e-9)
    m.setParam("IntFeasTol", 1e-9)
    m.setParam('NumericFocus', 2)
    # let Gurobi decide whether we want cuts
    m.setParam('Cuts', -1)

    # Hard constraints must hold; allow early stop on objective only
    m.Params.MIPFocus = 1            # focus on finding feasible solutions quickly
    m.Params.MIPGap = 0.50           # don't need the absolute optimum
    m.setParam('Presolve', 2)        # aggressive presolve
    m.setParam('Heuristics', 0.5)    # aggressive heuristics
    m.setParam('Threads', 4)         # use more threads if available

    # debug: stop when first solution found
    # m.setParam('SolutionLimit', 1)
    # m.setParam('IndicatorConstraintTol', 1e-5)

    m.optimize()
    
    # init. status return string
    status = "unknown"

    # Check the optimization status and output
    if m.status == GRB.OPTIMAL:
        print(f"    - optimization done --> obj: {m.objVal:.3f}, time: {m.Runtime:.3f} s")
        status = "optimal"
    elif m.status == GRB.INFEASIBLE:
        print(f"    - problem INFEASIBLE! (runtime: {m.Runtime:.3f} s)")
        return "infeasible", None
    elif m.status in [GRB.TIME_LIMIT, GRB.WORK_LIMIT, GRB.INTERRUPTED]:
        print("    - gurobi TIMED OUT!")
        if m.SolCount > 0:
            print(f"    - suboptimal solution found (obj: {m.objVal:.3f})")
            status = "suboptimal"
        else:
            return "timeout", None
    else:
        print(f"    - optimization ended with status {m.status}.")
        raise RuntimeError(f"Unexpected Gurobi status: {m.status}")

    # Reconstruct full leaf value matrix: start from original W, apply delta to touched leaves
    optimized_W = W.copy()
    optimized_W[0, touched] = W[0, touched] + delta.X

    return status, optimized_W


# ----------------------------- model/feature utilities ------------------------


def build_W(at):
    """
    Return W with shape [num_classes, total_leaves]
    --> each row has all the leaf values of the ensemble for a specific class
    --> so one column for each leaf in the ensemble
    """
    num_leaf_values = at.num_leaf_values()
    total_cols = sum(len(t.get_leaf_ids()) for t in at)
    W = np.zeros((num_leaf_values, total_cols), dtype=np.float64)

    offset = 0
    for t in at:
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        for cls in range(num_leaf_values):
            for cnt, leaf_id in enumerate(leaf_ids):
                W[cls, offset + cnt] = t.get_leaf_value(leaf_id, cls)
        offset += num_leaves
    return W


def _collect_touched_leaves(pairs):
    touched = set()
    for (phi0, phi1) in pairs:
        touched.update(phi0.indices)
        touched.update(phi1.indices)
    return np.array(sorted(touched), dtype=int)


def extract_big_M_local(at):
    """Heuristic big-M based on current leaf values.

    For binary (single-score) models, the largest absolute leaf value in any tree
    is a conservative per-tree bound; summing per-tree maxima and adding |base score|
    yields a global bound on |score| for the current leaf values.
    """
    per_tree_max = []
    for tree in at:
        leaf_ids = tree.get_leaf_ids()
        # binary in this file: num_leaf_values()==1
        local_max = 0.0
        for leaf_id in leaf_ids:
            v = float(tree.get_leaf_value(leaf_id, 0))
            local_max = max(local_max, abs(v))
        per_tree_max.append(local_max)
    return float(np.sum(per_tree_max)) + abs(float(at.get_base_score(0)))


def _set_new_leaf_vals_mc(at, base_scores, new_leaf_vals):
    for class_index, base_score in enumerate(base_scores):
        at.set_base_score(class_index, base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        for class_index in range(len(base_scores)):
            for count, leaf_id in enumerate(leaf_ids):
                t.set_leaf_value(leaf_id, class_index, new_leaf_vals[class_index, offset + count])
        offset += num_leaves
    return at


def _check_modified_weights(W, optimized_weights, touched, fl_only=False):
    changed_weights = np.sum(np.abs((W - optimized_weights).ravel()) > 1e-8)
    if np.any(np.abs(optimized_weights) > LEAF_BOUND + 1e-1):
        print("     - WARNING: some weights have been set to a very large value!")
    changed_columns = np.any(np.abs(W - optimized_weights) > 1e-9, axis=0)
    changed_column_indices = np.where(changed_columns)[0]
    changed_touched_columns = np.sum(np.isin(changed_column_indices, touched))

    print(f"         - tot. weights changed: {changed_weights}",
          f"({np.round(changed_weights / W.size * 100, 2)}%)",
          f"\n     - tot. leaves changed: {len(changed_column_indices)}",
          f"({np.round(len(changed_column_indices) / W.shape[1] * 100, 2)}%)",
          f"\n     - of which {changed_touched_columns} are 'touched' leaves")
    if fl_only and len(changed_column_indices) != changed_touched_columns:
        print("       - WARNING: modified some non-touched leaves due to small numerical issues!")
    # return {
    #     "n_modified": len(changed_column_indices),
    #     "n_leaves":   W.shape[1],
    # }


def connected_components(pairs):
    """Connected-component reduction over OC pairs.

    Trick: Reduce binary variables via connected-component analysis.
    --> Each unique OC is a node; each pair is an edge (same class required).
        All OCs in the same component must share the same binary label (pos/neg).
        This replaces #unique_pairs binary vars with #components binary vars,
        where #components <= #pairs.

    e.g., if pairs (A, B) and (B, C) are both present,
          then we can use on single binary for A, B, C,
          as they must all be on the same side of the decision boundary.

    Returns:
        oc_phis:      list of unique OC phi rows (oc_phis[id] = phi)
        oc_to_comp:   int array mapping each OC id to its component id
        n_components: number of components
    """
    # 1. Collect unique OCs and assign to each an integer ID
    oc_key_to_id = {}  # bytes key -> int id
    oc_phis = []       # oc_phis[id] = phi row (sparse)

    def get_oc_id(phi):
        key = phi.indices.tobytes()
        oc_id = oc_key_to_id.get(key)
        if oc_id is None:
            oc_id = len(oc_phis)
            oc_key_to_id[key] = oc_id
            oc_phis.append(phi)
        return oc_id

    edges = []
    for phi1, phi2 in pairs:
        a = get_oc_id(phi1)
        b = get_oc_id(phi2)
        if a != b:
            edges.append((a, b))

    n_ocs = len(oc_phis)

    # 2. Find maximal groups of OCs that must share the same label.
    #    group[i] starts as i (each OC in its own group).
    #    After processing all edges, group[i] holds the group ID for OC i:
    #    two OCs with the same group ID must be assigned the same binary label.
    group = list(range(n_ocs))

    def find_group(x):
        # follow pointers until we reach an OC pointing to itself: that's the group ID
        while group[x] != x:
            x = group[x]
        return x

    for a, b in edges:
        # merge: make the group of b point to the group of a
        ga, gb = find_group(a), find_group(b)
        if ga != gb:
            group[gb] = ga

    # assign a compact index to each group
    roots = sorted(set(find_group(i) for i in range(n_ocs)))
    root_to_comp = {r: i for i, r in enumerate(roots)}
    oc_to_comp = np.array([root_to_comp[find_group(i)] for i in range(n_ocs)], dtype=int)

    return oc_phis, oc_to_comp, len(roots)
