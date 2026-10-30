import numpy as np
from scipy.sparse import csr_matrix, vstack as sp_vstack
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from utils.utils import SEED
import time
import veritas

# hardcoded bound for magnitude of optimized leaf values (to avoid overflows)
LEAF_BOUND = 1e3
# hardcoded timeout for gurobi solver
GUROBI_TIMEOUT = 3600 * 3  # 3h
# small safety margin in class comparisons (fault constraints)
EPS = 1e-3
# small safety margin for score-based accuracy constraints
# --> correctly-classified OCs must keep a margin of at least EPS_ACC
#     over the best competing class
EPS_ACC = 1e-4
# objective weight: trade off (mean) leaf-change vs training correctness.
# Both objective terms are normalized (see _build_linear_problem), so this
# is a scale-free ratio, comparable across benchmarks / sample counts.
LAMBDA_L1 = 1e-3

def fix_milp(
    at,
    faults_info,
    training_set_info,
    objective: str = "hinge",
    verbose=False,
    lambda_slack = LAMBDA_L1,
    eps: float = EPS,      # safety margin in the fault constraints (use mr_config['gur_tol'])
):
    """
    Fix faults (HARD constraint) while trying to keep original model performance.

    Objective options:
    - hinge: minimize hinge loss on the training set

    Accuracy-related constraints options:
    - None: ignore explicit accuracy preservation
    - Hard: hard constraint on number of misclassified training samples
            (via binary indicators)

    Inputs:
        (...)

    Returns: (status, at_refined/None, (W, opt_W)/None, m, timings)
      timings: dict with the wall-clock seconds spent in
        "build"       -- W matrix + MILP construction
        "solve"       -- gurobi optimize + leaf-value update
        "diagnostics" -- post-solve sanity checks / reporting (not repair work)
    """
    timings = {"build": 0.0, "solve": 0.0, "diagnostics": 0.0}

    # 1) Build W from the model
    t0 = time.time()
    W = build_W(at)

    # 2) Build the MILP
    m, delta, touched = _build_linear_problem(
        at, W, faults_info,
        training_set_info,
        objective,
        verbose,
        lambda_slack,
        eps,
    )
    timings["build"] = time.time() - t0

    # 3) Solve MILP
    t0 = time.time()
    status, opt_W = _solve_linear_problem(m, W, delta, touched)
    if opt_W is None:
        timings["solve"] = time.time() - t0
        return status, None, None, m, timings

    # 4) Update leaf values (keeping original base scores)
    base_scores = np.array([at.get_base_score(c) for c in range(at.num_leaf_values())], dtype=np.float64)
    at_refined = at.copy()
    at_refined = _set_new_leaf_vals_mc(at_refined, base_scores, opt_W)
    timings["solve"] = time.time() - t0

    # 5) Diagnostics (timed separately; booked as "other" runtime by the caller)
    #    a. changed-weight statistics
    #    b. exact re-check of every fault on the refined model: the MILP only
    #       sees the constraints up to solver tolerances; if a fault survives,
    #       print what is needed to tell a numerical slip from a modelling error
    #    c. (verbose) which guarded training OCs lost their prediction
    t0 = time.time()
    _check_modified_weights(W, opt_W, touched)
    report_fault_violations(m, at_refined, faults_info, eps)
    if verbose:
        _report_lost_training_ocs(m)
    timings["diagnostics"] = time.time() - t0

    return status, at_refined, (W, opt_W), m, timings


def _get_guard_indices(OCs_train, Phi, old_scores, touched_mask, verbose=False):
    """
    Return the subset of OC indices that should be guarded in the objective.
    Applies two filters in sequence:
      1. Correctly-classified only  — misclassified OCs shouldn't be kept fixed
      2. Touches at least one touched leaf — frozen OCs can't be affected
    """
    items = list(OCs_train.values())
    num_OCs = len(items)
    labels = np.array([g["label"] for g in items], dtype=int)

    # filter 1: correctly classified = argmax of scores matches the true action/label
    # old_scores shape: (num_OCs, num_classes)
    predicted = np.argmax(old_scores, axis=1)  # shape (num_OCs,) # We assume that all actions are applicable for this unfortunately. Otherwise rather than np.argmax we would first maask out inapplicable actions.
    correctly_classified = np.where(predicted == labels)[0]
    n_misclassified = num_OCs - len(correctly_classified)
    guard_indices = correctly_classified

    # filter 2: only track OCs that reach at least one touched leaf
    # touched_mask shape: (num_classes, num_leaves) → flatten to (num_classes * num_leaves,)
    # Phi shape: (num_OCs, num_leaves) — need to think about how Phi interacts with touched_mask
    touched_any_leaf = touched_mask.any(axis=0)  # (num_leaves,) — leaf touched by ANY class
    at_risk_mask = np.array(Phi[guard_indices] @ touched_any_leaf.astype(float)).flatten() > 0
    n_frozen = int(np.sum(~at_risk_mask))
    guard_indices = guard_indices[at_risk_mask]

    if verbose:
        print(f"  [Training OCs] Discarding {n_misclassified} misclassified OCs")
        print(f"  [Training OCs] Discarding {n_frozen} OCs not touching any unfair leaf")

    return guard_indices


def _build_linear_problem(
    at,
    W,      
    faults_info,
    training_set_info,
    objective,
    verbose=False,
    lambda_slack = LAMBDA_L1,
    eps = EPS,
):
    """
    (...)
    """
    from contextlib import redirect_stdout
    from io import StringIO

    t0 = time.time()

    with redirect_stdout(StringIO()):
        m = Model()

    num_classes, num_leaves = W.shape

    faults = faults_info.get("faults", [])
    faulty_actions = faults_info.get("faulty_actions", [])
    applicable_actions = faults_info.get("applicable_actions", [])


    # 1. Distinguish between touched vs frozen leaves:
    #    - touched: leaves activated by at least one unfair pair 
    #    - frozen:  all remaining leaves (we won't touch these)
    faulty_leaves = _collect_faulty_leaves(at, faults, applicable_actions)
    touched_leaf_indices = faulty_leaves[:, 0]
    touched_classes      = faulty_leaves[:, 1]

    # Build a 2D mask (num_classes x num_leaves)
    touched_mask = np.zeros((num_classes, num_leaves), dtype=bool)
    touched_mask[touched_classes, touched_leaf_indices] = True

    # Bounds now respect per-class, per-leaf combinations
    lb_delta = np.zeros((num_classes, num_leaves))
    ub_delta = np.zeros((num_classes, num_leaves))
    lb_delta[touched_mask] = -LEAF_BOUND - W[touched_mask]
    ub_delta[touched_mask] =  LEAF_BOUND - W[touched_mask]

    # delta = m.addMVar(shape=(len(touched),), lb=-LEAF_BOUND, ub=LEAF_BOUND, name="delta")
    delta = m.addMVar(shape=touched_mask.shape, lb=lb_delta, ub=ub_delta, name="delta")

    # 3. Define objective
    assert objective == "hinge", "Only 'hinge' objective currently implemented."

    # Minimize misclassification on the training set + regularize with L1 loss
    abs_diff = m.addMVar(shape=(num_classes,num_leaves), lb=0.0, name="abs_diff")
    m.addConstr(abs_diff >= delta, name="abs_pos")
    m.addConstr(abs_diff >= -delta, name="abs_neg")

    # extract OCs from training set
    OCs_train = collect_unique_OCs(training_set_info.get("training_states"), training_set_info.get("training_actions"), at)
    all_items = list(OCs_train.values())
    counts_all = np.array([g["count"] for g in all_items], dtype=np.float64)
    labels_all = np.array([g["label"] for g in all_items], dtype=int)

    # extract base score and Phi matrix
    base_scores = [float(at.get_base_score(c)) for c in range(num_classes)]
    Phi = sp_vstack([g["phi"] for g in all_items])
    num_OCs = len(OCs_train)

    # define score threshold based on model type
    score_threshold = 0.0 if at.get_type() == veritas.AddTreeType.CLF_SOFTMAX else 0.5
    thr_pos = float(score_threshold) + float(EPS_ACC)

    # compute current scores from original weights to identify correctly-classified OCs
    old_scores_all = np.asarray(Phi @ W.T) + base_scores


    # filter out training OCs that are:
    #   1. misclassified [NOPE]
    #        --> these are already misclassified, so no point in trying to preserve their score (instead ignore them in the objective)
    #   2. not touching any unfair leaf
    #       --> no point in modelling them in the optimization procedure (instead preserve their original value)
    guard_indices = _get_guard_indices(OCs_train, Phi, old_scores_all, touched_mask, verbose=verbose)

    counts_guard = counts_all[guard_indices]
    fixed_scores_guard = old_scores_all[guard_indices] # restricted to modifable OCs only
    guard_labels = labels_all[guard_indices]

    # per-class weights: we minimize a balanced-accuracy loss, i.e. every class
    # contributes equally regardless of how many training samples it has.
    # (multiclass generalization of the binary scale/P, scale/N reweighting)
    class_totals = np.bincount(labels_all, weights=counts_all, minlength=num_classes)
    scale = counts_all.sum() / num_classes
    weights_guard = counts_guard.copy()
    for c in range(num_classes):
        if class_totals[c] > 0:
            weights_guard[guard_labels == c] *= scale / class_totals[c]


    if verbose:
        class_counts = {c: int(np.sum(guard_labels == c)) for c in range(num_classes)}
        print(f"  [Hinge obj] Guarding {len(guard_indices)}/{num_OCs} training OC groups "
            f"(per class: {class_counts})")
        
    # compute scores for selected OCs
    # print("Guarded phi:", Phi[guard_indices])
    # print(touched_leaf_indices)
    unique_touched_leaves = np.unique(faulty_leaves[:, 0])

    # Create a mask for columns to keep
    col_mask = np.zeros(num_leaves, dtype=bool)
    col_mask[unique_touched_leaves] = True

    # Zero out non-touched columns, preserving shape (num_guard, total_leaves)
    # (elementwise multiply by a column mask: no CSR structure edits, no
    #  SparseEfficiencyWarning, and much faster than slice-assignment)
    Phi_guard_t = Phi[guard_indices].multiply(col_mask.astype(np.float64)[None, :]).tocsr()

    scores_OC = m.addMVar(shape=(len(guard_indices),num_classes), lb=-GRB.INFINITY, name="scores_OC")
    m.addConstr(scores_OC == Phi_guard_t @ delta.T + fixed_scores_guard, name="scores_OC_def")

    slacks = m.addMVar(shape=(len(guard_indices),), lb=0.0, ub=2*LEAF_BOUND, name="slacks")
    predicted_guard = np.argmax(fixed_scores_guard, axis=1)  # shape (num_guard,)
    
    scores_OC_list = scores_OC.tolist()  # nested list [num_guard][num_classes] of Var

    max_other = m.addMVar(shape=(len(guard_indices),), lb=-GRB.INFINITY, name="max_other_scores")

    for i, pred_c in enumerate(predicted_guard):
        competing_vars = [scores_OC_list[i][c] for c in range(num_classes) if c != pred_c] # This is stricter than it needs to be since for the training data we never stored the set of applicable actions. They are required to further simplify the constraint. However, we can be guaranteed that the action in the training data is applicable.
        m.addGenConstrMax(max_other[i].item(), competing_vars, name=f"max_other_{i}")

    m.addConstr(
        slacks >= thr_pos - (scores_OC[np.arange(len(guard_indices)), predicted_guard] - max_other),
        name="hinge_multiclass"
    )

    # Objective: lambda * (mean leaf change over touched leaves)
    #          + (class-balanced misclassification rate on guarded training OCs)
    # Both terms are normalized so lambda has the same meaning regardless of
    # the number of touched leaves / training samples.
    n_touched = int(touched_mask.sum())
    l1_term = abs_diff.sum() / n_touched
    w_mass = float(weights_guard.sum())
    if len(guard_indices) > 0 and w_mass > 0:
        acc_term = (weights_guard @ slacks) / w_mass
        m.setObjective(lambda_slack * l1_term + acc_term, GRB.MINIMIZE)
    else:
        m.setObjective(l1_term, GRB.MINIMIZE)
    m._counts_guard = counts_guard  # stash for post-solve diagnostics

    avoid_faults_constraints(m, delta, W, faults, faulty_actions, at, applicable_actions, eps=eps)
    m.update()

    print(f"    - time to set up gurobi: {time.time() - t0:.3f} s.")

    return m, delta, touched_mask



def get_ocs_batch(X, at):
    from veritas import FloatT

    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=FloatT)
    else:
        X = np.asarray(X, dtype=FloatT)
    num_examples = X.shape[0]
    num_trees = len(at)

    offsets, leaf_maps = [], []
    total_leaves = 0
    for t in at:
        offsets.append(total_leaves)
        leaf_ids = t.get_leaf_ids()
        leaf_maps.append({leaf_id: i for i, leaf_id in enumerate(leaf_ids)})
        total_leaves += len(leaf_ids)

    indices = np.zeros(num_examples * num_trees, dtype=np.int32)
    for i in range(num_examples):
        x_row = X[i, :]
        for tree_idx, t in enumerate(at):
            leaf_id = t.eval_node(x_row)[0]
            indices[i * num_trees + tree_idx] = offsets[tree_idx] + leaf_maps[tree_idx][leaf_id]

    indptr = np.arange(0, (num_examples + 1) * num_trees, num_trees, dtype=np.int32)
    data = np.ones(num_examples * num_trees, dtype=FloatT)
    return csr_matrix((data, indices, indptr), shape=(num_examples, total_leaves))


def group_faults_by_oc(at, xforb, yforb, applicable_actions=None):
    """
    Groups faults by their OCs.
    Returns a dictionary where keys are OCs and values are lists of tuples
    (index, state, forbidden_class, applicable_actions).
    """
    from collections import defaultdict
    # print("In group faults by oc")
    grouped_faults = defaultdict(list)
    for idx, (x, y) in enumerate(zip(xforb.itertuples(index=False), yforb)):
        oc_x = tuple(get_oc(x, at).toarray()[0])
        actions = applicable_actions[idx] if applicable_actions else None
        grouped_faults[oc_x].append((idx, x, y, actions))
        # print(xforb, y, actions, idx, oc_x)

    # sanity check: all faults for each OC must have the same phi
    for idx, (__, fault_specs) in enumerate(grouped_faults.items()):
        avoid_condition_phi = get_oc(fault_specs[0][1], at)
    
        if len(fault_specs) > 1:
            for i in range(1, len(fault_specs)):
                assert np.array_equal(get_oc(fault_specs[i][1], at).toarray(), 
                                    avoid_condition_phi.toarray()), \
                    f"Fault OC no. {idx} has non unique phi???"
    return grouped_faults


def avoid_faults_constraints(m, delta, W, fault_states, faulty_actions, at,
                                 applicable_actions, eps=1e-4):
        
        num_classes = at.num_leaf_values()
        # extract base score and Phi matrix
        base_scores = [float(at.get_base_score(c)) for c in range(num_classes)]

        # group faults by their OCs
        compact_faults = group_faults_by_oc(at, fault_states, faulty_actions, applicable_actions)
        # manually set big-M for constraints to smallest possible value 
        # --> should help reducing numerical instabilities
        M = extract_big_M_local(at, eps)
        # extract allowed/forbidden classes per each OC and add constraints 
        for idx, (oc, fault_specs) in enumerate(compact_faults.items()):
            # print(oc)
            avoid_condition_phi = get_oc(fault_specs[0][1], at)
            all_forbidden = set()
            all_allowed = set(range(at.num_leaf_values()))
            for __, __, y_forbidden, aas in fault_specs:
                all_forbidden.add(y_forbidden)
                if aas is not None:
                    all_allowed &= set(aas)
            all_allowed -= all_forbidden
            # all_forbidden = set(range(at.num_leaf_values())) - all_allowed
            assert all_allowed, f"No allowed classes for fault no. {idx}??"
            # print(avoid_condition_phi @ W.T + base_scores)
            old_scores = avoid_condition_phi @ W.T + base_scores
            # print("Index:", idx, " faulty actions: ", all_forbidden, " allowed actions: ", all_allowed)
            # print("Avoid Condition:", avoid_condition_phi)
            # print("Old Scores", old_scores_all)
            # precompute all scores once
            old_scores = avoid_condition_phi @ W.T + base_scores
            scores = avoid_condition_phi @ delta.T + old_scores
            # encode that at least one allowed class has to beat all forbidden classes
            b_vars = []
            for c in all_allowed:
                b = m.addVar(vtype=GRB.BINARY, name=f"b_allow_{idx}_{c}")
                b_vars.append(b)
                for f in all_forbidden:
                    # print(old_scores_all)
                    m.addConstr(scores[0,c] >= scores[0,f] + eps - M * (1 - b),
                                name=f"force_{c}_gt_{f}_{idx}")
            m.addConstr(quicksum(b_vars) >= 1, name=f"at_least_one_valid_gt_forbidden_{idx}")


def report_fault_violations(m, at_refined, faults_info, eps, tol=1e-8):
    """
    Evaluate each fault OC on the refined AddTree and report those where no
    allowed action beats every forbidden action by eps, together with the
    values of the OC's b_allow_* binaries (a b close to but below 1 means the
    big-M constraint was satisfied only through IntFeasTol slack).

    `tol` absorbs floating-point noise: the solver places active margins
    exactly at eps (up to ~1e-12), which must not be reported as violations.
    """
    faults = faults_info["faults"]
    faulty_actions = faults_info["faulty_actions"]
    applicable_actions = faults_info.get("applicable_actions")
    compact_faults = group_faults_by_oc(at_refined, faults, faulty_actions, applicable_actions)
    b_vals = {v.VarName: v.X for v in m.getVars() if v.VarName.startswith("b_allow_")}

    n_viol = 0
    for idx, (__, fault_specs) in enumerate(compact_faults.items()):
        x = fault_specs[0][1]
        all_forbidden = set()
        all_allowed = set(range(at_refined.num_leaf_values()))
        for __, __, y_forbidden, aas in fault_specs:
            all_forbidden.add(y_forbidden)
            if aas is not None:
                all_allowed &= set(aas)
        all_allowed -= all_forbidden
        scores = np.asarray(at_refined.eval(np.asarray([list(x)], dtype=np.float64)))[0]
        margin = max(scores[c] for c in all_allowed) - max(scores[f] for f in all_forbidden)
        if margin < eps - tol:
            n_viol += 1
            bs = {c: round(b_vals.get(f"b_allow_{idx}_{c}", float("nan")), 6) for c in sorted(all_allowed)}
            print(f"    (!) fault OC {idx} still violated: margin={margin:.4f} < eps={eps}, "
                  f"allowed={sorted(all_allowed)}, forbidden={sorted(all_forbidden)}, b={bs}")
    if n_viol:
        print(f"    (!) {n_viol}/{len(compact_faults)} fault OCs violated after solve "
              f"(IntFeasTol={m.Params.IntFeasTol}, status={m.status})")
    return n_viol


def collect_unique_OCs(training_data, y_train, policy):
    """
    Collect the unique OCs that the training samples route to.
    """
    OCs_train = get_ocs_batch(training_data, policy)
    y_arr = np.asarray(y_train).reshape(-1)
    indptr = OCs_train.indptr
    indices = OCs_train.indices

    unique_OCs: dict[tuple[bytes, int], dict] = {}
    for i in range(OCs_train.shape[0]):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        oc_key = indices[start:end].tobytes()  # immutable + fast to hash
        yi = int(y_arr[i])
        key = (oc_key, yi)

        entry = unique_OCs.get(key)
        if entry is None:
            # Only build the CSR row once per unique OC
            unique_OCs[key] = {"phi": OCs_train[i, :], "count": 1, "label": yi}
        else:
            entry["count"] += 1

    print("Unique OCs in training set:", len(unique_OCs))
    return unique_OCs

def _report_lost_training_ocs(m):
    """Diagnostics: summarize guarded training OC groups whose hinge slack is
    active after the solve (i.e. whose original prediction was not kept)."""
    counts_guard = getattr(m, "_counts_guard", None)
    slack_vars = [v for v in m.getVars() if v.VarName.startswith("slacks[")]
    lost = [int(v.VarName.split("[")[1].rstrip("]")) for v in slack_vars if v.X > 1e-6]
    msg = f"  [Hinge] guarded training OC groups with active slack: {len(lost)}/{len(slack_vars)}"
    if counts_guard is not None and lost:
        msg += f" (covering {int(counts_guard[lost].sum())} training examples, i.e. 'Lost' in the confusion matrix)"
    print(msg)


def _solve_linear_problem(m, W, delta, touched):
    m.setParam('OutputFlag', 0)                 # silent mode
    m.setParam('TimeLimit', GUROBI_TIMEOUT)     # set max time for gurobi
    m.setParam('Seed', SEED)
    # (!) big-M constraints  score[c] >= score[f] + eps - M*(1-b)  with a valid
    # (large) M are only sound if the binaries are (near-)exact: with the default
    # IntFeasTol=1e-5, b=0.99999 relaxes the constraint by M*1e-5 (~0.4 for a
    # 20-tree ensemble), enough to accept an unrepaired fault. Keep M*IntFeasTol << eps.
    m.setParam("IntFeasTol", 1e-9)
    m.setParam('FeasibilityTol', 1e-9)
    # m.setParam("OptimalityTol", 1e-9) #1e-9)
    m.setParam('NumericFocus', 2) #2)
    # m.setParam("DualReductions", 0)

    # Hard constraints must hold; allow early stop on objective only.
    m.Params.MIPFocus = 1            # find a feasible incumbent early
    m.Params.MIPGap = 0.50           # don't need the absolute optimum
    m.setParam('Presolve', 2)        # aggressive presolve
    m.setParam('Heuristics', 0.5)    # aggressive heuristics
    m.setParam('Threads', 8)         # use more threads if available

    # let Gurobi decide whether we want cuts
    m.setParam('Cuts', -1)
    # stop when first solution found
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
    elif m.status == GRB.INF_OR_UNBD:
        print(f"    - problem INFEASIBLE or UNBOUNDED (try setting DualReductions=0)")
        return "infeasible", None
    elif m.status == GRB.UNBOUNDED:
        print(f"    - problem UNBOUNDED!")
        return "infeasible", None
    elif m.status == GRB.TIME_LIMIT:
        print("    - gurobi TIMED OUT!")
        if m.SolCount > 0:
            print(f"    - suboptimal solution found (obj: {m.objVal:.3f})")
            status = "suboptimal"
        else:
            return "timeout", None
    else:
        print(f"    - optimization ended with status {m.status}.")
        raise RuntimeError(f"Unexpected Gurobi status: {m.status}")

    # Reconstruct full leaf value matrix: original W plus the optimized deltas
    optimized_W = W + delta.X
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

def get_oc(x, at):
    "Creates indicator vector with 1s in correspondance of leaves activated by (single) example x."
    num_cols = sum(len(t.get_leaf_ids()) for t in at)
    phi = np.zeros((1, num_cols))

    offset = 0
    for t in at:
        leaf_ids = t.get_leaf_ids()
        leaf2index = {leaf_id: i for i, leaf_id in enumerate(leaf_ids)}
        col_index = offset + leaf2index[t.eval_node(x)[0]]
        phi[0, col_index] = 1
        offset += len(leaf_ids)

    return csr_matrix(phi)

def _collect_faulty_leaves(at, faults, applicable_actions):
    """
    Get the OCs of the forbidden states.
    """
    faulty = set()
    for i in range(len(faults)):
        fault = faults.iloc[i]
        applicable_actions_for_fault = applicable_actions[i]
        oc = get_oc(fault, at)
        for leaf_idx in oc.indices:
            for action in applicable_actions_for_fault:
                faulty.add((leaf_idx, action))
    
    faulty_leaves = np.array(sorted(faulty), dtype=int)  # shape (N, 2)
    
    tot_leaves = sum(len(t.get_leaf_ids()) for t in at)
    unique_leaves = len(set(idx for idx, _ in faulty))
    print(f"\t\t- faulty leaves: {unique_leaves} ({np.round(unique_leaves/tot_leaves*100)}%)\n")
    
    return faulty_leaves


def extract_big_M_local(at, eps=0.0):
    return 2.0 * LEAF_BOUND * len(at) + eps

def _set_new_leaf_vals_mc(at, base_scores, new_leaf_vals):
    # print("Leaf Values:", new_leaf_vals)
    for class_index, base_score in enumerate(base_scores):
        at.set_base_score(class_index, base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        # print("Original t", t)
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        for class_index in range(len(base_scores)):
            for count, leaf_id in enumerate(leaf_ids):
                t.set_leaf_value(leaf_id, class_index, new_leaf_vals[class_index, offset + count])
        # print("Changed t", t)
        offset += num_leaves
    return at


def _check_modified_weights(W, optimized_weights, touched, fl_only=False):
    changed_weights = np.sum(np.abs((W - optimized_weights).ravel()) > 1e-8)
    if np.any(np.abs(optimized_weights) > LEAF_BOUND + 1e-1):
        print("     - WARNING: some weights have been set to a very large value!")
    changed_columns = np.any(np.abs(W - optimized_weights) > 1e-9, axis=0)
    changed_column_indices = np.where(changed_columns)[0]

    print(f"         - tot. weights changed: {changed_weights}",
          f"({np.round(changed_weights / W.size * 100, 2)}%)",
          f"\n     - tot. leaves changed: {len(changed_column_indices)}",
          f"({np.round(len(changed_column_indices) / W.shape[1] * 100, 2)}%)")
    return changed_weights
