# from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
from gurobipy import Model, GRB, quicksum
from utils.utils import SEED

# hardcoded bound for magnitude of optimized leaf values (to avoid overflows)
LEAF_BOUND = 1e3 #GRB.INFINITY 
# hardcoded timeout for gurobi solver
GUROBI_TIMEOUT = 3600*3  # 3h # 300 # 5 min

def group_faults_by_oc(at, xforb, yforb, applicable_actions=None):
    """
    Groups faults by their OCs.
    Returns a dictionary where keys are OCs and values are lists of tuples
    (index, state, forbidden_class, applicable_actions).
    """
    from collections import defaultdict
    
    grouped_faults = defaultdict(list)
    for idx, (x, y) in enumerate(zip(xforb.itertuples(index=False), yforb)):
        oc_x = tuple(at[t].eval_node(x)[0] for t in range(len(at)))
        actions = applicable_actions[idx] if applicable_actions else None
        grouped_faults[oc_x].append((idx, x, y, actions))

    # sanity check: all faults for each OC must have the same phi
    for idx, (__, fault_specs) in enumerate(grouped_faults.items()):
        avoid_condition_phi = get_oc(fault_specs[0][1], at)
    
        if len(fault_specs) > 1:
            for i in range(1, len(fault_specs)):
                assert np.array_equal(get_oc(fault_specs[i][1], at).toarray(), 
                                    avoid_condition_phi.toarray()), \
                    f"Fault OC no. {idx} has non unique phi???"
    return grouped_faults

def build_W_phi(x, at):
    """
    Builds two matrices to recreate the ensemble output as a matrix multiplication.

    - W: each row has all the leaf values of the ensemble for a specific class
         (so one column for each leaf in the ensemble)

    - phi: each row is an indicator vector with 1s in the columns corresponding to 
           the leaves activated by the considered input example
    """
    time_ = time.time()
    row_ind, col_ind = [], []
    num_rows = x.shape[0]
    num_cols = 0
    tot_cols = sum(len(t.get_leaf_ids()) for t in at)
    num_leaf_values = at.num_leaf_values()
    W = np.zeros((num_leaf_values, tot_cols))
    for tree_index, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        offset = num_cols
        num_cols += num_leaves

        leaf2index = np.zeros(t.num_nodes(), dtype=int)
        for cnt, id in enumerate(leaf_ids):
            leaf2index[id] = cnt
        row_ind += list(range(num_rows))
        col_ind += list(map(lambda u: offset + leaf2index[u], t.eval_node(x)))
        # (!) alternative code for numpy>=2.0 (veritas eval_node is buggy)
        #     --> note that this is way slower, use only if previous line fails
        # manual_node_eval = np.array([t.eval_node(x.iloc[i].values) 
        #                                for i in range(len(x))]).flatten()
        # col_ind += list(map(lambda u: offset + leaf2index[u], manual_node_eval))
        offset_ = sum(len(at[i].get_leaf_ids()) for i in range(tree_index))
        for pos in range(num_leaf_values):
            for cnt, leaf_id in enumerate(leaf_ids):
                W[pos, offset_ + cnt] = t.get_leaf_value(leaf_id, pos)

    
    phi = csr_matrix((np.ones(len(row_ind), dtype=np.float64),
                              (row_ind, col_ind)), shape=(num_rows, num_cols))
    num_zeros = np.count_nonzero(phi.toarray() == 0)
    num_ones = np.count_nonzero(phi.toarray() == 1)
    # manual check for the computation of ensemble output
    base_score = np.array([at.get_base_score(l) for l in range(at.num_leaf_values())])
    result = np.matmul(phi.toarray(), W.T) + base_score
    assert np.allclose(result - at.eval(x), 0), "Something wrong with W and phi matrices!"

    print(f"\t\t- time to build W and phi: {time.time()-time_:.3f} s.")
    return W, phi


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


def count_faulty_leaves(at, forbidden_x):
    """
    Get the OCs of the forbidden states.
    """
    faulty_leaves = []
    for fx in forbidden_x.values:
        oc = get_oc(fx, at)
        faulty_leaves.append(oc.indices.tolist())
    faulty_leaves = np.unique(np.concatenate(faulty_leaves))
    tot_leaves = sum(len(t.get_leaf_ids()) for t in at)
    print(f"\t\t- faulty leaves: {len(faulty_leaves)}", \
        f"({np.round(len(faulty_leaves)/tot_leaves*100)}%)\n") 
    return faulty_leaves
    

def get_linear_problem(at, W, phi, xforb, yforb, faulty_leaves, fl_only=False,
                        applicable_actions=None, eps=1e-4, verbose=False):
    "Builds a linear program to tune leaf values (in W) adding constraints."

    from contextlib import redirect_stdout
    from io import StringIO

    # print("Shapes: ")
    # print("\t-phi: ", phi.shape)
    # print("\t-W: ", W.shape)
    t = time.time()

    # 1. Define the optimization problem
    with redirect_stdout(StringIO()): 
        m = Model()

    # 2. Variables to optimize: new leaf values
    new_weights = m.addMVar(shape=W.shape, lb=-LEAF_BOUND, ub=LEAF_BOUND, 
                            name="new_weights")
    # print("Done with new_weights")
    # set initial values for new_weights to W to help the solver
    # for i in range(W.shape[0]):
    #     for j in range(W.shape[1]):
    #         new_weights[i, j].start = W[i, j]
    new_weights.start = W


    # (new) possibly freeze weights for leaves not visited by at least one fault
    if fl_only: 
        for col in range(W.shape[1]):
            if col not in faulty_leaves:
                for row in range(W.shape[0]):
                    m.addConstr(new_weights[row, col] == W[row, col], \
                                name=f"constant_weight_{row}_{col}")

    # 3. Objective function: minimize absolute value difference btw new and old weights

    # Create auxiliary variables for absolute differences
    abs_diff = m.addMVar(shape=W.shape, lb=0, name="abs_diff")
    # print("Done with abs_diff")
    
    # Add constraints for absolute differences using vectorized operations
    m.addConstr(abs_diff >= new_weights - W, name="abs_diff_pos")
    # print("Done with abs_diff_pos")
    m.addConstr(abs_diff >= W - new_weights, name="abs_diff_neg")
    # print("Done with abs_diff_neg")

    # Set the objective to minimize the sum of absolute differences
    objective = abs_diff.sum()
    m.setObjective(objective, GRB.MINIMIZE)

    # 4. Constraints: Avoid given state-value pairs (x, y)
                
    def avoid_forbidden_pairs_mc(m, new_weights, xforb, yforb, at,
                                 applicable_actions, eps=1e-4):
        
        # group faults by their OCs
        compact_faults = group_faults_by_oc(at, xforb, yforb, applicable_actions)

        # manually set big-M for constraints to smallest possible value 
        # --> should help reducing numerical instabilities
        M = extract_big_M_local(at, eps)

        # extract allowed/forbidden classes per each OC and add constraints 
        for idx, (__, fault_specs) in enumerate(compact_faults.items()):

            avoid_condition_phi = get_oc(fault_specs[0][1], at)

            all_forbidden = set()
            all_allowed = set(range(at.num_leaf_values()))
            for __, __, y_forbidden, aas in fault_specs:
                all_forbidden.add(y_forbidden)
                if aas is not None:
                    all_allowed &= set(aas)
            all_allowed -= all_forbidden
            # NOTE: only the explicitly forbidden actions are constrained (same
            # semantics as milp). Inapplicable actions are masked out at
            # prediction time, so forcing the policy to beat them too only tightens
            # the MILP for no safety gain.
            assert all_allowed, f"No allowed classes for fault no. {idx}??"

            # precompute all scores once
            scores = []
            for cls in range(at.num_leaf_values()):
                score = (avoid_condition_phi @ new_weights[cls, :].T) + at.get_base_score(cls)
                scores.append(score)

            # encode that at least one allowed class has to beat all forbidden classes
            b_vars = []
            for c in all_allowed:
                # print("Could it be b addVar")
                b = m.addVar(vtype=GRB.BINARY, name=f"b_allow_{idx}_{c}")
                # print("Done with b addVar")
                b_vars.append(b)
                for f in all_forbidden:
                    m.addConstr(scores[c] >= scores[f] + eps - M * (1 - b),
                                name=f"force_{c}_gt_{f}_{idx}")
            m.addConstr(quicksum(b_vars) >= 1, name=f"at_least_one_valid_gt_forbidden_{idx}")

    assert at.num_leaf_values() > 1, "Binary classification not supported yet."
    avoid_forbidden_pairs_mc(m, new_weights, xforb, yforb, at, applicable_actions, eps)

    print(f"\t\t- time to set up gurobi: {time.time()-t:.3f} s.")

    m.update()
    
    if verbose:
        # print total number of constraints per type
        num_constraints = m.numConstrs
        num_abs_diff = sum(1 for c in m.getConstrs() if c.constrName.startswith("abs_diff_"))
        num_constant = sum(1 for c in m.getConstrs() if c.constrName.startswith("constant_weight_"))
        num_force = sum(1 for c in m.getConstrs() if c.constrName.startswith("force_"))
        num_at_least_one = sum(1 for c in m.getConstrs() if c.constrName.startswith("at_least_one_valid_gt_forbidden_"))
        print(f"\t\t- total constraints: {num_constraints}, "
                f"\n\t\t\t- abs_diff: {num_abs_diff}, "
                f"\n\t\t\t- constant: {num_constant}, "
                f"\n\t\t\t- forbidden: {num_force}, "
                f"\n\t\t\t- at_least_one_valid_gt_forbidden: {num_at_least_one}")
    return m, new_weights


def solve_linear_problem(m, W, new_weights):
    """ Solves the linear problem. """

    m.setParam('OutputFlag', 0) # silent mode
    m.setParam('TimeLimit', GUROBI_TIMEOUT) # set max time for gurobi
    m.setParam('Seed', SEED)  # Set a seed for reproducibility
    # (!) big-M constraints  score[c] >= score[f] + eps - M*(1-b)  with a valid
    # (large) M are only sound if the binaries are (near-)exact: with the default
    # IntFeasTol=1e-5, b=0.99999 relaxes the constraint by M*1e-5 (~0.4 for a
    # 20-tree ensemble), enough to accept an unrepaired fault. Keep M*IntFeasTol << eps.
    m.setParam("IntFeasTol", 1e-9)
    m.setParam('FeasibilityTol', 1e-9)
    # m.setParam("OptimalityTol", 1e-8)
    m.setParam('NumericFocus', 3) # 0-3 increasingly more careful for numerical issues
    # we want quick feasible solutions
    m.Params.MIPGap = 0.9  # Allow a 90% gap for suboptimal solutions
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    m.setParam('Presolve', 2)  # Aggressive presolve
    # m.setParam('Cuts', 0)      # Disable cuts to reduce overhead
    m.setParam('Heuristics', 0.5)   # Increase heuristic effort (default is 0.05)
    m.setParam('Threads', 8)   # Use more threads if available

    print("\t\t- optimization begins")
    m.optimize()

    # init. status return string
    status = "unknown"

    # Check the optimization status and output
    if m.status == GRB.OPTIMAL:
        print(f"\t\t- optimization done -->",
              f"obj. value: {m.objVal:.3f},",
              f"time: {m.Runtime:.3f} s")
        status = "optimal"
    elif m.status == GRB.INFEASIBLE:
        print(f"\t\t- problem INFEASIBLE! (runtime: {m.Runtime:.3f} s)")

        # # Analyze infeasible instances (now in ad hoc file)
        # m.computeIIS()
    
        # # Save to a file (optional)
        # m.write("model.ilp")  # or "model.iis" in LP format
        
        # # Print out the IIS
        # for c in m.getConstrs():
        #     if c.IISConstr:  # This constraint is part of the IIS
        #         print(f"Infeasible constraint: {c.constrName}")
    
        status = "infeasible"
        return status, None
    # elif m.status == GRB.TIME_LIMIT:
    #     print(f"\t\t- gurobi TIMED OUT!")
    #     status = "timeout"
    #     return status, None
    elif m.status == GRB.TIME_LIMIT:
        print(f"\t\t- gurobi TIMED OUT!")
        if m.SolCount > 0:
            print(f"\t\t- suboptimal solution found (obj: {m.ObjVal:.3f})")
            status = "suboptimal"
        else:
            return "timeout", None
    else:
        print(f"\t\t- optimization ended with status {m.status}.")
        raise RuntimeError(f"Optimization ended with unexpected status {m.status}.!")
        # exit()

    # Extract optimized weights
    optimized_weights = np.array(
        [new_weights[i, j].X 
        for i in range(W.shape[0]) 
        for j in range(W.shape[1])]
    ).reshape(W.shape)
    return status, optimized_weights


def check_modified_weights(W, optimized_weights, faulty_leaves, fl_only=False):
    """
    Check how many weights have been changed,
    and how many of these are in faulty leaves.
    """
    # count number of weights changed
    changed_weights = np.sum(np.abs((W-optimized_weights).ravel()) > 1e-8)
    # print magntiude of weights changed
    # print(f"changed weights: {np.abs(W-optimized_weights).ravel()[np.abs(W-optimized_weights).ravel() > 1e-8]}")
    
    # check that no weight has been given an extremely large value
    if np.any(np.abs(optimized_weights) > 1e3):
        print("WARNING: some weights have been set to a very large value!")
        print(f"changed weights: {np.abs(W-optimized_weights).ravel()[np.abs(W-optimized_weights).ravel() > 1e-8]}")
    
    # debug
    # print(faulty_leaves)
    # for col in range(W.shape[1]):
    #     if col not in faulty_leaves:
    #         for row in range(W.shape[0]):
    #             if not optimized_weights[row, col] == W[row, col]:
    #                 print(f"mismatch in column {col} and row {row} - ", \
    #                       f"difference: {optimized_weights[row, col] - W[row, col]}")  
                  
    # count number of leaves interested by at least one change
    changed_columns = np.any(np.abs(W - optimized_weights) > 1e-9, axis=0)
    changed_column_indices = np.where(changed_columns)[0]
    changed_faulty_columns = np.sum(np.isin(changed_column_indices, faulty_leaves))

    print(f"\t\t- tot. weights changed: {changed_weights}", \
          f"({np.round(changed_weights/W.size*100, 2)}%)", \
          f"\n\t\t- tot. leaves changed: {len(changed_column_indices)}", \
          f"({np.round(len(changed_column_indices)/W.shape[1]*100, 2)}%)", \
          f"\n\t\t   ... of which {changed_faulty_columns} are faulty leaves")
    
    if fl_only:
        # assert len(changed_column_indices) == changed_faulty_columns, \
        #     "Seems like you're somehow modifying some non-faulty leaves??"
        if not len(changed_column_indices) == changed_faulty_columns:
            print("\t\t- WARNING: modified non-faulty leaves due to small numerical issues!")
           
    return changed_weights
    

# Timo copyright
def set_new_leaf_vals(at, base_score, new_leaf_vals):
    at.set_base_score(0, base_score)
    offset = 0
    for __, t in enumerate(at):
        for count , leaf_id in enumerate(t.get_leaf_ids()):
            t.set_leaf_value(leaf_id, 0, new_leaf_vals[offset + count])
        offset += count + 1
    return at

def set_new_leaf_vals_mc(at, base_scores, new_leaf_vals):
    # same as above, but for a multiclass addtree 
    for class_index, base_score in enumerate(base_scores):
        at.set_base_score(class_index, base_score)

    offset = 0
    for tree_index, t in enumerate(at):
        num_leaves = len(t.get_leaf_ids())
        for class_index in range(len(base_scores)):
            for count, leaf_id in enumerate(t.get_leaf_ids()):
                t.set_leaf_value(leaf_id, class_index, new_leaf_vals[class_index, offset + count])
        offset += num_leaves
    return at

def extract_big_M_local(at, eps=0.0):
    return 2.0 * LEAF_BOUND * len(at) + eps
