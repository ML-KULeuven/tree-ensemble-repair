import veritas 
from fault_fixing_algorithms.helper_functions.leaf_refinement import *
from utils.utils import *
from policy.tree_ensemble_helpers import is_discrete_domain, transform_addtree

# PART 1: directly modify at for clashing faults sharing the SAME OC

def decouple_ocs(at, xforb, yforb, applicable_actions=None, discrete=False):
    """
    Adds trees to the AddTree to split each OC where there's no allowed actions.
    """
    one_more_time = True
    while one_more_time:

        # print(f"\t\t- starting at length: {len(at)}")
        one_more_time = False

        # (new) map each forbidden state to the list of forbidden classes and applicab. actions
        compact_faults = group_faults_by_oc(at, xforb, yforb, applicable_actions)

        for idx, (oc, fault_specs) in enumerate(compact_faults.items()):

            # collect all forbidden and allowed classes for this OC
            # --> forbidden classes: either not applicable or found in faults
            all_forbidden = set()
            all_allowed = set(range(at.num_leaf_values()))
            for __, __, y_forbidden, aas in fault_specs:

                all_forbidden.add(y_forbidden)
                if aas is not None:
                    all_allowed &= set(aas)
                all_allowed -= all_forbidden
                all_forbidden = set(range(at.num_leaf_values())) - all_allowed 
                
                # no allowed actions: need to add a tree to decouple OCs
                if not all_allowed:
                    # print(f"\t\t- PROBLEM: no allowed classes for OC no. {idx}")
                    infeasible_indices = find_minimal_infeasible_subset(fault_specs, at.num_leaf_values())
                    assert len(infeasible_indices) >= 2, "Less than 2 clashing faults?"
                    print(f"\n\t\t- found clashing faults with same OC: {infeasible_indices}")
                    print(f"\t\t- OC: {oc}")

                    fx = pd.DataFrame([fs[1] for fs in fault_specs if fs[0] in infeasible_indices])
                    fx = fx.drop_duplicates().reset_index(drop=True) # remove duplicates 
                    at = add_feasibility_tree(at, fx, discrete=discrete)
                    one_more_time = True
                    break
            if one_more_time:
                break
    print(f"\t\t- at length after adding feasibility trees: {len(at)}\n")
    return at


def find_minimal_infeasible_subset(fault_specs, num_classes):
    """
    Given a list of fault specs for a single OC, find the smallest subset
    that leads to no allowed actions (i.e., empty intersection).
    
    Each fault spec is a tuple: (index, state, forbidden_class, applicable_actions)
    """
    from itertools import combinations
    def allowed_actions_subset(specs):
        all_allowed = set(range(num_classes))
        for (_, _, _, allowed) in specs:
            if allowed is not None:
                all_allowed &= set(allowed)
        all_forbidden = {fs[2] for fs in specs}
        return all_allowed - all_forbidden
    n = len(fault_specs)
    for r in range(2, n + 1):
        for subset in combinations(fault_specs, r):
            if not allowed_actions_subset(subset):
                return [fs[0] for fs in subset]  # return indices of minimal infeasible subset
    # return []  # no infeasibility found
    raise ValueError("Infeasible model but no infeasible indices found?")


def add_feasibility_tree(at, x, discrete=False):
    from sklearn.tree import DecisionTreeClassifier
    y = pd.Series(range(x.shape[0]), dtype=int)
    clf = DecisionTreeClassifier(
        criterion='entropy',      # force splitting where possible
        max_depth=None,          
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0
    )
    clf.fit(x, y)

    # sanity check
    yhat = clf.predict(x)
    acc = (yhat == y.values).mean()
    print(f"\t\t- added new fault separator tree with acc={acc:.3f}")

    t = veritas.get_addtree(clf, silent=True)

    # set all leaf values to a zero vector
    t = t.make_multiclass(0, at.num_leaf_values())
    for lid in t[0].get_leaf_ids():
        for c in range(at.num_leaf_values()):
            t[0].set_leaf_value(lid, c, 0)

    # if acc<1:
    #     print("\t\t- NOTE: this tree does not separate all faults perfectly, " \
    #           "but it will help to decouple OCs.")
    #     print(x)
    #     print(t[0])

    # test
    # print(t[0])
    # print(t[0].eval_node(fx.iloc[0]))
    # print(t[0].eval_node(fx.iloc[1]))
    # exit()

    # discrete domains: DecisionTreeClassifier picks midpoint thresholds (e.g. 3.5);
    # round them up so every split is an integer (identical branching on integer states)
    if discrete:
        t = transform_addtree(t)

    # attach tree to at
    at.add_tree(t[0])
    return at


# PART 2: enforce feasibility when gurobi still fails 
#          ---> here IIS must contain multiple faults with multiple OCs

def enforce_feasibility(m, at, xtrain, fx, fy, aa, log, fl_only, gur_tol, discrete=False):
    print(f"\n\t(!) Infeasible model detected with {len(at)} trees.")

    infeasible = 0
    while infeasible>=0:

        print(f"\n\t\t *** feasibility iteration {infeasible} ***\n")

        # extract why model is infeasible (timed as infeasibility detection)
        t_inf = time.time()
        m.computeIIS()

        # create compact faults representation
        compact_faults = group_faults_by_oc(at, fx, fy, aa)

        # find infeasible constraints, gather responsible faults
        __, faults_x = find_infeasible_constraints(m, at, compact_faults)
        log["runtime_infeasibility"] = log.get("runtime_infeasibility", 0.0) + (time.time() - t_inf)

        # slightly modify the model to make it feasible (timed as tree learning)
        t_tree = time.time()
        at = add_feasibility_tree(at, faults_x, discrete=discrete)
        log["runtime_feasibility_trees"] = log.get("runtime_feasibility_trees", 0.0) + (time.time() - t_tree)

        # re-optimize the model
        faulty_leaves = count_faulty_leaves(at, fx)
        # print("\t(!) Retrying to solve the problem enforcing feasibility:")
        W, Phi = build_W_phi(xtrain, at)
        m, new_weights = get_linear_problem(at, W, Phi, fx, fy, 
                                            faulty_leaves, fl_only,
                                            aa, gur_tol, verbose=False)
        gur_time = time.time()
        gurobi_status, optimized_weights = solve_linear_problem(m, W, new_weights)
        # log["runtimes"]["gurobi"] += time.time() - gur_time
        log["runtime"] += time.time() - gur_time
        # assert gurobi_status != "infeasible", "Still infeasible after adding new tree!"

        if gurobi_status != "infeasible":
            infeasible = -1  # exit loop
        else:
            infeasible += 1
    return at, W, Phi, faulty_leaves, gurobi_status, optimized_weights, log


def find_infeasible_constraints(m, at, compact_faults):

    indices = []
    # oc_action_label = []
    for c in m.getConstrs():
        # the constraint is infeasible
        if c.IISConstr:  
            # print(f"\t\t- infeasible constraint: {c.constrName}")
            # extract indices of forbidden states that are part of the IIS
            if c.constrName.startswith("at_least"):
                idx = int(c.constrName.split('_')[-1])
                indices.append(idx)
                
    # analyze IIS faults' OCs
    ocs, fx = [], []
    for idx, (fx_oc, fault_specs) in enumerate(compact_faults.items()):
        if idx in indices:
            ocs.append(fx_oc)
            fx.append(fault_specs[0][1])
            # TODO: re-add stuff for debug?

    print(f"\t\t- indices of forbidden states in IIS: {indices}")
    # for i, fx_oc, actions, label in oc_action_label:
    #     print(f"\t\t\t --> forbidden state OC: {fx_oc}, applicable actions: {actions}, forbidden label: {label}")
        
    if len(indices) < 2:
        raise ValueError("Less than 2 faults in IIS?")
    else:
        # check if all OCs are the same
        ocs = set(ocs)
        if len(ocs) > 1:
            print(f"\t\t- NOTE: IIS contains {len(indices)} faults " \
                    f"but {len(ocs)} different OCs.")
            # print OCs
            print(f"\t\t--> OCs: {ocs}")
        else:
            # print(f"\t\t- all faults share the same OC: {list(ocs)[0]}")
            raise ValueError("All faults share the same OC, but that shouldn't happen anymore!")
        return indices, pd.DataFrame(fx)
    
   