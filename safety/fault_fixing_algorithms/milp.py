from fault_fixing_algorithms.fault_fixer import FaultFixer, check_safety
from utils.utils import *
from fault_fixing_algorithms.helper_functions.milp.leaf_refinement import *
from fault_fixing_algorithms.helper_functions.enforce_feasibility import (
    add_feasibility_tree, find_infeasible_constraints,
)
from policy.tree_ensemble_helpers import is_discrete_domain
OBJECTIVE = "hinge"

import time
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations


def find_local_infeasibility(fault_info, at, minimal=True):
    """
    Identifies Locally infeasible OCs.
    No union of forbidden actions and intersection of applicable actions leaves no allowed action.

    Returns a minimal subset of faults responsible for each such collapse.
    -------
    dict mapping each problematic OC -> {
        "subset": list of (idx, state, forbidden_action, applicable_actions),
        "all_faults_in_oc": list of all faults sharing this OC,
        "num_actions": int,
    }
    """
    xforb = fault_info["faults"]
    yforb = fault_info["faulty_actions"]
    aas_all = fault_info.get("applicable_actions")

    num_actions = at.num_leaf_values()
    universe = set(range(num_actions))

    # ---- Stage 1: group faults by OC (same logic as group_faults_by_oc) ----
    grouped = defaultdict(list)
    for idx, (x, y) in enumerate(zip(xforb.itertuples(index=False), yforb)):
        oc = tuple(get_oc(x, at).toarray()[0])
        actions = aas_all[idx] if aas_all is not None else None
        grouped[oc].append((idx, x, y, actions))

    # ---- Stage 2: per OC, check whether non faulty actions collapses, and if so find a minimal subset ----
    problematic = {}

    for oc, fault_specs in grouped.items():
        # Build exclusion set for each fault: every action this fault rules out,
        # whether by being forbidden or by not being in its applicable set.
        exclusions = []
        for idx, x, y_forb, aas in fault_specs:
            excl = {y_forb}
            if aas is not None:
                excl |= (universe - set(aas))
            exclusions.append((idx, x, y_forb, aas, excl))

        # Does the full group actually leave nothing allowed?
        total_excluded = set().union(*(e[4] for e in exclusions))
        if total_excluded < universe:
            continue  # this OC is fine, skip

        # Find a covering subset
        if minimal:
            subset = _minimum_cover(exclusions, universe)
        else:
            subset = _greedy_cover(exclusions, universe)

        problematic[oc] = {
            "subset": [(idx, x, y_forb, aas) for idx, x, y_forb, aas, _ in subset],
            "all_faults_in_oc": fault_specs,
            "num_actions": num_actions,
        }

    return problematic


def _minimum_cover(exclusions, universe):
    """Brute-force minimum set cover. Fine for small groups (<~20 faults)."""
    n = len(exclusions)
    for k in range(1, n + 1):
        for combo in combinations(exclusions, k):
            if set().union(*(e[4] for e in combo)) >= universe:
                return list(combo)
    return list(exclusions)  # fallback; should be unreachable if caller checked


def _greedy_cover(exclusions, universe):
    """Greedy set cover: ln(n)-approximation, fast on large groups."""
    remaining = set(universe)
    chosen = []
    pool = list(exclusions)
    while remaining:
        best = max(pool, key=lambda e: len(e[4] & remaining))
        if not (best[4] & remaining):
            break  # cannot cover further
        chosen.append(best)
        remaining -= best[4]
        pool.remove(best)
    return chosen

def dump_fault_info(fault_info, filepath):
    """
    Writes fault_info to a text file, one fault per row.
    Overwrites the file if it already exists.
    """
    faults = fault_info["faults"]
    faulty_actions = fault_info["faulty_actions"]
    applicable_actions = fault_info.get("applicable_actions")

    with open(filepath, "w") as f:
        f.write("idx\tstate\tfaulty_action\tapplicable_actions\n")
        for idx, (state, y) in enumerate(zip(faults.itertuples(index=False), faulty_actions)):
            aas = applicable_actions[idx] if applicable_actions is not None else None
            f.write(f"{idx}\t{tuple(state)}\t{y}\t{aas}\n")
            
def extract_hyperparams_from_name(repair_name):
    """Extracting the correct hyperparameters from the repair name. 
    
    Args:
        repair_name (string): The repair name is given as: milp_{lambda_slack}lambda_{num_training_samples}samples_some stuff for data

    Returns:
        lambda_slack, num_training_samples: Weight of L1 leaf changes relative to fidelity loss, and fidelity sample count.
    """
    # Remove the prefix "milp_"
    remainder = repair_name.removeprefix("milp_")
    
    # Split on "lambda_" to separate lambda_slack from the rest
    lambda_part, remainder = remainder.split("lambda_")
    
    # Remove the "samples" suffix to isolate num_training_samples
    samples_part, _ = remainder.split("samples_")
    num_training_samples = int(samples_part)
    lambda_slack = float(lambda_part)
    
    return lambda_slack, num_training_samples
    
    
class MILPFixer(FaultFixer):
    def fix_faults(self, policy): # Returns fixed policy (at_optimized)
        """
        Repair with the generic MILP objective (L1 leaf changes plus fidelity).
        """
        repair_name = self.mr_config["repair_method"]
        lambda_slack, num_samples = extract_hyperparams_from_name(repair_name)
        repair_log = {
            "#its": 1,                          # total number of iterations
            "#trees": 0,                        # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "runtime": 0.0,                     # repair runtime: MILP build + solve only
            # bookkeeping that is NOT repair work (summed into "runtime_other",
            # which the pipeline books under the "other" runtime):
            "runtime_infeasibility": 0.0,       # detecting infeasibility (local OC check, IIS)
            "runtime_feasibility_trees": 0.0,   # learning fault-separator trees
            "runtime_diagnostics": 0.0,         # post-solve sanity checks / reporting
            "runtime_other": 0.0,
        }

        # sample the fidelity set ONCE by index so states and labels stay aligned
        n_samples = min(num_samples, len(self.training_states))
        rng = np.random.RandomState(SEED)
        sample_idx = rng.choice(len(self.training_states), size=n_samples, replace=False)
        train_info = {
            "training_states": self.training_states.iloc[sample_idx].reset_index(drop=True),
            "training_actions": pd.Series(np.asarray(self.training_actions).reshape(-1)[sample_idx]),
        }
        eps = self.mr_config['gur_tol']

        fault_info = {
            "faults": self.faulty_states,
            "faulty_actions": self.faulty_actions,
            "applicable_actions": self.possible_actions
        }

        # dump_fault_info(fault_info, "fault_info.txt")

        def _run_milp(policy):
            status, fixed, _, model, timings = fix_milp(policy, fault_info, train_info, OBJECTIVE, True,
                                                        lambda_slack, eps=eps)
            repair_log["runtime"] += timings["build"] + timings["solve"]
            repair_log["runtime_diagnostics"] += timings["diagnostics"]
            return status, fixed, model

        def _finalize(log):
            log["runtime_other"] = (log["runtime_infeasibility"]
                                    + log["runtime_feasibility_trees"]
                                    + log["runtime_diagnostics"])
            return log

        # i. local infeasibility: faults sharing one OC that leave no allowed action
        len_at_before = len(policy)
        t_inf = time.time()
        problematic = find_local_infeasibility(fault_info, policy, minimal=True)
        repair_log["runtime_infeasibility"] += time.time() - t_inf
        while problematic:
            for oc, info in problematic.items():
                overconstrained_faults = []
                for idx, x, y_forb, aas in info["subset"]:
                    overconstrained_faults.append(x)
                    print(f"  fault #{idx}: state={x}, forbidden={y_forb}, applicable={aas}")
                t_tree = time.time()
                policy = add_feasibility_tree(policy, pd.DataFrame(overconstrained_faults),
                                              discrete=is_discrete_domain(self.benchmark))
                repair_log["runtime_feasibility_trees"] += time.time() - t_tree
                break # Just handle the first overconstrained OC; can be extended to loop over all OCs if needed
            t_inf = time.time()
            problematic = find_local_infeasibility(fault_info, policy, minimal=True)
            repair_log["runtime_infeasibility"] += time.time() - t_inf
        repair_log["#trees"] += len(policy) - len_at_before

        # ii. MILP repair
        gurobi_status, fixed_policy, grb_model = _run_milp(policy)

        if gurobi_status in ["infeasible", "timeout"]:
            
            # try solving infeasibility by adding new trees
            len_at_before = len(policy)
            while gurobi_status == "infeasible":
                # detect which faults are responsible (IIS)
                t_inf = time.time()
                grb_model.computeIIS()
                compact_faults = group_faults_by_oc(policy, self.faulty_states, self.faulty_actions, self.possible_actions)
                __, faults_x = find_infeasible_constraints(grb_model, policy, compact_faults)
                repair_log["runtime_infeasibility"] += time.time() - t_inf

                # separate them with a new tree
                t_tree = time.time()
                policy = add_feasibility_tree(policy, faults_x, discrete=is_discrete_domain(self.benchmark))
                repair_log["runtime_feasibility_trees"] += time.time() - t_tree

                # re-solve
                gurobi_status, fixed_policy, grb_model = _run_milp(policy)

            repair_log["#trees"] += len(policy) - len_at_before
                
            # [T2] termination scenario no. 2: problem times out or still infeasible 
            if gurobi_status in ["timeout", 'infeasible']:
                repair_log["repair_status"] = gurobi_status
                fixed_policy = None
                return None, _finalize(repair_log)
            
        repair_log["repair_status"] = gurobi_status
        
        # vii. Assert and guarantee repair works:
        # print("########## Original Policy")
        # for t in policy:
        #     print(t)
            
        # print("########## Final Policy")
        # for t in fixed_policy:
        #     print(t)
        t_diag = time.time()
        check_safety(policy, fixed_policy, self.faulty_states, self.faulty_actions, self.possible_actions)
        repair_log["runtime_diagnostics"] += time.time() - t_diag
        return fixed_policy, _finalize(repair_log)

