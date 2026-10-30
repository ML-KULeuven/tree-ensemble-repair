# from collections import defaultdict
import numpy as np
# import veritas
from collections import defaultdict

from fault_fixing_algorithms.fault_fixer import FaultFixer, check_safety
from utils.utils import *
from policy.tree_ensemble_helpers import is_discrete_domain

class PenaltyTreeFixer(FaultFixer):
    def fix_faults(self, policy): # Returns fixed policy (at_optimized)
        """
        Method for fault fixing using Gurobi
        """
        repair_log = {
            "#its": 1,                          # total number of iterations
            "#trees": 0,                        # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "runtime": 0.0,             # repair runtime
        }
        
        time_baseline = time.time()
        fixed_policy = policy.copy()
        add_penalty_trees(self.faulty_states, self.faulty_actions, fixed_policy,
                          discrete=is_discrete_domain(self.benchmark))
        repair_log["runtime"] += time.time() - time_baseline
        repair_log["repair_status"] = "optimal" # proxy to say we did something and it succeded 
        check_safety(policy, fixed_policy, self.faulty_states, self.faulty_actions, self.possible_actions)
        return fixed_policy, repair_log

def compute_penalty_value(at):
    """
        The penalty leaves should have a larger negative value than any value in our original ensemble. 
        We do this by selecting penalty = min_T - max_T
    """
    penalty = 0
    classes = range(0, at.num_leaf_values())
    for t in at:
        min_value = np.inf
        max_value = -np.inf
        label_minmax_values = t.find_minmax_leaf_value(0)
        flat_values = [v for pair in label_minmax_values for v in pair]
        min_value = min(flat_values)
        max_value = max(flat_values)
        penalty += min_value - max_value
    return penalty
    
def add_penalty_trees(forbidden_x, forbidden_y, at, discrete=False):
    # forbidden_x is a table with num_faults rows and num_vars columns. 
    # The columns are ordered according to features of the ensemble.
    # 0. calculate penalty value
    penalty = compute_penalty_value(at)
    # 1. Convert forbidden_x, forbidden_y into a map with <state, forbidden actions>
    # Convert each row of forbidden_x to a list
    states = forbidden_x.values.tolist()
    # Map each forbidden_x row (as list) to corresponding forbidden_y values
    temp_map = defaultdict(set)

    for state, action in zip(states, forbidden_y):
        temp_map[tuple(state)].add(action)  # use tuple temporarily to group

    # Convert back to list of lists for keys
    faults = [[list(k), sorted(v)] for k, v in temp_map.items()]
    # 2. Iterate through each element of the map and add a penalty tree
    for state, forbidden_actions in faults:
        t = at.add_tree()
        add_penalty_tree(state, forbidden_actions, penalty, t, discrete=discrete)
        
def add_penalty_tree(state, action_labels, penalty, t, discrete=False):
    # print("\nAdding penalty tree for state:", state, "forbidden actions:", action_labels)
    assert len(action_labels) > 0
    num_features = len(state)
    string = ""
    all_classes = range(0, t.num_leaf_values())
    for feature in range(0, num_features):
        last_var = (feature == num_features - 1)
        split_val = state[feature]
        # FIXED: if we are in a real domain but split_val=5.0,
        # old code will not be parsed as a float as it should
        # --> e.g., bug in bouncing ball ablation
        # if (int(split_val) == split_val):
        # discrete domains: states arrive as floats from the DataFrame (e.g. 5.0),
        # so build the equality test with exact integer thresholds [v, v+1)
        # instead of the float construction [v, v+1e-5) below
        if discrete or isinstance(split_val, int):
            split_val = int(round(split_val))
            t.split(t[string], feature, split_val + 1)
            t.split(t[string + "l"], feature, split_val)
            for label in all_classes:
                t.set_leaf_value(t[string + "r"], label, 0)
                t.set_leaf_value(t[string + "ll"], label, 0)
            if last_var:
                for label in all_classes:
                    t.set_leaf_value(t[string + "r"], label, 0)
                    t.set_leaf_value(t[string + "ll"], label, 0)
                for label in action_labels:
                    t.set_leaf_value(t[string + "lr"], label, penalty)
        else:
            t.split(t[string], feature, split_val + 0.00001) # manually set threshold
            # t.split(t[string], feature, split_val + 1e-8) # manually set threshold
            # print("THRESHOLD:", split_val + 1e-8)
            t.split(t[string + "l"], feature, split_val)
            for label in all_classes:
                t.set_leaf_value(t[string + "r"], label, 0)
                t.set_leaf_value(t[string + "ll"], label, 0)
            if last_var:
                for label in all_classes:
                    t.set_leaf_value(t[string + "r"], label, 0)
                    t.set_leaf_value(t[string + "ll"], label, 0)
                for label in action_labels:
                    t.set_leaf_value(t[string + "lr"], label, penalty)
        string = string + "lr"

    # print(state, action_labels, penalty)
    # print(t)
    # assert np.sum(t.eval(state)) == penalty, \
    #     f"Something wrong with the penalty tree? np.sum(t.eval(state)) = {np.sum(t.eval(state))}, expected penalty = {penalty}, penalty tree: {t}"
    assert np.abs(np.sum(t.eval(state))) > 0, f"Something wrong with the penalty tree?"
    # print(t.eval(state))
    # exit()
    return   