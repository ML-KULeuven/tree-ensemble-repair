import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import veritas

MAX_TIME = 10*60   # 10 min - max time to run fairness verification

# @TODO: alternative (and faster) fairness verification using ICML26 index?
def run_fairness_check(at, d, dtest, sensitive_col, ordinal_cols):
    
    # CHECK #1: exhaustive enumeration with veritas
    solutions, feat_map, __, stop_reason = veritas_fairness_check(at, d, sensitive_col)
    print(f"\n >> Found {len(solutions)} initial solutions (stop reason: {stop_reason})")
    # remove solutions where both instances are classified in the same class
    t_sc = time.time()
    solutions = remove_same_class_solutions(d, at, feat_map, solutions, ordinal_cols)
    # check we don't have solutions where the two instances coincide
    # disabled: guaranteed by design
    # check_no_identical_instances(d, feat_map, solutions, ordinal_cols)
    # optionally, remove solutions where >1 sensitive columns are set to true
    # (--> semantically wrong, e.g., African-American==1 and Caucasian==1 simultaneously)
    if isinstance(sensitive_col, list) and len(sensitive_col) > 1:
        print(f"\n >> Removing solutions where >1 sensitive columns are set to true")
        solutions = remove_infeasible_solutions(solutions, feat_map, d, [sensitive_col], ordinal_cols)
    print(f" (removed invalid solutions in {time.time() - t_sc:.3f} seconds)")

    # CHECK #2: empirically flip sensitive attribute in the test set, check if prediction changes
    emp_fairness = evaluate_empirical_fairness(at, dtest, sensitive_col)

    enumerable = True
    # Hard check: 
    #   If exact search terminated exhaustively with no solutions found,
    #   we cannot have empirical fairness < 1.0
    if stop_reason == veritas.StopReason.NO_MORE_OPEN and len(solutions) == 0:
        assert emp_fairness == 1.0, \
            f"Empirical fairness should be 1.0 but is {emp_fairness} with no unfair boxes found??"
    elif stop_reason in (veritas.StopReason.OUT_OF_TIME, veritas.StopReason.OUT_OF_MEMORY):
        print(f" * (!) Warning: veritas could not enumerate OC-space ({stop_reason}). Terminating...")
        enumerable = False

    return solutions, feat_map, emp_fairness, enumerable


def veritas_fairness_check(at, d, column, verbose=False):

    if isinstance(column, str):
        nonfixed_cols = {column}
    else:
        nonfixed_cols = set(column)

    if verbose:
        for col in nonfixed_cols:
            print(f"Unique values for attribute '{col}': ", d.X[col].unique())

    # create AddTree that contract two examples, duplicating nonfixed columns
    # and using the same feature IDs for the fixed columns
    at_contrast, featmap = constrast_two_examples(at, d.X.columns, nonfixed_cols)

    # NEW: if the model never uses the sensitive attribute, it's fair by design
    #      (sometimes happens in BuDRO)
    nonfixed_ids = {featmap.get_feat_id(c, i) for c in nonfixed_cols for i in (0, 1)}
    if not (nonfixed_ids & set(at_contrast.get_splits().keys())):
        # model never reads the sensitive attribute -> contrast tree is identically 0
        print(f" (!) Model never reads sensitive attribute(s) {nonfixed_cols} -> fair by design.")
        return [], featmap, None, veritas.StopReason.NO_MORE_OPEN

    # optimize output of at_contrast to find max difference in predictions
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    # Note: using config.ignore_state_when_worse_than = 0.0 generates solution boxes
    #       with output 0.0 where instance0 = instance1 (i.e., useless)
    #       --> hence I'm using this small eps
    config.ignore_state_when_worse_than = 1e-6 # >0: instance 0's output > instance 1's output
    # aggressive veritas search
    config.focal_eps = 0.95
    config.max_focal_size = 100
    # try enumerating ALL violating boxes, not only the one with max output
    config.stop_when_optimal = False
    config.stop_when_num_solutions_exceeds = int(1e10)
    config.stop_when_num_new_solutions_exceeds = int(1e10)
    # tmp: testing only retrieving up to N unfair boxes per call
    # config.stop_when_num_solutions_exceeds = 1000
    # config.stop_when_num_new_solutions_exceeds = 1000
    # allocate memory for veritas search
    config.max_memory = 24 * 1024 * 1024 * 1024   # 24 GB

    # optionally constrain certain feature values
    # (e.g., look only for cases where instance 0 describes a male person)
    # prune_box = {featmap.get_feat_id("male", 0): veritas.Interval(0, 0.5)}
    # prune_box = {feat_map.get_feat_id("age", 0): veritas.Interval(-np.inf, 0)}
    prune_box = {}
    search = config.get_search(at_contrast, prune_box)

    # run search
    t = time.time()
    bounds = []
    num_search_steps_per_iteration = 100
    stop_reason = veritas.StopReason.NONE
    with tqdm() as pbar:
        # search until all unfair pairs are found
        while stop_reason != veritas.StopReason.NO_MORE_OPEN:
            stop_reason = search.steps(num_search_steps_per_iteration)
            bound_lh = search.current_bounds()
            bounds.append((bound_lh.atleast, bound_lh.top_of_open))
            pbar.update(1)
            pbar.set_description(f"lower {bound_lh.atleast:.3f}, "
                                f"upper {bound_lh.top_of_open:.3f}, "
                                f"#sols {search.num_solutions():<4d}")

            if stop_reason == veritas.StopReason.OUT_OF_MEMORY:
                print(f" (!) veritas ran out of memory ({config.max_memory / 1024**3:.0f} GB): stopping.")
                break
            elif time.time() - t > MAX_TIME:
                print(f"Reached maximum time limit of {MAX_TIME} seconds: stopping.")
                stop_reason = veritas.StopReason.OUT_OF_TIME
                break
            # tmp
            # elif stop_reason == veritas.StopReason.NUM_SOLUTIONS_EXCEEDED:
            #     print("Search stopped: maximum number of solutions exceeded.")
            #     break

    tsearch = time.time() - t
    if verbose:
        print(f"\nSearch status at ending: {stop_reason}")
        print(f"Number of unique solution boxes found: {search.num_solutions()}")
        print(f"Search time: {tsearch:.3f} seconds\n")

    boxes = [search.get_solution(i) for i in range(search.num_solutions())]
    return boxes, featmap, search, stop_reason

def remove_same_class_solutions(d, at, feat_map, solutions, ordinal_cols=[], verbose=False):
    from utils import extract_pairs_batch
    t0 = time.time()
    if not solutions:
        return []

    X = extract_pairs_batch(solutions, feat_map, d.X.columns, ordinal_cols)
    all_preds = at.eval(X)
    all_classes = (all_preds >= 0.0).astype(int)

    unfair_mask = all_classes[0::2] != all_classes[1::2]
    unfair_solutions = [sol for sol, keep in zip(solutions, unfair_mask) if keep]

    if verbose:
        print(f" --> Removed {len(solutions) - len(unfair_solutions)} same-class solutions in {time.time()-t0:.3f}s.")
    return unfair_solutions
# def remove_same_class_solutions(d, at, feat_map, solutions, ordinal_cols=[], verbose=False):
#     """
#     Remove solutions where both instances are predicted to be in the same class.
#     Vectorized for performance.

#     Sampling convention (veritas uses LtSplit: x < threshold):
#       Left branch  (lo_is_unbound): hi is the split threshold; left branch contains x < hi.
#                    → ordinal: hi-1 if hi is exact integer, else floor(hi).
#                    → continuous: hi - 1e-4.
#       Right branch (!lo_is_unbound): lo is the split threshold; right branch contains x >= lo.
#                    → ordinal: ceil(lo) — equals lo for XGBoost integer thresholds (valid since x>=lo);
#                      equals next integer for LightGBM non-integer thresholds (valid since x>lo).
#                    → continuous: lo + 1e-4.
#     """
#     t = time.time()
#     num_sols = len(solutions)
#     if num_sols == 0:
#         return []

#     nb_features = len(feat_map)
#     columns = d.X.columns

#     # Pre-allocate array for all instances (2 instances per solution)
#     all_instances = np.zeros((2 * num_sols, nb_features), dtype=veritas.FloatT)

#     # Speed up: pre-fetch feature IDs for all columns to avoid repeated calls
#     col_feat_ids = []
#     for column in columns:
#         f0 = feat_map.get_feat_id(column, 0)
#         f1 = feat_map.get_feat_id(column, 1)
#         col_feat_ids.append((column, f0, f1, f0 in ordinal_cols))

#     for i, sol in enumerate(solutions):
#         box = sol.box()
#         # print(f"\nProcessing solution {i} / {num_sols} with box: {box}")
#         for instance_idx in (0, 1):
#             row_idx = 2 * i + instance_idx
#             # print(f" Instance {instance_idx} (row {row_idx}):")
#             for column, f0, f1, is_ordinal in col_feat_ids:
#                 # print(f"  Checking column '{column}' with feat_id {f0 if instance_idx == 0 else f1}...")
#                 feat_id = f0 if instance_idx == 0 else f1
#                 if feat_id in box:
#                     # if feat_id in [8, 28]:
#                     #     print(f"   Found interval {box[feat_id]} for feat_id {feat_id}")
#                     interval = box[feat_id]
#                     if interval.lo_is_unbound():
#                         if is_ordinal:
#                             # Left branch is x < hi (LtSplit in veritas).
#                             # If hi is an exact integer, floor(hi)=hi is NOT < hi → use hi-1.
#                             # If hi is non-integer (e.g. LightGBM stores threshold ≈ 1e-35 for a
#                             # {0,1} feature), floor(hi) is the correct largest integer < hi.
#                             hi = interval.hi
#                             value = hi - 1 if hi == np.floor(hi) else np.floor(hi)
#                         else:
#                             value = np.nextafter(interval.hi, -np.inf)  # strictly inside left branch
#                     else:
#                         if is_ordinal:
#                             # ceil(lo) gives the smallest integer >= lo.
#                             # XGBoost right branch is x >= lo, so lo itself is valid → ceil(lo) = lo for integer lo.
#                             # LightGBM right branch is x > lo, but thresholds for binary/ordinal features
#                             # are always non-integer midpoints (e.g. 0.5 for {0,1}), so ceil(lo) > lo. ✓
#                             value = np.ceil(interval.lo)
#                         else:
#                             value = np.nextafter(interval.lo, np.inf)  # strictly inside right branch
#                     # if feat_id in [8, 28]:
#                     #     print(f"   Assigned value {value} to feat_id {feat_id} (column '{column}')")
                        
#                     all_instances[row_idx, f0] = value

#     # Batch prediction
#     # all_preds = at.predict(all_instances)
#     # all_classes = (all_preds >= 0.5).astype(int)
#     # switched to at.eval() for better numerical stability
#     all_preds = at.eval(all_instances)
#     all_classes = (all_preds >= 0.0).astype(int)


#     # # ---- DEBUG: inspect raw scores and sensitive values for first few solutions ----
#     # all_scores = np.asarray(at.eval(all_instances)).flatten()
#     # _sens_cols = [(col, f0, f1) for col, f0, f1, _ in col_feat_ids if f0 != f1]
#     # _debug_n = min(5, num_sols)
#     # print(f"\n[DEBUG remove_same_class] Inspecting first {_debug_n} / {num_sols} solutions:")
#     # for _i in range(_debug_n):
#     #     _s0 = {col: all_instances[2*_i, f0] for col, f0, f1 in _sens_cols}
#     #     _s1 = {col: all_instances[2*_i+1, f0] for col, f0, f1 in _sens_cols}

#     #     # extract the two examples for this solution box for more detailed debugging output,
#     #     # print at.eval scores, predicted classes, and sensitive attribute values for both instances

#     #     ex1, ex2 = contrasting_examples_from_solutions(
#     #         feat_map, solutions[_i], d.X.columns, ordinal_feature_indexes=ordinal_cols  
#     #     ).to_numpy()

#     #     print(f"Solution {_i}:")
#     #     print(f"  Box: {solutions[_i].box()}")
#     #     print(at.get_splits()[8])
#     #     # print(at.get_splits()[28])
#     #     print(f"  Extracted examples from box:")
#     #     print(f"    Instance 0: {ex1}")
#     #     print(f"    Instance 1: {ex2}")
#     #     print(f"at.eval scores: instance 0 = {at.eval(ex1)}, instance 1 = {at.eval(ex2)}")

#     #     # build ex_minus as the copy of whichever of ex1, ex2 has value 0 for the sensitive feature, 
#     #     # but with that feature set to -1 instead of 0
#     #     ex_minus = None
#     #     for col, f0, f1 in _sens_cols:
#     #         if ex1[f0] == 0:
#     #             ex_minus = ex1.copy()
#     #             ex_minus[f0] = -1
#     #             print(f"  Created ex_minus from instance 0 by setting {col} to -1: {ex_minus}")
#     #             break
#     #         elif ex2[f0] == 0:
#     #             ex_minus = ex2.copy()
#     #             ex_minus[f0] = -1
#     #             print(f"  Created ex_minus from instance 1 by setting {col} to -1: {ex_minus}")
#     #             break


#     #     for t, tree in enumerate(at):
#     #         print(f"  Tree {t}:")
#     #         if t==1:
#     #             print(at[t])
#     #         n1 = at[t].eval_node(ex1)
#     #         n2 = at[t].eval_node(ex2)
#     #         n_minus = at[t].eval_node(ex_minus) 
#     #         print(f"    at.eval_node(ex1): {n1}")
#     #         print(f"    at.eval_node(ex2): {n2}")
#     #         print(f"    at.eval_node(ex_minus): {n_minus}")

#     #     print(
#     #         f"  sol[{_i}]: contrast={solutions[_i].output:.4f} | "
#     #         f"sens inst0={_s0} inst1={_s1} | "
#     #         f"raw inst0={all_scores[2*_i]:.4f} inst1={all_scores[2*_i+1]:.4f} | "
#     #         f"class inst0={int(all_classes[2*_i])} inst1={int(all_classes[2*_i+1])}"
#     #     )
#     #     exit()
#     # ---- END DEBUG ----

#     unfair_indices = []
#     for i in range(num_sols):
#         if all_classes[2*i] != all_classes[2*i + 1]:
#             unfair_indices.append(i)

#     unfair_solutions = [solutions[i] for i in unfair_indices]
#     same_class = num_sols - len(unfair_solutions)

#     if verbose:
#         print(f" --> Removed {same_class} solutions with both instances in the same class in {time.time() - t:.3f} seconds.")
#     return unfair_solutions


def check_no_identical_instances(d, feat_map, solutions, ordinal_cols=[]):
    # Solutions with output 0.0 are made by two identical instances.
    # These should already be avoided in check_fairness, 
    # but we still explicitly check...
    for sol in solutions:
        # extract concrete examples from boxes
        two_examples = contrasting_examples_from_solutions(
            feat_map, sol, d.X.columns, ordinal_feature_indexes=ordinal_cols
        )
        instance0 = two_examples.iloc[0, :].to_numpy()
        instance1 = two_examples.iloc[1, :].to_numpy()
        assert not np.allclose(instance0, instance1), \
            f"Found solution with identical instances - output: {sol.output}!"

# (!) If we check fairness across multiple one-hot encoded columns,
#     we need to remove solutions where >1 column is set to true, 
#     as they are semantically wrong (e.g., African-American==1 and Caucasian==1 simultaneously).
def find_active_values(d, group):
    """
    For each column in a mutually exclusive group, find the normalized value
    corresponding to original=1 (the "active" value).

    Strategy: the active values are the unique assignment where no row in d.X
    has two or more columns simultaneously at their active value (mutual exclusivity).
    """
    from itertools import product
    unique_vals = {col: list(d.X[col].unique()) for col in group}
    for candidates in product(*[unique_vals[col] for col in group]):
        candidate_map = dict(zip(group, candidates))
        mutually_exclusive = True
        for i, col_a in enumerate(group):
            for col_b in group[i+1:]:
                if ((d.X[col_a] == candidate_map[col_a]) &
                    (d.X[col_b] == candidate_map[col_b])).any():
                    mutually_exclusive = False
                    break
            if not mutually_exclusive:
                break
        if mutually_exclusive:
            return candidate_map
    raise ValueError(f"Could not determine active values for exclusive group {list(group)}")

def remove_infeasible_solutions(solutions, feat_map, d, exclusive_col_groups, ordinal_cols=[]):
    """
    Remove solutions where any instance has an infeasible combination of
    mutually exclusive column values (e.g., belonging to two races simultaneously).

    exclusive_col_groups: list of lists of mutually exclusive column names,
                          e.g. [['race_African-American', 'race_Caucasian']]
    """
    if not exclusive_col_groups or len(solutions) == 0:
        print(" --> No exclusive column groups provided or no solutions to filter.")
        return solutions

    # For each group, find the active (original=1) normalized value per column
    active_values_per_group = []
    for group in exclusive_col_groups:
        active_vals = find_active_values(d, group)
        unique_vals = {col: sorted(d.X[col].unique()) for col in group}
        active_values_per_group.append((group, active_vals, unique_vals))
        print(f" --> Active (original=1) normalized values for {list(group)}: {active_vals}")

    def snap(value, valid_values):
        return min(valid_values, key=lambda v: abs(value - v))

    valid_solutions = []
    removed = 0
    for sol in solutions:
        two_examples = contrasting_examples_from_solutions(
            feat_map, sol, d.X.columns, ordinal_feature_indexes=ordinal_cols
        )
        is_valid = True
        for group, active_vals, unique_vals in active_values_per_group:
            for instance_idx in range(2):
                num_active = sum(
                    snap(two_examples.iloc[instance_idx][col], unique_vals[col]) == active_vals[col]
                    for col in group
                )
                if num_active > 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            valid_solutions.append(sol)
        else:
            removed += 1

    print(f" --> Removed {removed} solutions with infeasible exclusive column combinations.")
    return valid_solutions


def analyze_best_solution(d, at, feat_map, search, solutions, ordinal_cols=[], verbose=False):
    print("\nMaximum difference between instance0 and instance1")
    if len(solutions) > 0:
        sol = solutions[0]  # best solution so far
        print("- best solution:", sol.output, "->",
            "optimal" if search.is_optimal() else "suboptimal", "solution")
        if verbose:
            print("- feature value ranges:")
            for feat_id, ival in sol.box().items():
                column = feat_map.get_name(feat_id)
                mark = "*" if column == 'male' else " "
                print(f"    {mark}{feat_id:>3d} {column:>25s}", ival)
    
        two_examples = contrasting_examples_from_solutions(
            feat_map, sol, d.X.columns, ordinal_feature_indexes=ordinal_cols
        )
        # predictions = pd.Series((at.predict(two_examples))>0.5, index=["instance 0", "instance 1"])
        if verbose:
            print("\n- contrasting examples:\n")
            print(two_examples.T.round(1))

        predictions = at.predict(two_examples)>0.5
        probabilities = at.predict(two_examples)       # probability of class 1
        raw_scores = at.eval(two_examples)             # pre-sigmoid raw scores
    
        outputs = pd.DataFrame(np.vstack((
                                    raw_scores.flatten(),
                                    100*probabilities.flatten(),
                                    predictions.flatten()
                                    )).T,
                            columns=["raw_score", "probability%", "class",],
                            index=["instance 0", "instance 1"])
    
        # if verbose:
        print("\n-->  model outputs on contrasting examples:\n")
        print(outputs.T.round(1))
        # print(" (!) PROBABILITIES SEEM BUGGED: CHECK (!)")

def check_fixes_hold(at_repaired, solutions, feat_map, d, ordinal_cols):
    """Check that the fixes hold for the original unfair pairs."""
    from utils import extract_pairs_batch
    fixed_count = 0
    for i, sol in enumerate(solutions):
        X_pair = extract_pairs_batch([sol], feat_map, d.X.columns, ordinal_cols)
        # preds = at_repaired.predict(X_pair.astype(veritas.FloatT)).flatten()
        # if (preds[0] >= 0.5) == (preds[1] >= 0.5):
        #     fixed_count += 1
        # else:
        #     print(f" (!) Pair {i} is STILL UNFAIR: {float(preds[0]):.4f} vs {float(preds[1]):.4f}")
        # switched to at.eval() for better numerical stability
        preds = at_repaired.eval(X_pair.astype(veritas.FloatT)).flatten()
        if (preds[0] >= 0.0) == (preds[1] >= 0.0):
            fixed_count += 1
        else:
            print(f" (!) Pair {i} is STILL UNFAIR: {float(preds[0]):.4f} vs {float(preds[1]):.4f}")
    return fixed_count

# from veritas/examples/global_fairness_verification.ipynb
def constrast_two_examples(at, columns, nonfixed_columns, verbose=False):
    """Create a `veritas.AddTree` that contrast two instances.
    
    The new AddTree outputs difference between the original AddTree's outputs
    for instances 0 and 1.

    at: The original veritas.AddTree tree ensemble model
    columns: array with column names
    nonfixed_columns: columns that are allowed to change between the two instances
    """
    feat_map = veritas.FeatMap(columns)
    for column in columns:
        if column not in nonfixed_columns:
            index_for_instance0 = feat_map.get_index(column, 0)
            index_for_instance1 = feat_map.get_index(column, 1)
            feat_map.use_same_id_for(index_for_instance0, index_for_instance1)

    at_for_instance1 = feat_map.transform(at, 1)
    at_for_instance1.set_base_score(0, at.get_base_score(0)) # <--- (!) BUG in feat_map.transform: base score not copied
    at_contrast = at.concat_negated(at_for_instance1)

    if verbose:
        print("  Feature IDs used by instance 0\n   and instance 1 respectively:")
        print("-"*(25+4+4))
        for column in columns:
            mark = "*" if column in nonfixed_columns else ""
            feat_id_instance0 = feat_map.get_feat_id(column, 0)
            feat_id_instance1 = feat_map.get_feat_id(column, 1)
            print(f"{column:25s} {feat_id_instance0:3d} {feat_id_instance1:3d}", mark)
    
    return at_contrast, feat_map

# from veritas/examples/global_fairness_verification.ipynb
def contrasting_examples_from_solutions(
    feat_map, sol, columns, ordinal_feature_indexes=[]
):
    nb_features = len(feat_map)
    two_examples = np.zeros((2, nb_features))
    box = sol.box()

    for instance in (0, 1):
        for column in columns:
            feat_id = feat_map.get_feat_id(column, instance)
            feat_id_untrasformed = feat_map.get_feat_id(column, 0)
            if feat_id in box:
                interval = box[feat_id]
                is_ordinal = feat_id_untrasformed in ordinal_feature_indexes
                if interval.lo_is_unbound():
                    assert not interval.hi_is_unbound()
                    if is_ordinal:
                        hi = interval.hi
                        value = hi - 1 if hi == np.floor(hi) else np.floor(hi)
                    else:
                        value = np.nextafter(interval.hi, -np.inf)  # strictly inside left branch
                else:
                    if is_ordinal:
                        value = np.ceil(interval.lo)
                    else:
                        value = np.nextafter(interval.lo, np.inf)  # strictly inside right branch
                two_examples[instance, feat_id_untrasformed] = value

    return pd.DataFrame(two_examples, columns=columns, index=["instance 0", "instance 1"])

def evaluate_empirical_fairness(at, dtest, sensitive_col):
    """
    Given an AddTree and test set, flip the value of sensitive_col (0/1)
    and check whether the model's binary prediction changes.
    Returns the fraction of samples where prediction is unchanged (fair).
    """
    X_np = dtest.X.to_numpy(dtype=veritas.FloatT)

    def _predict(X_arr):
        # return (at.predict(X_arr).flatten() >= 0.5).astype(int)
        # switched to at.eval() for better numerical stability
        return (at.eval(X_arr).flatten() >= 0.0).astype(int)

    preds_orig = _predict(X_np)

    if isinstance(sensitive_col, list):
        col_indices = [dtest.X.columns.get_loc(c) for c in sensitive_col]
        fair_mask = np.ones(len(dtest.X), dtype=bool)
        for active_col in range(len(col_indices)):
            X_alt = X_np.copy()
            for j, ci in enumerate(col_indices):
                X_alt[:, ci] = 1.0 if j == active_col else 0.0
            orig_encoding = X_np[:, col_indices]
            changed = ~np.all(X_alt[:, col_indices] == orig_encoding, axis=1)
            preds_alt = _predict(X_alt)
            fair_mask &= ~(changed & (preds_orig != preds_alt))
        return fair_mask.mean()
    else:
        col_idx = dtest.X.columns.get_loc(sensitive_col)
        X_flip = X_np.copy()
        X_flip[:, col_idx] = 1.0 - X_flip[:, col_idx]
        scores_orig = at.eval(X_np).flatten()
        scores_flip = at.eval(X_flip).flatten()
        # Ignore examples where both scores are on the decision boundary
        # (small machine precision issues can cause arbitrary flips here)
        eps_score = 1e-10
        on_boundary = (np.abs(scores_orig) <= eps_score) & (np.abs(scores_flip) <= eps_score)
        preds_flip = (scores_flip >= 0.0).astype(int)
        fair_mask = (preds_orig == preds_flip) | on_boundary
        return fair_mask.mean()






# Below: old functions kept for reference (not complete, with unneeded additons, or just buggy)

# def check_fairness(at, d, column, resume_search=None):
#     """Check fairness of `at` wrt `column`.

#     If `resume_search` is provided, the existing veritas search object is
#     continued from where it stopped (e.g. after NUM_SOLUTIONS_EXCEEDED) without
#     rebuilding at_contrast or re-running config setup.  feat_map is recomputed
#     from column structure only (independent of weights), so it is always correct.
#     """
#     if isinstance(column, str):
#         nonfixed_cols = {column}
#     else:
#         nonfixed_cols = set(column)

#     if resume_search is None:
#         for col in nonfixed_cols:
#             print(f"Unique values for attribute '{col}': ", d.X[col].unique())

#         # create AddTree that contrasts two examples, duplicating nonfixed columns
#         # and using the same feature IDs for the fixed columns
#         at_contrast, featmap = constrast_two_examples(at, d.X.columns, nonfixed_cols)

#         # optimize output of at_contrast to find max difference in predictions
#         config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
#         # Note: using config.ignore_state_when_worse_than = 0.0 generates solution boxes
#         #       with output 0.0 where instance0 = instance1 (i.e., useless)
#         #       --> hence I'm using this small eps
#         config.ignore_state_when_worse_than = 1e-6 # >0: instance 0's output > instance 1's output
#         # aggressive veritas search
#         config.focal_eps = 0.95
#         config.max_focal_size = 100
#         # try enumerating ALL violating boxes, not only the one with max output
#         config.stop_when_optimal = False
#         config.stop_when_num_solutions_exceeds = int(1e10)  # no total cap; allows resuming
#         config.stop_when_num_new_solutions_exceeds = 10000   # limit per search call
#         # allocate 128 gb for veritas search
#         config.max_memory = 128 * 1024 * 1024 * 1024

#         # optionally constrain certain feature values
#         # (e.g., look only for cases where instance 0 describes a male person)
#         # prune_box = {featmap.get_feat_id("male", 0): veritas.Interval(0, 0.5)}
#         # prune_box = {feat_map.get_feat_id("age", 0): veritas.Interval(-np.inf, 0)}
#         prune_box = {}
#         search = config.get_search(at_contrast, prune_box)
#     else:
#         # feat_map depends only on column structure, not weights — safe to recompute
#         _, featmap = constrast_two_examples(at, d.X.columns, nonfixed_cols)
#         print(f"  Resuming veritas search from {resume_search.num_solutions()} existing solutions...")
#         search = resume_search

#     # run search
#     t = time.time()
#     max_new_solutions = 10000          # solutions to collect per call (enables resuming)
#     n_sols_at_start = search.num_solutions()
#     stop_reason = veritas.StopReason.NONE
#     with tqdm() as pbar:
#         # search until all unfair pairs are found
#         while stop_reason != veritas.StopReason.NO_MORE_OPEN:
#             stop_reason = search.steps(100)
#             bound_lh = search.current_bounds()
#             pbar.update(1)
#             pbar.set_description(f"lower {bound_lh.atleast:.3f}, "
#                                 f"upper {bound_lh.top_of_open:.3f}, "
#                                 f"#sols {search.num_solutions():<4d}")

#             if time.time() - t > MAX_TIME:
#                 print(f"Reached maximum time limit of {MAX_TIME} seconds: stopping.")
#                 stop_reason = veritas.StopReason.OUT_OF_TIME
#                 break
#             elif search.num_solutions() - n_sols_at_start >= max_new_solutions:
#                 print("Search stopped: maximum number of new solutions exceeded.")
#                 stop_reason = veritas.StopReason.NUM_SOLUTIONS_EXCEEDED
#                 break

#     tsearch = time.time() - t
#     print(f"\nSearch status at ending: {stop_reason}")
#     print(f"Number of unique solution boxes found: {search.num_solutions()}")
#     print(f"Search time: {tsearch:.3f} seconds\n")

#     boxes = [search.get_solution(i) for i in range(search.num_solutions())]
#     return boxes, featmap, search, stop_reason


# def remove_same_class_solutions_old(d, at, feat_map, solutions, ordinal_cols=[], verbose=False):
#     """ Legacy version of solution filtering using a for-loop. """
#     # veritas fairness analysis can also find pairs of instances that are
#     # classified in the same class by the model
#     #  e.g., t(instance0)=0.2, t(instance1)=0.95
#     #  --> these will be returned as composite addtree output is 0.75 hence >0
#     # But these are not interesting for fairness --> we remove them manually
#     t = time.time()
#     unfair_solutions = []
#     same_class = 0
#     for sol in solutions:
#         # extract concrete examples from boxes
#         two_examples = contrasting_examples_from_solutions(
#             feat_map, sol, d.X.columns, ordinal_feature_indexes=ordinal_cols
#         )
#         instance0 = two_examples.iloc[0, :].to_numpy()
#         instance1 = two_examples.iloc[1, :].to_numpy()
#         pred0 = at.predict(instance0.reshape(1, -1))[0]
#         pred1 = at.predict(instance1.reshape(1, -1))[0]
#         class0 = 1 if pred0 > 0.5 else 0
#         class1 = 1 if pred1 > 0.5 else 0

#         if class0 == class1:
#             same_class += 1
#             # print(f" (!) Found solution with both instances in class ({class0})")
#             # print(sol.box())
#         else:
#             unfair_solutions.append(sol)
    
#     if verbose:
#         print(f" --> Removed {same_class} solutions with both instances in the same class in {time.time() - t:.3f} seconds.")
#     return unfair_solutions


# def remove_same_class_solutions_lo_bug(d, at, feat_map, solutions, ordinal_cols=[], verbose=False):
#     """
#     BUGGY: for ordinal right-branch intervals uses np.floor(lo).
#     LightGBM splits use strict > so lo itself lands in the LEFT branch; floor(lo) also goes left.
#     XGBoost was unaffected because it uses >= (inclusive) for right branches, and its thresholds
#     are the actual integer data values, so floor(threshold) == threshold >= threshold → right. ✓
#     """
#     t = time.time()
#     num_sols = len(solutions)
#     if num_sols == 0:
#         return []

#     nb_features = len(feat_map)
#     columns = d.X.columns

#     all_instances = np.zeros((2 * num_sols, nb_features), dtype=veritas.FloatT)

#     col_feat_ids = []
#     for col_idx, column in enumerate(columns):
#         f0 = feat_map.get_feat_id(column, 0)
#         f1 = feat_map.get_feat_id(column, 1)
#         col_feat_ids.append((column, f0, f1, f0 in ordinal_cols))

#     for i, sol in enumerate(solutions):
#         box = sol.box()
#         for instance_idx in (0, 1):
#             row_idx = 2 * i + instance_idx
#             for column, f0, f1, is_ordinal in col_feat_ids:
#                 feat_id = f0 if instance_idx == 0 else f1
#                 if feat_id in box:
#                     interval = box[feat_id]
#                     if interval.lo_is_unbound():
#                         value = interval.hi - 1e-4
#                     else:
#                         value = interval.lo  # BUG: lo is the split threshold (exclusive for LightGBM)
#                     if is_ordinal:
#                         value = np.floor(value)  # BUG: floor(lo) still lands in left branch
#                     all_instances[row_idx, f0] = value

#     all_preds = at.predict(all_instances)
#     all_classes = (all_preds >= 0.5).astype(int)

#     unfair_indices = []
#     for i in range(num_sols):
#         if all_classes[2*i] != all_classes[2*i + 1]:
#             unfair_indices.append(i)

#     unfair_solutions = [solutions[i] for i in unfair_indices]
#     same_class = num_sols - len(unfair_solutions)

#     if verbose:
#         print(f" --> Removed {same_class} solutions with both instances in the same class in {time.time() - t:.3f} seconds.")
#     return unfair_solutions

# def check_fixes_hold(at_repaired, solutions, feat_map, d, ordinal_cols):
#     """Check that the fixes hold for the original unfair pairs."""
#     fixed_count = 0
#     for i, sol in enumerate(solutions):
#         two_examples = contrasting_examples_from_solutions(
#             feat_map, sol, d.X.columns, ordinal_feature_indexes=ordinal_cols
#         )
#         preds = at_repaired.predict(two_examples.to_numpy(dtype=veritas.FloatT))
#         preds = np.asarray(preds).reshape(-1)
#         p0 = float(preds[0])
#         p1 = float(preds[1])

#         if (p0 >= 0.5) == (p1 >= 0.5):
#             fixed_count += 1
#         else:
#             print(f" (!) Pair {i} is STILL UNFAIR: {p0:.4f} vs {p1:.4f}")
    
#     return fixed_count
# def check_fixes_hold(at_repaired, solutions, feat_map, d, ordinal_cols):
#     """Check that the fixes hold for the original unfair pairs.

#     IMPORTANT: The MILP in `leaf_refinement_top.py` enforces fairness in *score space*
#     (linear in the leaf weights). To avoid mismatches between `predict` probabilities
#     and `eval` scores (which depend on AddTree type/calibration), this checker uses
#     `at.eval` with threshold 0.0.
#     """
#     from leaf_refinement_top import extract_pairs_batch
#     fixed_count = 0
#     for i, sol in enumerate(solutions):
#         X_pair = extract_pairs_batch([sol], feat_map, d.X.columns, ordinal_cols)
#         scores = np.asarray(at_repaired.eval(X_pair.astype(veritas.FloatT))).reshape(-1)

#         if (scores[0] >= 0.0) == (scores[1] >= 0.0):
#             fixed_count += 1
#         else:
#             # Also print probabilities for debugging if available.
#             try:
#                 probs = np.asarray(at_repaired.predict(X_pair.astype(veritas.FloatT))).reshape(-1)
#                 print(
#                     f" (!) Pair {i} is STILL UNFAIR: score {float(scores[0]):+.6f} vs {float(scores[1]):+.6f} "
#                     f"(prob {float(probs[0]):.4f} vs {float(probs[1]):.4f})"
#                 )
#             except Exception:
#                 print(
#                     f" (!) Pair {i} is STILL UNFAIR: score {float(scores[0]):+.6f} vs {float(scores[1]):+.6f}"
#                 )
    
#     return fixed_count

# def contrasting_examples_from_solutions_lo_bug(
#     feat_map, sol, columns, ordinal_feature_indexes=[]
# ):
#     nb_features = len(feat_map)
#     two_examples = np.zeros((2, nb_features))
#     box = sol.box()
#
#     for instance in (0, 1):
#         for column in columns:
#             feat_id = feat_map.get_feat_id(column, instance)
#             feat_id_untrasformed = feat_map.get_feat_id(column, 0)
#             if feat_id in box:
#                 interval = box[feat_id]
#                 if interval.lo_is_unbound():
#                     assert not interval.hi_is_unbound()
#                     value = interval.hi - 1e-4  # not inclusive
#                 else:
#                     value = interval.lo  # BUG: lo is split threshold (exclusive for LightGBM)
#                 if feat_id_untrasformed in ordinal_feature_indexes:  # ordinal feature
#                     value = np.floor(value)  # BUG: floor(lo) still lands in left branch
#                 two_examples[instance, feat_id_untrasformed] = value
#
#     return pd.DataFrame(two_examples, columns=columns, index=["instance 0", "instance 1"])