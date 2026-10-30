import json
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import veritas
from pathlib import Path


# Fix the store to the project root so paths are absolute and identical no matter which directory the process is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(PROJECT_ROOT, "store", "neurips")

def _first_existing(*candidates):
    """Return the first path that exists (or the first candidate if none does)."""
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0]

# External tools/benchmarks: env var override > cluster path > local checkout.
PLAJA_DIR = os.environ.get("PLAJA_BIN") or _first_existing(
    f"{PROJECT_ROOT}/third_party/plaja/build/PlaJA",
    "/home/jain/plaja/build/PlaJA",
    "/home/chaahat/Desktop/PhD_projects/code/plaja/build/PlaJA",
)
BENCH_DIR = os.environ.get("PLAJA_BENCH_DIR") or _first_existing(
    f"{PROJECT_ROOT}/benchmarks",
    "/mnt/shared/benchmarks",
    "/home/jain/PlaJABenchmarks",
    "/home/chaahat/Desktop/PhD_projects/code/PlaJABenchmarks",
)
BENCHMARK_DIR = _first_existing(f"{BENCH_DIR}/jani_models_and_teachers", f"{BENCH_DIR}/benchmarks")
DATA_DIR = _first_existing(
    f"{BENCH_DIR}/training_data",
    f"{BENCH_DIR}/benchmarks_archive/icaps26_fault_fixing/training_data",
)
# Locally generated data (e.g. store/neurips/data) is searched as a fallback.
LOCAL_DATA_DIR = f"{STORE_DIR}/data"

# (!) must come after BENCHMARK_DIR: path_scripts imports it back from this module
import utils.path_scripts as NEURIPS26


IMITATION_DIR = f"{STORE_DIR}/imitation-learning"
MODEL_DIR = f"{STORE_DIR}/models"
ABLATION_DIR = f"{STORE_DIR}/ablation"
FAULT_DIR = f"{STORE_DIR}/faults"
SAFE_DIR = f"{STORE_DIR}/safe_states"
VERIF_DIR = f"{STORE_DIR}/verification"
RESULT_DIR = f"{STORE_DIR}/results"
BENCHMARKS_VERSION = "neurips26" 
CACHED_FAULTS_DIR = f"{FAULT_DIR}"
CACHED_ENSEMBLE_DIR = f"{BENCH_DIR}/initial_ensembles"

FA_TOTAL_TIME = 60*60*12  # 12 hours
FA_TIME_PER_QUERY = 1800 # 30 minutes
VERIFICATION_TIME = 60*60*24 # 24 hours

NFOLDS = 5
SEED = 10
TEST_SIZE = 0.2

np.random.seed(SEED)

###### Loading training data for tree ensembles ###############
def load_dataset_for_ablation(benchmark, prefix):
    # check how many columns to keep reading interface file
    num_inputs = filter_data_columns(benchmark, prefix)

    # construct correct path to data 
    game = get_game(benchmark)
    problem = get_problem(benchmark)
    # p = f"/{prefix}" if prefix is not None else ""
    p = f"/{prefix}" if prefix else ""
    # data_dir = f"{data_dir}/{game}/{prefix}/{problem}.json"
    data_dir = get_data_file(benchmark, prefix)
    assert os.path.exists(data_dir)
    with open(data_dir, "r") as file:
        print("\n\tLoading data from ", data_dir)
        data = json.load(file)['sample']

        # always drop first column
        # keep following num_inputs columns and drop remaining ones (if any)
        states = [entry["state"][1:1+num_inputs] for entry in data]

        labels = [entry["label"] for entry in data]
        data = {"states": states, "labels": labels}

    df = pd.DataFrame(data['states']).astype(np.float32).astype(veritas.FloatT)
    y = pd.DataFrame(data['labels'])
    print("\n\tClass distribution in the dataset: ", \
        y.value_counts(normalize=True).to_dict())

    # extract predicted class in case we're dealing with a Regressor
    if y.shape[1] > 1:
        print("\n\tNote: converting raw scores to class labels.\n")
        # y_scores = y.values
        y = y.values.argmax(axis=1)
    return df, y


############### Helper functions to parse benchmarks ##############################

def get_benchmark_name(benchmark):
    # keep only benchmark name (remove all digits)
    return '_'.join(chunk for chunk in benchmark.split("_") if not chunk[0].isdigit())

def remove_nn_size(benchmark):
    # remove only NN size from benchmark name
    if re.search(r"_\d+_\d+$", benchmark):  
        benchmark = '_'.join(benchmark.split("_")[:-2])
    return benchmark

def filter_data_columns(benchmark, prefix):
    interface_file = get_interface_file(prefix, benchmark)
    with open(interface_file, 'r') as f:
        data = json.load(f)

    num_inputs = len(data.get("input", []))
    return num_inputs

def get_num_outputs(benchmark, prefix):
    # Get the number of outputs for a benchmark
    interface_file = get_interface_file(prefix, benchmark)
    with open(interface_file, 'r') as f:
        data = json.load(f)

    num_outputs = len(data.get("output", []))
    return num_outputs



########################## Policy Performance Parsers ##############################
def evaluate_ensemble_performance(prefix, benchmark, repair_method, tconfig, iteration,
                                   fconfig, uf_approx=False, penalty=False):
    """
    Wrapper function to evaluate both goal fraction and bug fraction.
    """
    from utils.plaja_interface import simulate_ensemble_for_policy_performance
    # _, train_stats, _ = \
    #     simulate_ensemble_for_policy_performance(prefix, benchmark, tconfig, iteration, fconfig,
    #                                    penalty=penalty)
    train_stats, _ = \
        simulate_ensemble_for_policy_performance(prefix, benchmark, repair_method, tconfig, iteration, 
                                                    fconfig, penalty=penalty, approx=uf_approx)
    
    uf_train_exh = None   # envelope (exhaustive)
    gf_train_app = uf_train_app = None   # some_paths (approx)

    # uf_train = False
    for tstats in train_stats:
        assert isinstance(tstats, str), f"Expected CSV path, got {type(tstats)} with value {tstats}"
        if tstats.endswith("envelope.csv"):
            __, uf_train_exh = parse_performance(tstats)
        elif tstats.endswith("some_paths.csv"):
            gf_train_app, uf_train_app = parse_performance(tstats)

    # keep exhaustive bug fraction if available, otherwise use approximate one
    # prefer exhaustive unsafe% - fall back to approximate if it went oom
    if uf_train_exh is not None:
        gf_train, uf_train = gf_train_app, uf_train_exh
    else:
        # print("\t [NOTE] %Unsafe for training is approximate as full envelope evaluation went OOR.")
        uf_approx = True
        gf_train, uf_train = gf_train_app, uf_train_app
    # gf_train, uf_train = parse_performance(train_stats)

    type_string = "Approximate" if uf_approx else "Exhaustive"
    print(f"\t- {type_string} performance on TRAINING start states:  ")
    print(f"\t\t--> %Goal: {gf_train}%")
    print(f"\t\t--> %Unsafe: {uf_train}%")
    
    try:
        fixed_state_property_file = get_fixed_property_file(prefix, benchmark, fconfig['num_starts'])
        if (count_start_states(fixed_state_property_file) >= fconfig['num_starts']) and not fconfig['enumerable']:
            # _, eval_stats, _ = \
            #     simulate_ensemble_for_policy_performance(prefix, benchmark, tconfig, iteration, fconfig, 
            #                                 evaluation_set=True, penalty=penalty)
            # gf_eval, uf_eval = parse_performance(eval_stats)
            # do the same thing as for train
            eval_stats, _ = \
                simulate_ensemble_for_policy_performance(prefix, benchmark, repair_method, tconfig, 
                                            iteration, fconfig, evaluation_set=True, penalty=penalty,
                                            approx=uf_approx)
            
            uf_eval_exh = None
            gf_eval_app = uf_eval_app = None

            for estats in eval_stats:
                assert isinstance(estats, str), f"Expected CSV path, got {type(estats)} with value {estats}"
                if estats.endswith("envelope.csv"):
                    __, uf_eval_exh = parse_performance(estats)
                elif estats.endswith("some_paths.csv"):
                    gf_eval_app, uf_eval_app = parse_performance(estats)
            
            if uf_eval_exh is not None:
                # print("\t [NOTE] %Unsafe for eval. is approximate as full envelope evaluation went OOR.")
                gf_eval, uf_eval = gf_eval_app, uf_eval_exh
            else:
                uf_approx = True
                gf_eval, uf_eval = gf_eval_app, uf_eval_app

            print(f"\t- {type_string} performance on EVAL start states:  ")
            print(f"\t\t--> %Goal: {gf_eval}%")
            print(f"\t\t--> %Unsafe: {uf_eval}%")

            # check that we have sets of start states disjoint between train and eval for some_paths
            approx_train_stats = [s for s in train_stats if s.endswith("some_paths.csv")][0] 
            approx_eval_stats = [s for s in eval_stats if s.endswith("some_paths.csv")][0]
            # assert they are disjoint
            assert_disjoint_start_states(approx_train_stats, approx_eval_stats)

        else:
            # assert False, count_start_states(fixed_state_property_file)
            # for beluga and blocksword we use the same start states for training and evaluation
            gf_eval, uf_eval = gf_train, uf_train
    except:
        # if no property file, we use the same start states for training and evaluation
        gf_eval, uf_eval = gf_train, uf_train

    # if oom_train or oom_eval:
    #     print(f"\t WARNING: Model evaluation OOR.")
    
    # fixed_state_property_file = get_fixed_property_file(prefix, benchmark, fconfig['num_starts'])
    # if (count_start_states(fixed_state_property_file) >= fconfig['num_starts']):
    #     # Assert that training and evaluation states are disjoint
        # assert_disjoint_start_states(train_stats, eval_stats)
    return (gf_train, uf_train, gf_eval, uf_eval), uf_approx

def parse_performance(stats_dir):
    stats = pd.read_csv(stats_dir, low_memory=False)
    last_row = stats.iloc[-1] 

    # if stats_dir.endswith("some_paths.csv"):
    #     print("\tUsing approximate performance simulation ('some_paths'):")
    gf, uf, tp = last_row['DoneGoal'], last_row['DoneAvoid'], last_row['PathsChecked'] 
    rest = np.round((tp - gf - uf)/tp*100, 2)
    gf = np.round(gf/tp*100, 2)
    uf = np.round(uf/tp*100, 2)
    # print(f"\t\t--> at: goal fraction {gf}%")
    # print(f"\t\t--> at: unsafe fraction {uf}%")
    # print(f"\t\t--> at: cycles/terminal states {rest}%")

    # elif stats_dir.endswith("envelope.csv"):
    #     print("\tUsing exhaustive performance simulation ('envelope'):")
    #     gf, uf = last_row['DoneGoal'], last_row['DoneAvoid'] 
    #     total = last_row['Episodes']
    #     gf = np.round(gf/total*100, 2)
    #     uf = np.round(uf/total*100, 2)
    #     print(f"\t\t--> at: goal fraction {gf}%")
    #     print(f"\t\t--> at: unsafe fraction {uf}%")
    return gf, uf

def assert_disjoint_start_states(filepath1, filepath2):
    df1 = pd.read_csv(filepath1, low_memory=False)
    df2 = pd.read_csv(filepath2, low_memory=False)

    # Extract non-empty unique states
    states1 = set(df1["StateChecked"].dropna().unique())
    states2 = set(df2["StateChecked"].dropna().unique())
    # Assert disjointness
    assert states1.isdisjoint(states2), f"Overlap found: in states"




############################### FAULT FIXING HELPER FUNCTIONS ###############################
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def build_fault_keys(fx: pd.DataFrame, fy: pd.Series) -> List[Tuple[Tuple[float, ...], int]]:
    """
    Build hashable keys for faults as:
        (tuple(state_values), int(forbidden_label))
    Returns a list of keys aligned to current Fx/Fy order.
    """
    fx_np = fx.to_numpy()                  # (N_faults, D)
    fy_np = np.asarray(fy).reshape(-1)     # (N_faults,)
    keys: List[Tuple[Tuple[float, ...], int]] = [
        (tuple(map(float, fx_np[i].tolist())), int(fy_np[i]))
        for i in range(fx_np.shape[0])
    ]
    return keys

def violators_to_mask(
    fx: pd.DataFrame, fy: pd.Series,
    prev_viol_map: Dict[Tuple[Tuple[float, ...], int], bool]
) -> np.ndarray:
    """
    Construct a boolean mask (len == len(fx)) where True means 'upweight this fault row',
    based on the previous iteration's violators map.
    New faults (not in the map) default to False.
    """
    keys = build_fault_keys(fx, fy)
    return np.array([bool(prev_viol_map.get(k, False)) for k in keys], dtype=bool)

def mask_to_violators_map(
    fx: pd.DataFrame, fy: pd.Series, viol_mask: np.ndarray
) -> Dict[Tuple[Tuple[float, ...], int], bool]:
    """
    Convert the current-iteration violators mask (aligned to Fx/Fy order)
    into a dict keyed by (tuple(state), int(forbidden_label)).
    """
    keys = build_fault_keys(fx, fy)
    assert len(keys) == len(viol_mask), "viol_mask length must match number of faults"
    return {k: bool(m) for k, m in zip(keys, viol_mask)}


# utils.py
# from typing import List, Tuple, Dict, Any, Optional
# import numpy as np
# import pandas as pd
# import torch
# import veritas  # for veritas.FloatT dtype

# def prepare_faults_with_nn_labels(
#     Fx: pd.DataFrame,       # fault states (may contain duplicates)
#     Fy: pd.Series,          # forbidden labels (aligned to Fx)
#     Aa: List[List[int]],    # Applicable actions per fault (aligned to Fx)
#     nn_model,               # nnet.NNet instance 
#     num_actions: int,       # number of discrete actions (from interface)
#     fault_weight_multipliers: Optional[np.ndarray] = None,  # aligned to ORIGINAL Fx (same length)
#     ) -> Tuple[
#         pd.Series,             # Cy_full: one label per ORIGINAL Fx (duplicates share same label)
#         pd.DataFrame,          # Fx_unique: deduplicated states
#         pd.Series,             # Cy_unique: deduplicated corrected labels
#         List[List[int]],       # Aa_unique: updated Applicable per unique state
#         List[List[int]],       # Forbid_unique: aggregated forbidden per unique state
#         Optional[np.ndarray],  # w_unique: MAX merged weights per unique state or None
#     ]:
#         """
#         Preparation of NN-corrected labels for augmented retraining.

#         Behavior:
#         1) Group rows by unique fault state.
#         2) Aggregate Applicable actions per state.
#         3) Aggregate all forbidden actions per state (union).
#         4) For each unique state, NN picks best action among ALLOWED = Applicable \ Forbidden.
#         5) Generate:
#             - Cy_full: corrected labels aligned 1:1 with ORIGINAL Fx (duplicates share same label).
#             - Deduplicated training inputs: (Fx_unique, Cy_unique) + Aa_unique + Forbid_unique
#             - w_unique: MAX of fault_weight_multipliers over duplicates of the same state (if provided).

#         Notes:
#         - If a state ends up with no allowed action, a ValueError is raised.
#         - Dict insertion order preserves first occurrence order for deduplicated outputs.

#         Returns:
#         (Cy_full, Fx_unique, Cy_unique, Aa_unique, Forbid_unique, w_unique)
#         """

#         # --- 1) Aggregate per unique state ---
#         # We keep indices to compress weights with MAX later.
#         bucket: Dict[Tuple[float, ...], Dict[str, Any]] = {}
#         for idx, (x, fy, actions) in enumerate(zip(Fx.values, Fy.values, Aa)):
#             key = tuple(map(float, x))
#             if key not in bucket:
#                 bucket[key] = {
#                     "state": x,
#                     "app": set(map(int, actions)),       # start with current Aa
#                     "forbid": {int(fy)},                 # start with current forbidden
#                     "indices": [idx],                    # original positions
#                 }
#             else:
#                 # applicable actions must be the same across duplicates
#                 # so just check
#                 if bucket[key]["app"] != set(map(int, actions)):
#                     raise ValueError(
#                         f"Conflicting Applicable actions for state {x}: "
#                         f"existing={sorted(bucket[key]['app'])}, new={sorted(actions)}"
#                     )
#                 bucket[key]["forbid"].add(int(fy))
#                 bucket[key]["indices"].append(idx)

#         # --- 2) NN-corrected label per unique state (avoid ALL forbidden) ---
#         key2label: Dict[Tuple[float, ...], int] = {}
#         for key, info in bucket.items():
#             x           = info["state"]
#             app_agg     = sorted(info["app"])
#             forbid_agg  = set(info["forbid"])

#             # Allowed = Applicable \ Forbidden
#             allowed = [a for a in app_agg if a not in forbid_agg]
#             if len(allowed) == 0:
#                 raise ValueError(
#                     f"No safe allowed action for state {x}. "
#                     f"Applicable={app_agg}, Forbidden={sorted(forbid_agg)}"
#                 )

#             with torch.no_grad():
#                 preds = nn_model.evaluate_network(torch.FloatTensor(x))  # shape: (num_actions,)

#             # Mask everything not in 'allowed'
#             mask = torch.zeros(num_actions, dtype=torch.bool)
#             for a in allowed:
#                 if 0 <= a < num_actions:
#                     mask[a] = True
#             preds[~mask] = -float("inf")

#             label = int(torch.argmax(preds).item())
#             # Double-safety: must be allowed and NOT forbidden
#             if (label not in allowed) or (label in forbid_agg):
#                 raise ValueError(
#                     f"NN selected a disallowed/forbidden action {label} for state {x} "
#                     f"(allowed={allowed}, forbidden={sorted(forbid_agg)})."
#                 )

#             key2label[key] = label

#         # --- 3) Build Cy_full (aligned to ORIGINAL Fx order) ---
#         cy_rows: List[int] = []
#         for x in Fx.values:
#             key = tuple(map(float, x))
#             cy_rows.append(key2label[key])
#         Cy_full = pd.Series(cy_rows, dtype=int)

#         # --- 4) Deduplicate for training + MAX merged weights ---
#         Fx_unique_rows: List[np.ndarray] = []
#         Cy_unique_rows: List[int]        = []
#         Aa_unique: List[List[int]]       = []
#         Forbid_unique: List[List[int]]   = []
#         w_unique: Optional[List[float]]  = [] if fault_weight_multipliers is not None else None

#         for key, info in bucket.items():
#             Fx_unique_rows.append(info["state"])
#             Cy_unique_rows.append(key2label[key])
#             Aa_unique.append(sorted(info["app"]))
#             Forbid_unique.append(sorted(info["forbid"]))

#             if w_unique is not None:
#                 # MAX weight across duplicates of the same state (your request)
#                 idxs = info["indices"]
#                 max_w = float(np.max(fault_weight_multipliers[idxs])) if len(idxs) > 0 else 1.0
#                 w_unique.append(max_w)

#         Fx_unique = (
#             pd.DataFrame(Fx_unique_rows)
#             .astype(np.float32)
#             .astype(veritas.FloatT)
#             .reset_index(drop=True)
#         )
#         Cy_unique = pd.Series(Cy_unique_rows, dtype=int)
#         if w_unique is not None:
#             w_unique = np.asarray(w_unique, dtype=float)

#         return Cy_full, Fx_unique, Cy_unique, Aa_unique, Forbid_unique, w_unique


################################
# TODO update
# def write_results(res_file, benchmark, prefix, tree_config, fa_config, 
#                   perf_nn, num_faults, num_feasibility_trees, res, perf_orig, perf_mid,
#                   perf_final, perf_final_baseline, vres, runtime, repair_info):
#     import random
#     time.sleep(1 + int(10 * random.random()))  # to avoid concurrent accesses running in parallel

#     with open(res_file, "a") as f:
#         # model_type = tree_config['model_type']
#         # max_depth = tree_config['max_depth']
#         # num_trees = tree_config['num_trees']
#         # lr = tree_config['lr']
#         # compact = fa_config['compact']
#         # num_starts = fa_config['num_starts']
#         fa_total_time = fa_config['fa_total_time']
#         fa_tq = fa_config['fa_time_per_query']
#         rv = fa_config['run_verification']
#         vtt = fa_config['verification_total_time']

#         p = f"_{prefix}" if prefix else ""
#         benchmark_string = f"{benchmark}{p}"
#         # model_string = f"{model_type}_{max_depth}_{num_trees}_{lr}"

#         # plaja_string = f"{num_starts}_{fa_total_time}_{fa_tq}_{compact}_{rv}_{vtt}"
#         plaja_string = f"{fa_total_time}_{fa_tq}_{vtt}"   
#         #f.write(f"\n{benchmark_string} {model_string} {plaja_string} ")
#         f.write(f"\n{benchmark_string} {plaja_string} ")
#         f.write(f"{'_'.join(map(str, perf_nn))} ")
#         f.write(f"{num_faults} ")
#         f.write(f"{num_feasibility_trees} ")
#         f.write(f"{'_'.join(map(str, res))} ")
#         f.write(f"{'_'.join(map(str, perf_orig))} ")
#         f.write(f"{'_'.join(map(str, perf_mid))} ")
#         f.write(f"{'_'.join(map(str, perf_final))} ")
#         f.write(f"{'_'.join(map(str, perf_final_baseline))} ")
#         if isinstance(runtime, int):
#             f.write(f"{runtime} ")
#         else:
#             # f.write(f"{runtime['tot']}_{runtime['gf_bf_o']}_{runtime['fa']}_{runtime['mr']}_{runtime['gur']}_{runtime['gf_bf_f']}_{runtime['pen_bl']}_{runtime['gf_bf_b']}_{runtime['ver']} ")
#             f.write(f"{runtime['tot']}_{runtime['gf_bf_o']}_{runtime['fa']}_{runtime['mr']}_{runtime['gur']}_{runtime['gf_bf_mid']}_{runtime['gf_bf_f']}_{runtime['pen_bl']}_{runtime['gf_bf_b']}_{runtime['ver']} ")
#         f.write(f"{repair_info['#inner_it']}_{repair_info['#faults_fixed']} ")
#         if vres is not None:
#             f.write(f"{'_'.join(map(str, vres))}")


############################# HELPER FUNCTIONS TO GET BENCHMARK FILES ##############################
def get_problem(benchmark):
    """
    Get the specific problem instance we are looking at. Gives hints to the number of objects in the domain.
    """
    return '_'.join(benchmark.split("_")[:-2]) or benchmark

def get_game(benchmark):
    """
    Each benchmark instance is part of a Planning Task. Get this task. 
    """
    p_to_g = {
        "blocksworld": "blocksworld",
        "sum_linetrack": "transport_feature",
        "mod_linetrack": "mod_transport",
        "race_linetrack": "race_transport",
        "linetrack": "transport",
        "beluga": "beluga",
        "npuzzle_3_3": "npuzzle",
        "bouncing_ball": "bouncing_ball",
        "stopping_car": "stopping_car",
        "cart_pole": "cart_pole",
        "inverted_pendulum": "inverted_pendulum",
        "flight_diversion": "flight_diversion",
        "one_way_line": "one_way_line",
        "two_way_line": "two_way_line"
    }
    for k in p_to_g.keys():
        if k in benchmark:
            return p_to_g[k]
    assert False, "Cannot find game name for benchmark: " + benchmark


def get_model_file(prefix, benchmark):
    """
    Get the model file for this benchmark. This specifies the state space as an Markov Decision Process or a Labelled Transition System.
    """
    if BENCHMARKS_VERSION == "neurips26":
        return NEURIPS26.get_model_file(prefix, benchmark)

def get_fixed_property_file(prefix, benchmark, num_starts):
    """
    Load a fixed set of start states generated from the generator scripts inside the benchmarks directory. Good for aligning experiments over multiple clusters.
    """
    if BENCHMARKS_VERSION == "neurips26":
        return NEURIPS26.get_fixed_property_file(prefix, benchmark, num_starts)

def get_compact_property_file(prefix, benchmark):
    if BENCHMARKS_VERSION == "neurips26":
        return NEURIPS26.get_compact_property_file(prefix, benchmark)

def get_property_file(prefix, benchmark, compact, num_starts=0):
    """
    Load the correct property file based on whether it is compact or fixed.
    """
    if compact:
        return get_compact_property_file(prefix, benchmark)
    else:
        return get_fixed_property_file(prefix, benchmark, num_starts)

def get_interface_file(prefix, benchmark):
    """
    An interface file specifies which state variable corresponds to the input features of the tree ensemble. It also specifies the ordering of the action labels.
    """
    if BENCHMARKS_VERSION == "neurips26":
        return NEURIPS26.get_interface_file(prefix, benchmark)

def get_neural_network_file(prefix, benchmark):
    """
    An interface file specifies which state variable corresponds to the input features of the tree ensemble. It also specifies the ordering of the action labels.
    """
    interface_file = get_interface_file(prefix, benchmark)
    nn_file = Path(interface_file).with_suffix('.nnet')
    # if not os.path.exists(nn_file):
    #     raise FileNotFoundError(f"Network file {nn_file} does not exist.")
    return nn_file

def get_ensemble_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0, penalty=False):
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    penalty_string = "_penalty" if penalty else ""

    if iteration == 0:
        if prefix: 
            return f"{CACHED_ENSEMBLE_DIR}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it0.json"
        else:
            return f"{CACHED_ENSEMBLE_DIR}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it0.json"
    # ensemble_file = f"{MODEL_DIR}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.json"
    if prefix:
        ensemble_file = f"{MODEL_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{penalty_string}.json"
    else:
        ensemble_file = f"{MODEL_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{penalty_string}.json"
    # print(f"    --> using ensemble at file path: {ensemble_file}")
    return ensemble_file

def get_ablation_ensemble_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, fault_frac = 0, penalty=False):
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    penalty_string = "_penalty" if penalty else ""
    # ensemble_file = f"{MODEL_DIR}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.json"
    # create directory if it does not exist
    if not os.path.exists(f"{ABLATION_DIR}/{repair_method}/{problem}"):
        os.makedirs(f"{ABLATION_DIR}/{repair_method}/{problem}", exist_ok=True)
    if prefix:
        ensemble_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_ff{fault_frac}{penalty_string}.json"
    else:
        ensemble_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_ff{fault_frac}{penalty_string}.json"
    return ensemble_file

def log_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0, 
             shield=False, extensive_fuzzing=False):
    """
    Storing fault analysis logs as text files. Can be eventually incorporated into the stats file if needed.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    _shield = f"_shielded" if shield else ""
    # if extensive fuzzing, use ad-hoc subdir 
    # (same for every repair method, as for now we only do it at iteration 0)
    repair_method = "extensive_fuzzing" if extensive_fuzzing else repair_method 
    if prefix:
        log_file = f"{FAULT_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{_shield}.txt"
    else:
        log_file = f"{FAULT_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{_shield}.txt"
    
    # print(f"    --> using fault analysis log at file path: {log_file}")
    # exit()
    return log_file

def stat_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0, 
              shield=False, extensive_fuzzing=False):
    """
    Storing fault analysis logs as stats file. Currently only being used to identify goalFraction in a particular run.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    _shield = f"_shielded" if shield else ""
    # if extensive fuzzing, use ad-hoc subdir 
    # (same for every repair method, as for now we only do it at iteration 0)
    repair_method = "extensive_fuzzing" if extensive_fuzzing else repair_method
    if prefix:
        stats_file = f"{FAULT_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{_shield}.csv"
    else:
        stats_file = f"{FAULT_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}{_shield}.csv"
    return stats_file


def verification_log_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0):
    """
    Storing fault analysis logs as text files. Can be eventually incorporated into the stats file if needed.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    if prefix:
        log_file = f"{VERIF_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.txt"
    else:
        log_file = f"{VERIF_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.txt"
    return log_file

def verification_stat_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0):
    """
    Storing fault analysis logs as stats file. Currently only being used to identify goalFraction in a particular run.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    if prefix:
        stats_file = f"{VERIF_DIR}/{repair_method}/{problem}/{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.csv"
    else:
        stats_file = f"{VERIF_DIR}/{repair_method}/{problem}/{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}.csv"
    return stats_file

def _data_dir(root, benchmark, prefix):
    parts = [root, get_game(benchmark)] + ([prefix] if prefix else [])
    return os.path.join(*parts)


def get_data_file(benchmark, prefix, only_goal_data=False, repair_method=None, iteration = -1):
    """
    Canonical filepath for <state, action> pairs from a NN teacher, faithful to the
    benchmark archive layout:  {DATA_DIR}/{game}/{prefix}/{only_goal_data_}{benchmark}.json
    (repair_method is accepted for backwards compatibility; the archive files are
    method-independent). Per-iteration files (augment baseline) are generated
    artifacts and live under LOCAL_DATA_DIR.
    """
    goal_prefix = "only_goal_data_" if only_goal_data else ""
    if iteration == -1:
        return os.path.join(_data_dir(DATA_DIR, benchmark, prefix), f"{goal_prefix}{benchmark}.json")
    return os.path.join(_data_dir(LOCAL_DATA_DIR, benchmark, prefix), f"{benchmark}_{iteration}.json")


def find_data_file(benchmark, prefix, only_goal_data=False, repair_method=None):
    """Find shared teacher data in the archive or local cache (.json or .jsonl).

    The repair method is ignored, including for legacy callers. All methods use
    the same benchmark/prefix/data-selection file. If absent, return the canonical
    archive path for generation.
    """
    canonical = get_data_file(benchmark, prefix, only_goal_data)
    local_dir = _data_dir(LOCAL_DATA_DIR, benchmark, prefix)
    goal_prefix = "only_goal_data_" if only_goal_data else ""
    candidates = [
        os.path.join(directory, f"{goal_prefix}{benchmark}.{ext}")
        for directory in (os.path.dirname(canonical), local_dir)
        for ext in ("json", "jsonl")
    ]
    return _first_existing(*candidates)



def nn_stat_file(benchmark, prefix, only_goal_data):
    """
    NN performance stats (archive layout): {DATA_DIR}/{game}/{prefix}/{only_goal_data_}{benchmark}.csv
    """
    goal_prefix = "only_goal_data_" if only_goal_data else ""
    return os.path.join(_data_dir(DATA_DIR, benchmark, prefix), f"{goal_prefix}{benchmark}.csv")


def find_stat_file(benchmark, prefix, only_goal_data):
    """Existing NN stats file: archive first, then LOCAL_DATA_DIR; archive path if none."""
    goal_prefix = "only_goal_data_" if only_goal_data else ""
    return _first_existing(
        nn_stat_file(benchmark, prefix, only_goal_data),
        os.path.join(_data_dir(LOCAL_DATA_DIR, benchmark, prefix), f"{goal_prefix}{benchmark}.csv"),
    )


def perf_log_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0, evaluation = False, ablation = False, eval_method="some_paths"):
    """
    Storing logs for running the tree ensemble policy for goal fraction.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    ev = "eval" if evaluation else "train"
    if ablation:
        if prefix:
            log_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_ff{iteration}_{eval_method}.txt"
        else:
            log_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{depth}depth_{num_tree}trees{_lr}_ff{iteration}_{eval_method}.txt"
        return log_file
    if prefix:
        log_file = f"{FAULT_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.txt"
    else:
        log_file = f"{FAULT_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.txt"
    return log_file

def perf_stat_file(benchmark, prefix, repair_method, model_type, depth, num_tree, lr = 0, iteration = 0, evaluation=False, ablation = False, eval_method="some_paths"):
    """
    Currently being used to identify fraction of policy runs that reach goal or unsafety (DoneGoal/DoneAvoid/PathsChecked) in a particular run.
    """
    problem = get_problem(benchmark)
    _lr = f"_{lr}LR" if model_type=='xgb' else ""
    ev = "eval" if evaluation else "train"
    if ablation:
        if prefix:
            stats_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.csv"
        else:
            stats_file = f"{ABLATION_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.csv"
        return stats_file
    if prefix:
        stats_file = f"{FAULT_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{prefix}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.csv"
    else:
        stats_file = f"{FAULT_DIR}/{repair_method}/{problem}/{ev}_perf_{benchmark}_{depth}depth_{num_tree}trees{_lr}_it{iteration}_{eval_method}.csv"
    return stats_file


import json 
def find_key_in_structure(data, target_key):
    """Recursively search JANI file for specific key"""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            result = find_key_in_structure(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_in_structure(item, target_key)
            if result is not None:
                return result
    return None
def count_start_states(file_path):
    try:
        # Read and parse the JANI file
        with open(file_path, 'r') as file:
            jani_data = json.load(file)
        # Look for properties that contain "start"
        if 'properties' in jani_data:
            for property in jani_data['properties']:
                if 'expression' in property:
                    expression = property['expression']
                    if 'start' in expression:
                        start = expression['start']
                        # Recursively search for "start" in the expression
                        start_info = find_key_in_structure(expression, 'values')
                        if start_info:
                            # print("Found 'start' in property expression:")
                            # print(json.dumps(start_info, indent=2))
                            return len(start_info)
        
        print("No 'start' found in properties")
        assert(False)
        return None
        
    except Exception as e:
        print(f"Error reading JANI file: {e}")
        return None
