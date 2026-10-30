import os
import numpy as np
import time
from paths import PRADA_DATA_DIR
os.environ.setdefault("PRADA_DATA_DIR", PRADA_DATA_DIR)   # read by prada at import
import prada
import veritas


# ---------------------------   1. Constants   ----------------------------

SEED = 6
NFOLDS = 5
NJOBS = 5           # parallel jobs for faster model training
MAX_IT = 100        # max iterations for repair loop

from paths import CACHE_DIR, FATT_DIR
MODEL_DIR = f"{CACHE_DIR}/models"

# FATT baseline (Ranzato & Zanella, 2021)
FATT_BIN = f"{FATT_DIR}/bin/meta-silvae"
SILVA_BIN = f"{FATT_DIR}/bin/silva"
FATT_CACHE = f"{CACHE_DIR}/fatt"

# Create cache dirs
for _d in (MODEL_DIR, f"{CACHE_DIR}/results", FATT_CACHE):
    os.makedirs(_d, exist_ok=True)

SENSITIVE_COLS = {
    "CompasTwoYears":           "sex", 
    "DutchCensus":              "male",
    "BankMarketingFull":        "married", 
    "LawSchool":                "male", 
    "ACSIncome":                "SEX",
    "ACSEmployment":            "SEX",
    "ACSPublicCoverage":        "SEX",
    "ACSMobility":              "SEX",
    "ACSTravelTime":            "SEX",
    "MEPSPanel21":              "SEX", 
    "DefaultCreditCardClients": "x2",   # x2: sex 
}

# -------------------  2. Data Loading (using prada)  -------------------

def get_dataset(dname, seed, fold, silent, remove_col=[]):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    # d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    if d.is_binary():
        d.use_balanced_accuracy()

    # optionally remove protected feature 
    if remove_col:
        d.X = d.X.drop(columns=remove_col)

    dtrain, dtest = d.train_and_test_fold(fold, nfolds=NFOLDS)
    dtrain, dvalid = dtrain.split(fold % (NFOLDS - 1), nfolds=NFOLDS - 1)

    return d, dtrain, dvalid, dtest


def remap_sensitive_to_01(sensitive_col, d, dtrain, dvalid, dtest):
    """Remap sensitive column(s) to {0,1} in-place across all splits."""
    cols = [sensitive_col] if isinstance(sensitive_col, str) else sensitive_col
    for col in cols:
        vals = np.sort(dtrain.X[col].unique())
        assert len(vals) == 2, f"Expected binary sensitive column '{col}', got {vals}"
        lo, hi = vals[0], vals[1]
        if lo == 0 and hi == 1:
            continue
        mapping = {lo: 0, hi: 1}
        for ds in (dtrain, dvalid, dtest):
            ds.X[col] = ds.X[col].map(mapping).astype(veritas.FloatT)
        d.X[col] = d.X[col].map(mapping).astype(veritas.FloatT)
        print(f" * Remapped sensitive column '{col}' values from {vals} to [0,1].")


# ----------------------   3. Model Training   ------------------------

def get_model_path(dname, fold, mtype="xgb", compressed=False):
    compressed_str = "_compressed" if compressed else ""
    return f"{MODEL_DIR}/{dname}_fold{fold}{compressed_str}.{mtype}"

def load_cached_timing(dname, model_type, fold, exp, block, key):
    """
    When reading a cached model, port its build time
    from the results of the run that built it
    (exp="orig": training time, exp="lop": compression time).
    """
    import json, os
    path = f"{CACHE_DIR}/results/{dname}_{model_type}_{fold}_{exp}.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            log = json.load(f)
    except (OSError, ValueError):
        return None
    return log.get(block, {}).get(key)

def get_model_params(model_type):
    if model_type == "xgb":
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'tree_method': 'hist',
            'n_jobs': NJOBS,
            # 'n_threads': 1,
        }
    elif model_type == "rf":
        return {
            'n_estimators': 50,
            'max_depth': 8, 
            'min_samples_split': 5,
            'n_jobs': NJOBS,
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def train_model(
    dname, dtrain, dtest, model_type="xgb", fold=0, cache=True, log={}
):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    import os

    model_path = get_model_path(dname, fold, mtype=model_type, compressed=False)

    if cache and os.path.exists(model_path):
        print(f" * Loading cached {model_type} model from {model_path}")
        at = veritas.AddTree.read(model_path)
        model = None
        # read cached timings from earlier run (if available)
        train_time = load_cached_timing(
            dname, model_type, fold, "orig", "original_at", "train_time"
        )
        train_time_source = "cached" if train_time is not None else None

    else:
        # compute class ratio
        scale_pos_weight = float((dtrain.y == 0).sum()) / float((dtrain.y == 1).sum())
        t_train = time.time()

        # train new model
        best_params = get_model_params(model_type)

        if model_type == "xgb":
            model = xgb.XGBClassifier(
                random_state=SEED,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,  # train for balanced accuracy
                **best_params,
            )
        elif model_type == "rf":
            model = RandomForestClassifier(
                random_state=SEED,
                class_weight="balanced",            # train for balanced accuracy
                **best_params,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(dtrain.X, dtrain.y)
        print(f" * Trained {model_type} with default params: {best_params}")

        train_time = time.time() - t_train
        train_time_source = "measured"
        print(f" * Training took {train_time:.2f}s")

        # convert to veritas addtree
        at = veritas.get_addtree(model.get_booster()) if model_type == "xgb" \
             else veritas.get_addtree(model)

        # check conversion
        ok = veritas.test_conversion(
            at, dtrain.X, model.predict_proba(dtrain.X)[:, 1], silent=True
        )
        assert ok, "veritas.test_conversion failed!"

        # optionally cache model
        if cache:
            with open(model_path, "w") as f:
                f.write(at.to_json())
                print(f" * Saved {model_type} model to {model_path}.")

    # evaluate original model
    acc_train = at_accuracy(at, dtrain.X, dtrain.y)
    acc_test = at_accuracy(at, dtest.X, dtest.y)
    bacc_train = at_balanced_accuracy(at, dtrain.X, dtrain.y)
    bacc_test = at_balanced_accuracy(at, dtest.X, dtest.y)
    print(f"\n * Original accuracy: {acc_train:.4f} train, {acc_test:.4f} test")
    print(f" * Original balanced accuracy: {bacc_train:.4f} train, {bacc_test:.4f} test")

    log["original_at"] = {
        "train_acc": acc_train,
        "test_acc": acc_test,
        "train_bacc": bacc_train,
        "test_bacc": bacc_test,
        "num_leaves": at.num_leafs(),
        "num_trees": len(at),
        "max_depth": at.max_depth(),
        "train_time": train_time,
        # "measured" = timed in this run, "cached" = reused from an earlier run
        "train_time_source": train_time_source,
    }

    return at

def lop_compress(
        dname, fold, dtrain, dvalid, dtest, at_orig, 
        model_type="xgb", abserr=0.005, timeout=21600, 
       cache=True, log={}
):
    import tree_compress
    from sklearn.metrics import balanced_accuracy_score
    import os

    # HACK: we need a bit more compression for one dataset,
    #       or not enumerable.
    if dname in ["BankMarketingFull"] and fold == 2 and model_type == "xgb":
        abserr = 0.0075

    model_path = get_model_path(dname, fold, model_type, compressed=True)
    mtrain = at_balanced_accuracy(at_orig, dtrain.X, dtrain.y)
    # mvalid = dvalid.metric(at_orig)
    mtest =  at_balanced_accuracy(at_orig, dtest.X, dtest.y)
    
    if cache and os.path.exists(model_path):
        print(f" * Loading cached compressed model from {model_path}")
        at_refined = veritas.AddTree.read(model_path)
        # read cached timings from earlier run (if available)
        refine_time = load_cached_timing(
            dname, model_type, fold, "lop", "compressed_at", "lop_time"
        )
        lop_time_source = "cached" if refine_time is not None else None

    else:
        # compress using LOP
        refine_time = time.time()

        data = tree_compress.Data(
            dtrain.X.to_numpy(dtype=veritas.FloatT), dtrain.y.to_numpy(),
            dtest.X.to_numpy(dtype=veritas.FloatT), dtest.y.to_numpy(),
            dvalid.X.to_numpy(dtype=veritas.FloatT), dvalid.y.to_numpy())
        
        # Note: we edit LOP here to reweight leaf refit by class frequency
        class BalancedCompress(tree_compress.Compress):
            def _get_regularized_lin_clf(self, xxtrain):
                clf = super()._get_regularized_lin_clf(xxtrain)
                if not self.is_regression():
                    clf.class_weight = "balanced"
                return clf
            
        compr = BalancedCompress(
            data,
            at_orig,
            score=balanced_accuracy_score,
            isworse=lambda v, ref: ref-v > abserr,
            seed=5823,
            silent=True
        )

        compr.no_convergence_warning = True
        at_refined = compr.compress(max_rounds=2, timeout=timeout)
        refine_time = time.time() - refine_time
        lop_time_source = "measured"
        print(f"\n * LOP compression done in {refine_time:.1f}s")

    mtrain_refined = at_balanced_accuracy(at_refined, dtrain.X, dtrain.y)
    # mvalid_refined = dvalid.metric(at_refined)
    mtest_refined = at_balanced_accuracy(at_refined, dtest.X, dtest.y)
    print(f" * Train bal. accuracy: {mtrain:.4f} --> {mtrain_refined:.4f} (Δ {mtrain_refined - mtrain:.4f})")
    print(f" * Test bal. accuracy:  {mtest:.4f} --> {mtest_refined:.4f} (Δ {mtest_refined - mtest:.4f})")
    print(f" * #leaves: {at_orig.num_leafs()} --> {at_refined.num_leafs()}")

    if cache:
        with open(model_path, "w") as f:
            f.write(at_refined.to_json())
            print(f" * Saved compressed model to {model_path}.")

    log["compressed_at"] = {
        "train_bacc": mtrain_refined,
        "test_bacc": mtest_refined,
        "train_acc": at_accuracy(at_refined, dtrain.X, dtrain.y),
        "test_acc": at_accuracy(at_refined, dtest.X, dtest.y),
        "num_leaves": at_refined.num_leafs(),
        "num_trees": len(at_refined),
        "max_depth": at_refined.max_depth(),
        "lop_time": refine_time,
        # "measured" = timed in this run, "cached" = reused from an earlier run
        "lop_time_source": lop_time_source,
    }

    return at_refined

# ---------------   4. Model conversion (to veritas addtree)  -------------

def fairgbm_to_addtree(model):
        # Register a veritas converter for fairgbm.basic.Booster 
        #  --> it simply uses the existing LGBM converter
        import fairgbm.basic as fgbm_basic
        from veritas.lgbm_converter import addtree_lgbm
        from veritas import AddTreeType, AddTreeConverter, add_addtree_converter

        class FairGBMConverter(AddTreeConverter):
            def convert(self, model, silent):
                if not isinstance(model, fgbm_basic.Booster):
                    from veritas import InapplicableAddTreeConverter
                    raise InapplicableAddTreeConverter()
                return addtree_lgbm(model, AddTreeType.CLF_SOFTMAX)

        add_addtree_converter(FairGBMConverter())
        at = veritas.get_addtree(model.booster_)

        return at

# -----------------------   5. Veritas utilities    -----------------------

def get_ocs_batch(X, at):
    from scipy.sparse import csr_matrix
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

    indices = np.empty(num_examples * num_trees, dtype=np.int32)
    for tree_idx, t in enumerate(at):
        leaf_ids_batch = t.eval_node(X)  # shape (num_examples,)
        lmap = leaf_maps[tree_idx]
        offset = offsets[tree_idx]
        local = np.fromiter((lmap[lid] for lid in leaf_ids_batch), dtype=np.int32, count=num_examples)
        indices[tree_idx::num_trees] = offset + local

    indptr = np.arange(0, (num_examples + 1) * num_trees, num_trees, dtype=np.int32)
    data = np.ones(num_examples * num_trees, dtype=FloatT)
    return csr_matrix((data, indices, indptr), shape=(num_examples, total_leaves))

# @TODO can probably be made quicker
def extract_pairs_batch(solutions, feat_map, columns, ordinal_cols):
    """Extract all (instance0, instance1) pairs from solutions in a batch.

    Sampling convention (veritas uses LtSplit: x < threshold):
      Left branch  (lo_is_unbound): hi is the split threshold, branch contains x < hi.
                   ordinal: hi-1 if hi is exact integer, else floor(hi).
                   continuous: hi - 1e-4.
      Right branch (!lo_is_unbound): lo is the split threshold, branch contains x >= lo.
                   ordinal: ceil(lo) — equals lo for XGBoost integer thresholds;
                            equals next integer for non-integer thresholds (e.g. 0.5 for {0,1} RF).
                   continuous: lo (raw lower bound).
    """
    import veritas
    num_sols = len(solutions)
    nb_features = len(feat_map)
    X = np.zeros((2 * num_sols, nb_features), dtype=veritas.FloatT)

    col_info = []
    for col in columns:
        f0 = feat_map.get_feat_id(col, 0)
        f1 = feat_map.get_feat_id(col, 1)
        f0_idx = f0
        is_ordinal = f0 in ordinal_cols
        col_info.append((f0, f1, f0_idx, is_ordinal))

    for i, sol in enumerate(solutions):
        box = sol.box()
        for instance in (0, 1):
            row_idx = 2 * i + instance
            for f0, f1, f0_idx, is_ordinal in col_info:
                feat_id = f0 if instance == 0 else f1
                if feat_id in box:
                    interval = box[feat_id]
                    if interval.lo_is_unbound():
                        if is_ordinal:
                            hi = interval.hi
                            value = hi - 1 if hi == np.floor(hi) else np.floor(hi)
                        else:
                            # value = interval.hi - 1e-4
                            value = np.nextafter(interval.hi, -np.inf)
                    else:
                        if is_ordinal:
                            value = int(np.ceil(interval.lo))
                        else:
                            # value = interval.lo
                            value = np.nextafter(interval.lo, np.inf)
                    X[row_idx, f0_idx] = value
    return X
# doesn't really improve
# def extract_pairs_batch(solutions, feat_map, columns, ordinal_cols):
#     import veritas
#     num_sols = len(solutions)
#     nb_features = len(feat_map)
#     X = np.zeros((2 * num_sols, nb_features), dtype=veritas.FloatT)

#     boxes = [sol.box() for sol in solutions]

#     for col in columns:
#         for instance in (0, 1):
#             feat_id = feat_map.get_feat_id(col, instance)
#             f0_idx = feat_map.get_feat_id(col, 0)
#             is_ordinal = f0_idx in ordinal_cols
#             row_indices = [2 * i + instance for i in range(num_sols)]

#             intervals = [boxes[i].get(feat_id) for i in range(num_sols)]
#             for i, interval in enumerate(intervals):
#                 if interval is None:
#                     continue
#                 if interval.lo_is_unbound():
#                     hi = interval.hi
#                     value = (hi - 1 if hi == np.floor(hi) else np.floor(hi)) if is_ordinal else hi - 1e-4
#                 else:
#                     value = int(np.ceil(interval.lo)) if is_ordinal else interval.lo
#                 X[row_indices[i], f0_idx] = value

#     return X

def collect_unique_OCs(OCs_train, y_train):
    """
    Collect the unique OCs that the training samples route to.
    """
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

    return unique_OCs

## def collect_unique_OCs_pairs(pairs):
##     """
##     Collect the unique pairs of OCs that the unfair pairs route to.
##     """
##     unique_OCs: dict[tuple[bytes, bytes], dict] = {}
##     for phi1, phi2 in pairs:
##         # CSR row indices are already sorted; use bytes for a fast immutable key.
##         key = (phi1.indices.tobytes(), phi2.indices.tobytes())
##         if key not in unique_OCs:
##             unique_OCs[key] = {"phi1": phi1, "phi2": phi2}
##     return unique_OCs
def collect_unique_OCs_pairs(pairs):
    """
    Collect the unique pairs of OCs that the violating pairs route to.
    New version: does NOT care about order.
    Returns a list of (phi1, phi2) tuples.
    """
    seen = {}
    for phi1, phi2 in pairs:
        key1 = phi1.indices.tobytes()
        key2 = phi2.indices.tobytes()
        if key1 == key2:
            continue
        pair_key = (min(key1, key2), max(key1, key2))
        if pair_key not in seen:
            seen[pair_key] = (phi1, phi2)
    return list(seen.values())

# -----------------------   6. Evaluation Metrics   -----------------------

def at_accuracy(at, X, y):
    """Compute accuracy of an AddTree on data (X, y)."""
    from veritas import FloatT
    from sklearn.metrics import accuracy_score
    X_np = X.to_numpy(dtype=FloatT) if hasattr(X, "to_numpy") else np.asarray(X, dtype=FloatT)
    # preds = at.predict(X_np)
    # preds_bin = (preds >= 0.5).astype(int)
    # switched to at.eval() for better numerical stability
    preds = at.eval(X_np)
    preds_bin = (preds > 0.0).astype(int)
    return accuracy_score(y, preds_bin)

def at_balanced_accuracy(at, X, y):
    """Compute balanced accuracy of an AddTree on data (X, y)."""
    from sklearn.metrics import balanced_accuracy_score
    preds_bin = (at.eval(X.to_numpy(dtype=veritas.FloatT)) > 0.0).astype(int)
    return balanced_accuracy_score(y, preds_bin)

def evaluate_accuracies(at_model, dtrain_set, dtest_set, label="Initial", verbose=True):
    acc_train_val = at_accuracy(at_model, dtrain_set.X, dtrain_set.y)
    acc_test_val = at_accuracy(at_model, dtest_set.X, dtest_set.y)
    bacc_train_val = at_balanced_accuracy(at_model, dtrain_set.X, dtrain_set.y)
    bacc_test_val = at_balanced_accuracy(at_model, dtest_set.X, dtest_set.y)

    # debug
    # manually evaluate balanced accuracy to avoid any caching issues with d.metric()
    # X_train_np = dtrain_set.X.to_numpy(dtype=veritas.FloatT)
    # X_test_np  = dtest_set.X.to_numpy(dtype=veritas.FloatT)
    # y_train_pred = (at_model.predict(X_train_np) >= 0.5).astype(int)
    # y_test_pred  = (at_model.predict(X_test_np)  >= 0.5).astype(int)
    # # y_train_pred = (at_model.predict(dtrain_set.X)>= 0.5).astype(int)
    # # y_test_pred = (at_model.predict(dtest_set.X)>= 0.5).astype(int)
    # from sklearn.metrics import balanced_accuracy_score
    # bacc_train_val = balanced_accuracy_score(dtrain_set.y, y_train_pred)
    # bacc_test_val = balanced_accuracy_score(dtest_set.y, y_test_pred)
    
    if verbose:
        print(f"\n >> {label} accuracy: {acc_train_val:.4f} train, {acc_test_val:.4f} test")
        print(f" >> {label} balanced accuracy: {bacc_train_val:.4f} train, {bacc_test_val:.4f} test")
    return acc_train_val, acc_test_val, bacc_train_val, bacc_test_val

# def compute_allowed_misclassifications(at, X, y, max_acc_drop):
#     """
#     Compute how many training samples can be misclassified
#     given the max_acc_drop budget (regular accuracy).
#     """
#     n = X.shape[0] if hasattr(X, "shape") else len(X)
#     old_acc = at_accuracy(at, X, y)
#     old_correct = int(np.round(old_acc * n))
#     allowed_drop = int(np.floor(float(max_acc_drop) * float(n)))
#     assert max_acc_drop >= 0.0 and max_acc_drop <= 1.0, "max_acc_drop not in [0,1]??"
#     min_correct = max(0, old_correct - allowed_drop)
#     return min_correct, old_correct
def compute_allowed_misclassifications(at, X, y, max_bacc_drop):
    """
    Compute the minimum balanced accuracy 
    the MILP must preserve given the budget.
    """
    from veritas import FloatT
    y_arr = np.asarray(y).ravel()
    X_np = X.to_numpy(dtype=FloatT) if hasattr(X, "to_numpy") else np.asarray(X, dtype=FloatT)
    preds = (at.eval(X_np).ravel() >= 0.0).astype(int)
    P = int(np.sum(y_arr == 1))
    N = int(np.sum(y_arr == 0))
    old_tp = int(np.sum((preds == 1) & (y_arr == 1)))
    old_tn = int(np.sum((preds == 0) & (y_arr == 0)))
    old_bacc = 0.5 * (old_tp / P + old_tn / N)
    min_bacc = max(0.5, old_bacc - max_bacc_drop)
    print(f"\n * Total samples: {len(y_arr)}, P: {P}, N: {N}")
    print(f" * Old TP: {old_tp}, Old TN: {old_tn}")
    print(f" * Old balanced accuracy: {old_bacc:.4f}, minimum required: {min_bacc:.4f} (drop budget {max_bacc_drop:.4f})")
    assert min_bacc >= 0.5 and min_bacc <= 1.0, f"min bacc ={min_bacc}??"
    return min_bacc, old_tp, old_tn, P, N


# --------------------------- PRINTING SUMMARY BLOCKS ---------------------------
_ACC    = [("Test Accuracy", "test_acc"), ("Test Balanced Acc", "test_bacc")]
_FAIR   = [("Unfair Boxes", "#unfair_boxes"), ("Empirical Fairness", "emp_fairness")]
_LEAVES = [("Leaves", "num_leaves")]

# summary block -> (title, [(label, log metric), ...])
_BLOCKS = {
    "original_at":   ("Original Model",         _ACC + _FAIR + [("Enumerable", "enumerable")] + _LEAVES),
    "compressed_at": ("Compressed Model (LOP)", _ACC + _FAIR + _LEAVES),
    "repaired_at":   ("Repaired (MILP)",        [("Status", "status"), ("Iterations", "iterations")] + _ACC + _FAIR),
    "fatt":          ("FATT",                   _ACC + _FAIR + _LEAVES),
    "budro":         ("BuDRO",                  _ACC + _FAIR + _LEAVES),
}

SUMMARY_SECTIONS = {
    "orig":   ["original_at"],
    "lop":    ["compressed_at"],
    "milp":   ["repaired_at"],
    "milp-b": ["repaired_at"],
    "fatt":   ["fatt"],
    "budro":  ["budro"],
}

def _fmt(v, spec=".4f"):
    # bool is a subclass of int: check it first so True doesn't print as 1.0000
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return str(v) if isinstance(v, int) else format(v, spec)

def print_summary(log, dname, fold, exp):
    print(" ===================== SUMMARY ==================== ")
    print(f"\nDataset: {dname}, Fold: {fold}" + (f", Experiment: {exp}" if exp else ""))

    sections = SUMMARY_SECTIONS[exp] if exp else list(_BLOCKS)
    for i, section in enumerate(sections, 1):
        title, rows = _BLOCKS[section]
        key = section
        if section == "repaired_at":
            key = next((k for k in log if k.endswith("_repaired_at")), "composite_repaired_at")
            title = f"Repaired ({key[:-len('_repaired_at')]})"
        block = log.get(key, {})

        print(f"\n{i}. {title}")
        for label, metric in rows:
            print(f"   {label}: {_fmt(block.get(metric, 'N/A'))}")

    print(" ==============================================")
