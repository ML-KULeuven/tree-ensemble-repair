# FATT (Fairness-Aware Tree Training) - Ranzato & Zanella, 2021
# https://github.com/fatt21/fatt

# @TODO: try training FATT Forests?
#  (not in NeurIPS paper as the authors say they aren't better than single trees)

import subprocess
import numpy as np
import pandas as pd
import veritas

from utils import FATT_BIN, FATT_CACHE, SILVA_BIN

# ---------------------------------------------------------------------------
# 1. Dataset export for FATT: from (X, y) to FATT's CSV format
# ---------------------------------------------------------------------------

def export_dataset_fatt(X, y, path):
    """
    Export (X, y) to FATT's CSV format:
        # N M
        label,feat0,feat1,...
    Label column comes first (no header row).
    """
    df = pd.DataFrame(X)
    df.insert(0, '__label__', np.asarray(y).astype(int))
    with open(path, 'w') as f:
        f.write(f'# {len(df)} {len(X.columns) if hasattr(X, "columns") else X.shape[1]}\n')
    df.to_csv(path, header=False, index=False, mode='a')


# ---------------------------------------------------------------------------
# 2. FATT hyperparameter grid (optional) and defaults
# ---------------------------------------------------------------------------

def get_fatt_param_grid():
    return {
        "fitness_stability": [0.1, 0.3],
        "fitness_accuracy": [0.5, 0.9, 1.0],
        "max_iterations": [10, 100],
        "aggressiveness": [0.01, 0.1, 0.3],
    }

DEFAULT_FATT_CONFIG = {
    'fitness_epsilon': 0.5,
    'fitness_accuracy': 0.9, # 0.9: worse on Compas?
    'fitness_stability': 0.1,
    'mutation': 'Z',
    'max_iterations': 100,
    'aggressiveness': 0.01,
    'seed': 1,
}

# ---------------------------------------------------------------------------------
# 3. Train FATT decision tree using compiled meta-silvae binary, with given config
#    a) single decision tree with specified/default config
#    b) hyperparameter tuning on validation set --> returns best config
# ---------------------------------------------------------------------------------

def train_fatt(meta_silvae_bin, train_csv, output_silva, config=None):
    """
    Run meta-silvae to train a fair decision tree.

    Parameters
    ----------
    meta_silvae_bin : str  path to compiled meta-silvae binary
    train_csv       : str  path to training CSV in FATT format (label first)
    output_silva    : str  path where the trained .silva model will be written
    config          : dict | None  training configuration (see DEFAULT_FATT_CONFIG)

    Returns
    -------
    str  path to the output .silva file (same as output_silva)
    """
    cfg = {**DEFAULT_FATT_CONFIG, **(config or {})}

    cmd = [
        meta_silvae_bin,
        train_csv,
        output_silva,
        '--fitness', 'linear',
        str(cfg['fitness_epsilon']),
        str(cfg['fitness_accuracy']),
        '0',
        str(cfg['fitness_stability']),
        '0', '0', '0', '0', '0', '0',
        '--mutation', cfg['mutation'],
        '--max-iteration', str(cfg['max_iterations']),
        '--split-search-aggressiveness', str(cfg['aggressiveness']),
        '--seed', str(cfg['seed']),
    ]

    print(f"Running meta-silvae: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"meta-silvae failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_silva

# OLD: selects the highest validation bacc, ignoring fairness.
# @TODO: I don't think it makes sense to select the best bacc.
#        We want the property to hold,
#        so highest bacc. SUCH THAT emp. fair. == 1 on validation set?
# def tune_fatt_config(d, fold, dtrain, dvalid, base_config):
#     from fatt_baseline import (
#         get_fatt_param_grid,
#         export_dataset_fatt,
#         train_fatt,
#         silva_accuracy
#     )
#     import itertools
#     import os
#     import tempfile
#
#     param_grid = get_fatt_param_grid()
#     param_names = list(param_grid.keys())
#     param_values = [param_grid[name] for name in param_names]
#     param_combos = list(itertools.product(*param_values))
#
#     train_csv = f"{FATT_CACHE}/{d.name()}_fold{fold}_tune_train.csv"
#     export_dataset_fatt(dtrain.X, dtrain.y, train_csv)
#
#     best_config = None
#     best_bacc = -1.0
#
#     print(f"\n  Tuning FATT over {len(param_combos)} configs...")
#     for combo in param_combos:
#         config = {**base_config}
#         for name, val in zip(param_names, combo):
#             config[name] = val
#
#         with tempfile.NamedTemporaryFile(suffix=".silva", delete=False) as tmp:
#             silva_path = tmp.name
#
#         try:
#             train_fatt(FATT_BIN, train_csv, silva_path, config)
#             _, bacc_val = silva_accuracy(SILVA_BIN, silva_path, dvalid.X, dvalid.y)
#             if bacc_val > best_bacc:
#                 best_bacc = bacc_val
#                 best_config = config
#         finally:
#             if os.path.exists(silva_path):
#                 os.remove(silva_path)
#
#     if os.path.exists(train_csv):
#         os.remove(train_csv)
#
#     print(f"  Best FATT config (val bacc={best_bacc:.4f}): {best_config}")
#     return best_config

# NEW: same rule as BuDRO tuning: highest validation bacc among configs with
#      validation empirical fairness == 1.0. If none, fall back to highest empirical
#      fairness, then bacc (and print a warning). ordinal_cols is unused.
def tune_fatt_config(d, fold, dtrain, dvalid, base_config, sensitive_col, ordinal_cols):
    from fatt_baseline import (
        get_fatt_param_grid,
        export_dataset_fatt,
        train_fatt,
        silva_accuracy
    )
    from fairness import evaluate_empirical_fairness
    import itertools
    import os
    import tempfile

    param_grid = get_fatt_param_grid()
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combos = list(itertools.product(*param_values))

    train_csv = f"{FATT_CACHE}/{d.name()}_fold{fold}_tune_train.csv"
    export_dataset_fatt(dtrain.X, dtrain.y, train_csv)

    best_config = None
    best_key = None

    print(f"\n  Tuning FATT over {len(param_combos)} configs...")
    for combo in param_combos:
        config = {**base_config}
        for name, val in zip(param_names, combo):
            config[name] = val

        with tempfile.NamedTemporaryFile(suffix=".silva", delete=False) as tmp:
            silva_path = tmp.name

        try:
            train_fatt(FATT_BIN, train_csv, silva_path, config)
            _, bacc_val = silva_accuracy(SILVA_BIN, silva_path, dvalid.X, dvalid.y)
            at = fatt_to_addtree(silva_path)
            emp_fair_val = evaluate_empirical_fairness(at, dvalid, sensitive_col)
            key = (emp_fair_val == 1.0, emp_fair_val, bacc_val)
            if best_key is None or key > best_key:
                best_key = key
                best_config = config
        finally:
            if os.path.exists(silva_path):
                os.remove(silva_path)

    if os.path.exists(train_csv):
        os.remove(train_csv)

    if not best_key[0]:
        print("  [!] No FATT config is empirically fair on validation: picked the one with the highest val. empirical fairness")
    print(f"  Best FATT config (val bacc={best_key[2]:.4f}, val emp_fair={best_key[1]:.4f}): {best_config}")
    return best_config

# ---------------------------------------------------------------------------
# 4. Gather .silva model predictions using original Silva binary
#    (for comparison with addtree)
# ---------------------------------------------------------------------------
def silva_predict(silva_bin, model_path, X, y_dummy=None):
    """
    Return binary predictions (0/1 array) by calling the silva C++ binary.
    
    Used as a sanity check to validate AddTree predictions against the original
    FATT implementation.
    
    Parameters
    ----------
    silva_bin : str  path to compiled silva binary
    model_path : str  path to .silva model file
    X : np.ndarray or pd.DataFrame  shape [N, n_features]
    y_dummy : np.ndarray, optional  dummy labels for CSV format (default: all 0)
    
    Returns
    -------
    np.ndarray  shape [N,], binary predictions (0 or 1)
    """
    import tempfile
    import os
    
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy(dtype=np.float64)
    
    # Create dummy labels if not provided
    if y_dummy is None:
        y_dummy = np.zeros(len(X), dtype=int)
    
    # Use export_dataset_fatt to create the test CSV 
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
    temp_out = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False).name
    
    export_dataset_fatt(X, y_dummy, temp_csv)
    
    # Call silva binary with --voting average
    cmd = [
        silva_bin,
        model_path,
        temp_csv,
        '--voting', 'average'
    ]
    
    with open(temp_out, 'w') as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, 
                                text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"silva binary failed (exit {result.returncode}):\n{result.stderr}"
        )
    
    # Output format: Classifier Dataset ID Label Concrete Result Time
    preds = []
    with open(temp_out) as f:
        for line in f:
            line = line.strip()
            if not line or 'Classifier' in line or 'SUMMARY' in line:
                # skip header, summary or empty lines
                continue
            parts = line.split()
            try:
                # 'Concrete' column is the predicted label (0 or 1)
                concrete = int(parts[4]) 
                if concrete not in (0, 1):
                    raise ValueError(f"Unexpected label: {concrete}")
                preds.append(concrete)
            except (ValueError, IndexError) as e:
                raise RuntimeError(f"Cannot parse silva output line: {line!r}") from e
    
    # Clean up temp files
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    if os.path.exists(temp_out):
        os.remove(temp_out)

    return np.array(preds, dtype=int)

def silva_accuracy(silva_bin, model_path, X, y):
    """Compute accuracy and balanced accuracy of a .silva model on (X, y)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score as bacc_score
    preds = silva_predict(silva_bin, model_path, X)
    return accuracy_score(y, preds), bacc_score(y, preds)
  

# ---------------------------------------------------------------------------
# 5. Converter from .silva to veritas AddTree
# ---------------------------------------------------------------------------

def fatt_to_addtree(path):
    """
    Parse a FATT model (.silva format) into a veritas AddTree (CLF_MEAN).
    """
    with open(path) as f:
        content = f.read()
    tokens = content.split()
    idx = 0

    # The file is either a forest or a single tree
    if tokens[idx] == 'classifier-forest':
        idx += 1
        n_trees = int(tokens[idx]); idx += 1
        print("*** [!] WARNING: FATT forest parsing has not been tested yet [!] ***")
    else:
        n_trees = 1  # single tree, header will be classifier-decision-tree

    # build the AddTree incrementally
    state = {'idx': idx}

    def next_tok():
        t = tokens[state['idx']]
        state['idx'] += 1
        return t

    def parse_tree_header():
        tok = next_tok()
        assert tok == 'classifier-decision-tree', \
            f"Expected classifier-decision-tree, got {tok!r}"
        n_features = int(next_tok())
        n_classes = int(next_tok())
        labels = []
        for _ in range(n_classes):
            label_tok = next_tok()
            # Handle both numeric labels (0, 1) and boolean labels (True, False)
            if label_tok == 'True':
                labels.append('1')
            elif label_tok == 'False':
                labels.append('0')
            else:
                labels.append(label_tok)
        return n_features, n_classes, labels

    def parse_node(vtree, node_id, n_classes, class1_idx):
        tok = next_tok()
        if tok == 'LEAF':
            counts = [float(next_tok()) for _ in range(n_classes)]
            total = sum(counts)
            p1 = counts[class1_idx] / total if total > 0.0 else 0.5
            vtree.set_leaf_values(node_id, [p1])
        elif tok == 'SPLIT':
            feat_id = int(next_tok())
            threshold = float(next_tok())
            # Silva splits on x <= threshold (left), veritas on x < threshold (left).
            # Shift by one ULP so veritas's x < nextafter(k) matches silva's x <= k.
            # vtree.split(node_id, feat_id, threshold)
            vtree.split(node_id, feat_id, np.nextafter(threshold, np.inf))
            parse_node(vtree, vtree.left(node_id), n_classes, class1_idx)
            parse_node(vtree, vtree.right(node_id), n_classes, class1_idx)
        else:
            raise ValueError(f"Unknown node token: {tok!r}")

    at = veritas.AddTree(1, veritas.AddTreeType.CLF_MEAN)

    for _ in range(n_trees):
        _, n_classes, labels = parse_tree_header()
        assert n_classes == 2, "Only binary classification supported"
        class1_idx = labels.index('1')
        vtree = at.add_tree()
        parse_node(vtree, vtree.root(), n_classes, class1_idx)

    at.set_base_score(0, -n_trees/2)
    return at

# OLD: issue in <= splits: 
#           silva uses x <= threshold for left, 
#           veritas uses x < threshold for left. 
#      This means that samples with x == threshold would go left in silva but right in veritas. 
#      The fix is to shift the split threshold by one ULP (the smallest representable float difference) 
#      so that veritas's x < nextafter(k) matches silva's x <= k.
# 
# +    it was hardcoded to subtract 0.5 to each leaf value without using base score 
#      (weird but not strictly wrong)
# def fatt_to_addtree(path):
#     """
#     Parse a FATT model (.silva format) into a veritas AddTree (CLF_MEAN).
#     Leaf values are stored as (p1 - 0.5) so that eval(x) >= 0 indicates class 1.
#     """
#     with open(path) as f:
#         content = f.read()
#     tokens = content.split()
#     idx = 0

#     # The file is either a forest or a single tree
#     if tokens[idx] == 'classifier-forest':
#         idx += 1
#         n_trees = int(tokens[idx]); idx += 1
#         print("*** [!] WARNING: FATT forest parsing has not been tested yet[!] ***")
#     else:
#         n_trees = 1  # single tree, header will be classifier-decision-tree

#     # build the AddTree incrementally
#     state = {'idx': idx}

#     def next_tok():
#         t = tokens[state['idx']]
#         state['idx'] += 1
#         return t

#     def parse_tree_header():
#         tok = next_tok()
#         assert tok == 'classifier-decision-tree', \
#             f"Expected classifier-decision-tree, got {tok!r}"
#         n_features = int(next_tok())
#         n_classes = int(next_tok())
#         labels = []
#         for _ in range(n_classes):
#             label_tok = next_tok()
#             # Handle both numeric labels (0, 1) and boolean labels (True, False)
#             if label_tok == 'True':
#                 labels.append('1')
#             elif label_tok == 'False':
#                 labels.append('0')
#             else:
#                 labels.append(label_tok)
#         return n_features, n_classes, labels

#     def parse_node(vtree, node_id, n_classes, class1_idx):
#         tok = next_tok()
#         if tok == 'LEAF':
#             counts = [float(next_tok()) for _ in range(n_classes)]
#             total = sum(counts)
#             p1 = counts[class1_idx] / total if total > 0.0 else 0.5
#             vtree.set_leaf_values(node_id, [p1 - 0.5])
#         elif tok == 'LEAF_LOGARITHMIC':
#             log_scores = [float(next_tok()) for _ in range(n_classes)]
#             max_s = max(log_scores)
#             exp_s = [math.exp(s - max_s) for s in log_scores]
#             p1 = exp_s[class1_idx] / sum(exp_s)
#             vtree.set_leaf_values(node_id, [p1 - 0.5])
#         elif tok == 'SPLIT':
#             feat_id = int(next_tok())
#             threshold = float(next_tok())
#             vtree.split(node_id, feat_id, threshold)
#             parse_node(vtree, vtree.left(node_id), n_classes, class1_idx)
#             parse_node(vtree, vtree.right(node_id), n_classes, class1_idx)
#         else:
#             raise ValueError(f"Unknown node token: {tok!r}")

#     at = veritas.AddTree(1, veritas.AddTreeType.CLF_MEAN)

#     for _ in range(n_trees):
#         _, n_classes, labels = parse_tree_header()
#         assert n_classes == 2, "Only binary classification supported"
#         class1_idx = labels.index('1')
#         vtree = at.add_tree()
#         parse_node(vtree, vtree.root(), n_classes, class1_idx)

#     at.set_base_score(0, 0.0)
#     return at

# ---------------------------------------------------------------------------
# old pure-Python parsing and predictors for .silva models  
# (alternative to check consistency wrt addtree -- useless,
#  because it parses the .silva model in the same way)
# ---------------------------------------------------------------------------

# def _binary_predict(at, X_np):
#     """Return binary predictions (0/1 array) from a veritas AddTree."""
#     preds = at.predict(X_np)
#     preds_bin = (preds >= 0.5).astype(int)
#     if preds_bin.ndim > 1:
#         # multi-output: take column for class 1
#         preds_bin = preds_bin[:, -1]
#     return preds_bin.flatten()


# def silva_predict_py(path, X, trace_idx=None):
#     """
#     Pure-Python tree traversal of a .silva model.

#     If trace_idx is an int, prints the leaf reached by X[trace_idx] along
#     with the raw counts and computed probabilities for that sample.

#     Returns binary predictions (0/1 array, same convention as the silva binary).
#     """
#     if hasattr(X, 'to_numpy'):
#         X = X.to_numpy(dtype=np.float64)

#     with open(path) as f:
#         tokens = f.read().split()
#     pos = [0]  # mutable index

#     def tok():
#         t = tokens[pos[0]]; pos[0] += 1; return t

#     first = tok()
#     if first == 'classifier-forest':
#         n_trees = int(tok())
#     else:
#         pos[0] -= 1
#         n_trees = 1

#     # parse labels and build per-tree (probs_per_class, ) structures
#     all_labels = None
#     trees = []  # list of (labels, root_fn) where root_fn(x) -> probs array

#     for _ in range(n_trees):
#         assert tok() == 'classifier-decision-tree'
#         int(tok())  # n_features, unused
#         n_classes = int(tok())
#         labels = []
#         for _ in range(n_classes):
#             t = tok()
#             if t == 'True':   labels.append(1)
#             elif t == 'False': labels.append(0)
#             else:              labels.append(int(t))
#         if all_labels is None:
#             all_labels = labels

#         def parse_node():
#             t = tok()
#             if t == 'LEAF':
#                 counts = np.array([float(tok()) for _ in range(n_classes)])
#                 total = counts.sum()
#                 probs = counts / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
#                 return ('leaf', probs, counts)
#             elif t == 'LEAF_LOGARITHMIC':
#                 log_s = np.array([float(tok()) for _ in range(n_classes)])
#                 log_s -= log_s.max()
#                 exp_s = np.exp(log_s)
#                 probs = exp_s / exp_s.sum()
#                 return ('leaf', probs, None)
#             elif t == 'SPLIT':
#                 feat_id = int(tok()); threshold = float(tok())
#                 left  = parse_node()
#                 right = parse_node()
#                 return ('split', feat_id, threshold, left, right)
#             else:
#                 raise ValueError(f"Unknown token: {t!r}")

#         root = parse_node()

#         def make_predict(root_node):
#             def predict(x):
#                 node = root_node
#                 while node[0] == 'split':
#                     _, fid, thr, left, right = node
#                     node = left if x[fid] <= thr else right  # silva uses <=
#                 return node  # ('leaf', probs, counts)
#             return predict

#         trees.append(make_predict(root))

#     label_map = np.array(all_labels)
#     n_classes = len(all_labels)
#     probs_acc = np.zeros((len(X), n_classes))

#     for predict_fn in trees:
#         for i, x in enumerate(X):
#             leaf = predict_fn(x)
#             probs_acc[i] += leaf[1]
#             if trace_idx is not None and i == trace_idx and predict_fn is trees[0]:
#                 print(f"\n[TRACE] Sample {trace_idx}:")
#                 print(f"  labels (header order): {all_labels}")
#                 print(f"  leaf counts (header order): {leaf[2]}")
#                 print(f"  leaf probs  (header order): {leaf[1].round(4)}")
#                 p1_idx = list(all_labels).index(1)
#                 print(f"  class-1 is at header index {p1_idx}")
#                 print(f"  p1 = {leaf[1][p1_idx]:.4f}  → py-pred = {label_map[np.argmax(leaf[1])]}")

#     probs_acc /= len(trees)
#     return label_map[np.argmax(probs_acc, axis=1)]


# class _SilvaTree:
#     """Internal node for pure-Python .silva tree traversal."""
#     __slots__ = ('feat_id', 'threshold', 'left', 'right', 'leaf_probs')

#     def predict_proba(self, x):
#         if self.leaf_probs is not None:
#             return self.leaf_probs
#         if x[self.feat_id] < self.threshold:
#             return self.left.predict_proba(x)
#         return self.right.predict_proba(x)


# def _parse_silva_tree_py(tokens_state, n_classes):
#     """Parse one tree into a pure-Python _SilvaTree structure."""
#     def next_tok():
#         t = tokens_state['tokens'][tokens_state['idx']]
#         tokens_state['idx'] += 1
#         return t

#     def parse_node():
#         tok = next_tok()
#         node = _SilvaTree()
#         node.leaf_probs = None
#         if tok == 'LEAF':
#             counts = np.array([float(next_tok()) for _ in range(n_classes)])
#             total = counts.sum()
#             node.leaf_probs = counts / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
#             node.feat_id = node.threshold = node.left = node.right = None
#         elif tok == 'LEAF_LOGARITHMIC':
#             log_scores = np.array([float(next_tok()) for _ in range(n_classes)])
#             log_scores -= log_scores.max()
#             exp_s = np.exp(log_scores)
#             node.leaf_probs = exp_s / exp_s.sum()
#             node.feat_id = node.threshold = node.left = node.right = None
#         elif tok == 'SPLIT':
#             node.feat_id = int(next_tok())
#             node.threshold = float(next_tok())
#             node.left = parse_node()
#             node.right = parse_node()
#         else:
#             raise ValueError(f"Unknown node token: {tok!r}")
#         return node

#     return parse_node()


# def silva_predict(path, X):
#     """
#     Return binary predictions (0/1 array) for X using a .silva model.
#     Uses pure-Python tree traversal; does not rely on veritas.

#     Parameters
#     ----------
#     path : str  path to .silva file
#     X    : np.ndarray or pd.DataFrame  shape [N, n_features]
#     """
#     if hasattr(X, 'to_numpy'):
#         X = X.to_numpy(dtype=np.float64)

#     with open(path) as f:
#         content = f.read()
#     tokens_state = {'tokens': content.split(), 'idx': 0}

#     def next_tok():
#         t = tokens_state['tokens'][tokens_state['idx']]
#         tokens_state['idx'] += 1
#         return t

#     first = next_tok()
#     if first == 'classifier-forest':
#         n_trees = int(next_tok())
#     else:
#         # first token was 'classifier-decision-tree'; put it back
#         tokens_state['idx'] -= 1
#         n_trees = 1

#     py_trees = []
#     labels = None
#     for _ in range(n_trees):
#         tok = next_tok()
#         assert tok == 'classifier-decision-tree'
#         int(next_tok())  # n_features, unused
#         n_classes = int(next_tok())
#         tree_labels = []
#         for _ in range(n_classes):
#             label_tok = next_tok()
#             # Handle both numeric labels (0, 1) and boolean labels (True, False)
#             if label_tok == 'True':
#                 tree_labels.append(1)
#             elif label_tok == 'False':
#                 tree_labels.append(0)
#             else:
#                 tree_labels.append(int(label_tok))
#         if labels is None:
#             labels = tree_labels
#         tree = _parse_silva_tree_py(tokens_state, n_classes)
#         py_trees.append(tree)

#     n_classes = len(labels)
#     probs = np.zeros((len(X), n_classes))
#     for tree in py_trees:
#         for i, x in enumerate(X):
#             probs[i] += tree.predict_proba(x)
#     probs /= len(py_trees)
#     # map argmax index back to actual label value (order in .silva may not be 0,1)
#     label_map = np.array(labels)
#     return label_map[np.argmax(probs, axis=1)]


# def silva_accuracy(path, X, y):
#     """Compute accuracy of a .silva model on (X, y)."""
#     from sklearn.metrics import accuracy_score, balanced_accuracy_score as bacc_score
#     preds = silva_predict(path, X)
#     return accuracy_score(y, preds), bacc_score(y, preds)
