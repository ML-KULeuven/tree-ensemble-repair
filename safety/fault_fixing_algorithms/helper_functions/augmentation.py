from typing import List, Tuple, Dict, Any, Optional
import torch
import pandas as pd
import numpy as np
from nnet import *


def prepare_faults_with_nn_labels(
    Fx: pd.DataFrame,       # fault states (may contain duplicates)
    Fy: pd.Series,          # original forbidden labels (aligned to ORIGINAL Fx)
    Aa: List[List[int]],    # Applicable actions per original fault row
    nn_model,               # nnet.NNet instance
    num_actions: int,       # total number of actions
    # fault_weight_multipliers: Optional[np.ndarray] = None, # aligned to ORIGINAL Fx
):
    """
    Preparation of NN-corrected labels for augmented retraining.
    Behavior:
    1) Group rows by unique fault state.
    2) Aggregate all forbidden actions per unique state.
    3) For each unique state, NN picks best action among ALLOWED = Applicable \ Forbidden.
    4) Generate: Fx_retrain, Fy_retrain, Aa_retrain:
        - Fx_retrain: unique fault states
        - Fy_retrain: unique fault NN-corrected labels
        - Aa_retrain: applicable actions per unique state
    """
   
    # 1. Group by unique state, assert identical Aa across duplicates, union forbidden 
    bucket = {}
    for idx, (x, fy, actions) in enumerate(zip(Fx.values, Fy.values, Aa)):
        key = tuple(x)
        actions_set = set(actions)
        if key not in bucket:
            bucket[key] = {
                "state": x,
                "app": actions_set,     # store first Aa
                "forbid": {int(fy)},    # start forbid set
                "indices": [idx],
            }
        else:
            if actions_set != bucket[key]["app"]:
                raise ValueError(
                    f"Applicable actions mismatch for duplicate state {x}. "
                    f"First seen: {sorted(bucket[key]['app'])}, "
                    f"Now: {sorted(actions_set)}"
                )
            bucket[key]["forbid"].add(int(fy))
            bucket[key]["indices"].append(idx)

    # 2. Extract NN-corrected label per unique state: --> Allowed = Applicable \ union(forbid) 
    key2label = {}
    for key, info in bucket.items():
        x          = info["state"]
        app_list   = sorted(info["app"])
        forbid_set = set(info["forbid"])

        allowed = [a for a in app_list if a not in forbid_set]
        if not allowed:
            raise ValueError(
                f"No safe allowed action for state {x}. "
                f"Applicable={app_list}, Forbidden={sorted(forbid_set)}"
            )

        # with torch.no_grad():
        #     preds = nn_model.evaluate_network(torch.FloatTensor(x)) # shape: (num_actions,)
        # # mask to allowed only
        # mask = torch.zeros(num_actions, dtype=torch.bool)
        # for a in allowed:
        #     if 0 <= a < num_actions: mask[a] = True
        # preds[~mask] = -float("inf")

        # label = int(torch.argmax(preds).item())
        # if (label not in allowed) or (label in forbid_set):
        #     raise ValueError(
        #         f"NN selected a disallowed/forbidden action {label} for state {x} "
        #         f"(allowed={allowed}, forbidden={sorted(forbid_set)})."
        #     )
        # key2label[key] = label
        with torch.no_grad():
            # Convert state to tensor and get NN predictions
            state_tensor = torch.FloatTensor(x)
            preds = nn_model.evaluate_network(state_tensor)  # Keep as tensor

            # Mask invalid actions (allowed actions only)
            action_mask = torch.zeros(num_actions, dtype=torch.bool)
            for a in allowed:  # 'allowed' should be the applicable actions for this state
                if 0 <= a < num_actions:
                    action_mask[a] = True

            # Apply mask: set invalid actions to -inf
            preds[~action_mask] = -float('inf')

            # Safety check: ensure at least one valid action
            assert action_mask.any(), f"No safe actions available for state {x}"

            # Pick the best allowed action
            label = int(np.argmax(preds).item())
            if (label not in allowed) or (label in forbid_set):
                raise ValueError(
                    f"NN selected a disallowed/forbidden action {label} for state {x} "
                    f"(allowed={allowed}, forbidden={sorted(forbid_set)})."
                )
            key2label[key] = label


    Fx_retrain = pd.DataFrame([info["state"] for info in bucket.values()])#.astype(np.float32).astype(veritas.FloatT)
    Fy_retrain = pd.Series([key2label[key] for key in bucket.keys()], dtype=int)
    Aa_retrain = [sorted(info["app"]) for info in bucket.values()]

    return Fx_retrain, Fy_retrain, Aa_retrain

    # # --- 3) Extract unique state-value pairs for training (using max weight for duplicates) ---
    # Fx_retrain, Fy_retrain, Aa_retrain, w_retrain = [], [], [], []
    # for key, info in bucket.items():
    #     Fx_retrain.append(info["state"])
    #     Fy_retrain.append(key2label[key])
    #     Aa_retrain.append(sorted(info["app"]))  

    #     # assign max weight across duplicates
    #     if fault_weight_multipliers is not None:
    #         idxs = info["indices"]
    #         max_w = float(np.max(fault_weight_multipliers[idxs])) if len(idxs) > 0 else 1.0
    #         w_retrain.append(max_w)

    # Fx_retrain = (
    #     pd.DataFrame(Fx_retrain)
    #     .astype(np.float32)
    #     .astype(veritas.FloatT)
    #     .reset_index(drop=True)
    # )
    # Fy_retrain = pd.Series(Fy_retrain, dtype=int)
    # if w_retrain is not None:   w_retrain = np.asarray(w_retrain, dtype=float)

    # return Fx_retrain, Fy_retrain, Aa_retrain, w_retrain