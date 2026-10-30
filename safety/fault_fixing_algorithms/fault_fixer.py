from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple
from fault_fixing_algorithms.helper_functions.milp.leaf_refinement import get_oc
def check_safety(at, at_optimized, forbidden_x, forbidden_y, applicable_actions):
    """
    Check that no faults happen anymore on updated model.
    """

    for i, (fx, fy, aa) in enumerate(zip(forbidden_x.values, forbidden_y, applicable_actions)):

        # print(f"Checking fault {i}/{forbidden_x.shape[0]}... ")
        # original_prediction = np.argmax(at.predict(fx), axis=1)[0]
        # original_prediction = aa[np.argmax(at.predict(fx)[0][aa])]
        # refined_prediction = aa[np.argmax(at_optimized.predict(fx)[0][aa])]
        original_prediction = aa[np.argmax(at.eval(fx)[0][aa])]
        refined_prediction = aa[np.argmax(at_optimized.eval(fx)[0][aa])]

        assert refined_prediction != fy, \
            f"Refined model still at fault for forbidden state-value pair! " \
            f"State: {fx}, " \
            f"Faulty label: {fy}, " \
            f"Original prediction: {original_prediction}, " \
            f"Refined prediction: {refined_prediction}, " \
            f"Applicable actions: {aa}," \
            f"at.eval(fx): {[ i for i in at.eval(fx)[0]]}," \
            f"at_optimized.eval(fx)[0]: {[i for i in at_optimized.eval(fx)[0]]}, " \
            f"OC for this prediction: {get_oc(fx, at_optimized)}"
            # f"np.max(at_optimized.predict(fx)[0][aa]): {np.max(at_optimized.predict(fx)[0][aa])}" 

########################### HELPERS FOR RETRAINING BASELINES ##############################

# alternative check_safety that just reports number of unfixed faults (i.e., no hard check)
def safety_report(at, at_optimized, forbidden_x, forbidden_y, applicable_actions):

    count_unfixed, count_fixed = 0, 0
    for __, (fx, fy, aa) in enumerate(zip(forbidden_x.values, forbidden_y, applicable_actions)):

        # original_prediction = aa[np.argmax(at.eval(fx)[0][aa])]
        refined_prediction = aa[np.argmax(at_optimized.eval(fx)[0][aa])]

        if refined_prediction == fy:
            count_unfixed += 1
        else:
            count_fixed += 1
    
    print(f"\n\t- safety report: {count_fixed} faults fixed, {count_unfixed} faults unfixed.")
    return count_unfixed > 0

def remove_conflicting_rows(
    df: pd.DataFrame,
    y: pd.DataFrame | pd.Series | np.ndarray,
    fault_df: pd.DataFrame,
    fault_y: pd.DataFrame | pd.Series | np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows from (df, y) that also appear in fault_df AND have the same label
    as the forbidden label, to avoid contradictory training signals.
    """
    # Early exit: no faults -> nothing to remove
    if fault_df is None or len(fault_df) == 0:
        y_flat = pd.Series(np.asarray(y).reshape(-1))
        return df.reset_index(drop=True), y_flat.reset_index(drop=True)

    # Coerce labels to 1-D vectors (never scalars)
    y_flat        = pd.Series(np.asarray(y).reshape(-1))
    fault_y_flat  = pd.Series(np.asarray(fault_y).reshape(-1))

    # Sanity: lengths match (each fault row has one label)
    assert len(fault_df) == len(fault_y_flat), \
        f"Fault states ({len(fault_df)}) and labels ({len(fault_y_flat)}) length mismatch."

    # Build set of (state_tuple, forbidden_label) pairs
    fault_pairs = { (tuple(row.astype(float)), int(lbl)) 
                    for row, lbl in zip(fault_df.to_numpy(), fault_y_flat.to_numpy()) }

    # Keep rows whose (state, label) NOT in fault_pairs
    mask = [ (tuple(row.astype(float)), int(lbl)) not in fault_pairs
             for row, lbl in zip(df.to_numpy(), y_flat.to_numpy()) ]
    mask = np.array(mask, dtype=bool)

    df_clean = df[mask].reset_index(drop=True)
    y_clean  = y_flat[mask].reset_index(drop=True)

    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"\n\t[NOTE] Ignoring {removed} conflicting training examples (faulty label!)\n")

    return df_clean, y_clean

class FaultFixer:
    def __init__(self, training_data_states, training_data_actions, mr_config, 
                 tree_config, prefix, benchmark):
        self.mr_config = mr_config # set the correct configuration
        self.tree_config = tree_config
        self.training_states = training_data_states
        self.training_actions = training_data_actions
        self.prefix = prefix
        self.benchmark = benchmark
        self.faulty_states, self.possible_actions, self.faulty_actions = pd.DataFrame(), [], pd.Series()        # initially there's no faults
        
    def set_faults(self, Fx, Aa, Fy):
        self.faulty_states = Fx
        self.possible_actions = Aa
        self.faulty_actions = Fy
    
    """Abstract base for fault fixing strategies"""
    @abstractmethod
    def fix_faults(self, policy): # Returns fixed policy (at_optimized)
        """
        Abstract base method for fault fixing strategies

        Args:
            policy (Veritas AddTree): Tree ensemble policy
        
        Given:
            training_states: States generated by imitation learning on the teacher policy
            training_actions: Actions generated by imitation learning on the teacher policy
            faulty_states: All faulty states found till now
            faulty_actions: All faulty actions found till now
            possible_actions: The suggested possible actions (applicable actions - faulty actions)
        """
        pass