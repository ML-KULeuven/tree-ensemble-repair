from policy.tree_ensemble import EnsembleLearner
from fault_detection.oracle import FaultAnalysis
from utils.utils import RESULT_DIR, filter_data_columns
import os
import numpy as np
import pandas as pd
from itertools import combinations
from results_processors.processor import Processor
from results_processors.processor_factory import create_results_processor

from fault_fixing_algorithms.fault_fixer import FaultFixer
from fault_fixing_algorithms.fixer_factory import create_fault_fixer


def hash_fault(state, action):
    """Create a hashable representation of (row, value) combination"""
    state_tuple = tuple(state.fillna('NaN').astype(int).values)
    return (state_tuple, action)

def get_set_of_faults(faulty_states: pd.DataFrame, faulty_actions: pd.Series):
    """Convert (df, series) to a dictionary of hashable combinations"""
    faults = {}
    for idx, state in faulty_states.iterrows():
        action = faulty_actions[idx]
        fault_hash = hash_fault(state, action)
        faults[fault_hash] = idx  # Store index for reference
    return set(faults.keys())

class AnalyzeLoop:
    def __init__(self, prefix, benchmark, mr_config, tree_config, res_file=None):
        self.prefix = prefix
        self.benchmark = benchmark
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.log = {}
        self.results_processor: Processor = \
            create_results_processor(
                "count_faults", mr_config=self.mr_config, tree_config=self.tree_config, fa_config={}
            )
        self.ensemble_learner = EnsembleLearner(
            self.mr_config, self.tree_config, {}, self.prefix, self.benchmark, False
        )
        self.policy = self.ensemble_learner.retrieve_policy("milp-l1", 0) # This is the original policy!

    # Fault loading function
    def load_debugging_loop_faults(self, repair_method):
        self.fault_analysis = FaultAnalysis(self.mr_config, self.tree_config, {}, 
                                      self.prefix, self.benchmark)
        
        # NOTE: what if we have more than 1000 iterations? HACK for now...
        missing_count = 0
        duplicates_ok = (repair_method != "milp-l1")  # retraining may have duplicates
        # print(f"DEBUG Duplicates allowed in stacking: {duplicates_ok} -- repair method: {repair_method}")
        for iteration in range(0, 1001, 1):
            faults_path = self.fault_analysis.load_cached_faults(repair_method, iteration)
            faults_from_verification = False
            # print(faults_path)
            print(f"Trying to load faults from iteration {iteration} for method {repair_method}...")
            if not os.path.exists(faults_path):
                print("Faults path not found:", faults_path)

                # check if instead we have a verification faults file
                # from utils.utils import verification_log_file
                # ver_log_path = verification_log_file(
                #     self.benchmark, self.prefix, repair_method, 
                #     self.tree_config['model_type'], self.tree_config['max_depth'],
                #     self.tree_config['num_trees'], self.tree_config['lr'], iteration)
                ver_log_path = self.fault_analysis.load_cached_verification_faults(
                    repair_method, iteration
                )
                if os.path.exists(ver_log_path):
                    print("Loading faults from verification log:", ver_log_path)
                    faults_path = ver_log_path
                    faults_from_verification = True
                else:
                    print("Verification faults path not found:", ver_log_path)
                    missing_count += 1
                    # exit loop after 3 consecutive missing files
                    if missing_count == 3:
                        break
                    continue
            missing_count = 0  # Reset missing count if file is found

            if not faults_from_verification:
                fx_before = len(self.fault_analysis.faulty_states)
                __ = self.fault_analysis.parse_and_add_faults(
                    self.policy, faults_path, sanity_check=False, preserve_non_inputs=True, duplicates_ok=duplicates_ok
                )
                fx_after = len(self.fault_analysis.faulty_states)
                # print(f"{fx_after - fx_before} new faults loaded from fault analysis (#it {iteration})")
                # print("Faults shape:", self.fault_analysis.faulty_states.shape)
            else:
                # manually parse verification fault 
                # (copied from fault_detection/verification.py)
                from fault_detection.helpers.faults import (
                    parse_faults, 
                    attach_verification_fault
                )
                fx, aa, fy = parse_faults(
                    ver_log_path, self.benchmark, self.prefix, preserve_non_inputs=True
                )
                if not fx.empty:
                    faulty_states, possible_actions, faulty_actions = \
                        self.fault_analysis.get_faults()
                    
                    ignore_final_cols = faulty_states.shape[1] - filter_data_columns(
                        self.benchmark, self.prefix
                    )
                    faulty_states, possible_actions, faulty_actions = \
                        attach_verification_fault(
                            None, fx, aa, fy, faulty_states, possible_actions, 
                            faulty_actions, duplicates_ok=duplicates_ok, 
                            ignore_final_cols=ignore_final_cols
                        )
                    self.fault_analysis.set_faults(  
                        faulty_states, possible_actions, faulty_actions
                    )
                # print(f"{len(fx)} fault loaded from verification (#it {iteration})")

    # a slightly different function loading only fault from iteration 0
    def load_iteration_0_faults(self, repair_method):
        self.fault_analysis0 = FaultAnalysis(
            self.mr_config, self.tree_config, {}, self.prefix, self.benchmark) 
        faults_path = self.fault_analysis.load_cached_faults(repair_method, 0)
        if not os.path.exists(faults_path):
            # print("Faults Path not found:", faults_path)
            return
        _ = self.fault_analysis0.parse_and_add_faults(self.policy, faults_path, sanity_check=False)
        print(f"{len(self.fault_analysis0.faulty_states)} faults loaded from iteration 0")

    def run(self):
        print("Running loop analysis")
        faults_per_method = {}
        self.log["same_faults"] = {} # added to avoid KeyError
        for repair_method in ["milp-l1", "retraining-loss", "retraining-augment"]:
            print(f"\n *** {repair_method} *** \n")
            self.mr_config["repair_method"] = repair_method
            self.fault_analysis = FaultAnalysis(self.mr_config, self.tree_config, {}, 
                                                self.prefix, self.benchmark) 
                                                # reset engine with correct repair method

            self.load_debugging_loop_faults(repair_method)

            print(f"Done! Collected {len(self.fault_analysis.faulty_states)} faults for method {repair_method}")
            # exit()
    
            # (!) RQ1 (!): 
            #   How many faults found in later iterations are actually not faults in the original policy?
            #   --> This checks whether we introduce mistakes into the policy through our repair
            self.fault_analysis.save_faulty_states_as_json("count_faults")
            # print(f"Now we have {len(self.fault_analysis.faulty_states)} faults for method {repair_method}")
   
            states, not_pi_faults = self.fault_analysis.check_policy_not_fault_on_states("count_faults")
            if repair_method not in self.log:
                self.log[repair_method] = {}
            print("States and how many weren't faults", states, not_pi_faults)

            self.log[f"{repair_method}"]["#states"] = states
            self.log[f"{repair_method}"]["#NotPiFaults"] = not_pi_faults
            
            # (!) RQ2 (!): On how many faults is the original policy actually safe?
            states2, num_unsafe = self.fault_analysis.check_policy_safe_on_states("count_faults")
            # states2, num_unsafe = states, np.nan # TMP OFF
            print("States and how many were safe", states2, states2 - num_unsafe)
            assert states == states2, 'Mismatch in number of states. Something is wrong!'
            self.log[f"{repair_method}"]["#SafePi"] = states2 - num_unsafe 

            # extract num. faults from iteration 0 for this method
            self.load_iteration_0_faults(repair_method)
            it0_faults = self.fault_analysis0.get_faults() if hasattr(self, "fault_analysis0") else (pd.DataFrame(), pd.Series(), pd.Series())
            it0_num_faults = len(it0_faults[0]) if not it0_faults[0].empty else 0
            # hack for Follow Car (bug)
            # if num_unsafe < it0_num_faults:
            #     print(f"Warning: Num. Unsafe = {num_unsafe} is less than Num. Faults (it0)= {it0_num_faults}. Stopping.")
            #     num_unsafe = np.nan
            #     it0_num_faults = np.nan
            # NOTE: it fails for Follow Car: @TODO CHECK!!!
            assert num_unsafe >= it0_num_faults, \
                f"Num. Unsafe = {num_unsafe}, Num. Faults (it0)= {it0_num_faults}. This shouldn't happen!"
            
            # Store faults for further processing
            faulty_states, _, faulty_actions = self.fault_analysis.get_faults()
            faults_per_method[repair_method] = get_set_of_faults(faulty_states, faulty_actions)
            
        # (!) RQ3 (!): 
        #   How many faults are found by multiple repair approaches?
        results = []
        print("Here:", faults_per_method.keys())

        for (method1, faults1), (method2, faults2) in combinations(faults_per_method.items(), 2):
            # compute intersection of faults sets from each pair of methods
            common_faults = faults1.intersection(faults2)
            common_count = len(common_faults)
            results.append({
                'method_1': method1,
                'method_2': method2,
                'common_count': common_count,
            })
            print(f"{method1} vs {method2}: {common_count} common combinations")
            
            self.log["same_faults"][f"{method1}_{method2}"] = common_count

        print(self.log)
        
        # Save results in a log file
        self.results_processor.write_results(self.log, prefix=self.prefix, benchmark=self.benchmark) 
        return 

