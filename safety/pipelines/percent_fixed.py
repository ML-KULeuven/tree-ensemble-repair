import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policy.tree_ensemble import EnsembleLearner
from fault_detection.oracle import FaultAnalysis
from utils.utils import filter_data_columns, get_ensemble_file, get_game, get_problem
from pipelines.count_faults import AnalyzeLoop
import pandas as pd
from results_processors.processor_factory import *
from utils.utils import (RESULT_DIR, log_file)

def get_result_file_name(tree_config, mr_config, fa_config):        
    lr_string = f"_{tree_config['lr']}" if tree_config['model_type'] == "xgb" else ""
    tree_string = f"{tree_config['model_type']}_{tree_config['num_trees']}_"\
                    f"{tree_config['max_depth']}{lr_string}"
    
    mr_string = f"{mr_config['iterations']}it" \
                    f"_{mr_config['repair_method']}" 
                    # f"_flonly{mr_config['fl_only']}" 
    
    # compact_string = "_compact" if fa_config['compact'] else ""
    verifier_string = f"_{fa_config['verifier']}" if fa_config['run_verification'] else ""
    fa_string = f"{fa_config['num_starts']}N{verifier_string}" #{compact_string}"

    return f"{RESULT_DIR}/results-{tree_string}-{mr_string}-{fa_string}.txt"


def read_plaja_result(filename, benchmark, prefix):
    # Return the final iteration
    with open(filename, 'r') as f:
        data = f.readlines()
    for line in data:
        parts = line.strip().split()
        if len(parts) < 11:
            continue
        benchmark_full, __, block_nn, num_faults, added_trees, block_test, block_orig, block_mid, block_final, block_baseline, block_runtime, block_ver = parts
        if block_ver == "0_None":
            block_ver = "N/A"
        if benchmark_full == f"{benchmark}_{prefix}" or benchmark_full == f"{benchmark}":
            tot_its, testing_status = block_test.split("_")
            if block_test == "0_None":
                testing_status = "N/A"
                tot_its = False
            try:
                verification_status = block_ver.split('_')[1]
                ver_its = block_ver.split('_')[0]
            except IndexError:
                verification_status = "N/A"
                ver_its = 0
            gur_its = int(tot_its) - int(ver_its) if tot_its else 0
            try:
                final_it = int(tot_its)-1 if int(tot_its) else 0
            except ValueError:
                final_it = 0
            return verification_status, final_it
    return False, False

def find_first_verified_policy(benchmark, prefix, repair_method):
    from utils.utils import VERIF_DIR
    ver_log_dir = f"{VERIF_DIR}/{repair_method}/{get_problem(benchmark)}"

    # list files in ver_log_dir
    start_string = f"{benchmark}_{prefix}" if prefix else benchmark
    print(f"Looking for verification logs in {ver_log_dir} starting with {start_string}...")
    try:
        files = [f for f in os.listdir(ver_log_dir) if f.startswith(start_string)]
    except FileNotFoundError:
        # the directory does not exist, meaning no policies were verified for this benchmark + repair method
        print(f"No verification logs found for {benchmark} with prefix {prefix} and repair method {repair_method}.")
        return None
    its = []
    for f in files:
        if not f.endswith(".txt"):
            continue
        its.append(int(f.split("_")[-1].replace(".txt", "").replace("it", "")))
    first_it = min(its) if its else None
    # print(f"First verified iteration for {benchmark} with prefix {prefix} and repair method {repair_method}: {first_it}")
    return first_it 

REPAIR_METHODS = ["milp-l1", "retraining-augment", "retraining-loss", "penalty"]

class PercentFixedFaults:
    def __init__(self, prefix, benchmark, mr_config, tree_config, fa_config, res_file=None):
        self.prefix = prefix
        self.benchmark = benchmark
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.fa_config = fa_config
        self.log = {}
        self.results_processor: Processor = \
            create_results_processor(
                "percent_fixed", mr_config=self.mr_config, tree_config=self.tree_config, fa_config={}
            )
        self.ensemble_learner = EnsembleLearner(
            self.mr_config, self.tree_config, {}, self.prefix, self.benchmark, debug=False
        )

    def load_extensive_fuzzing_faults(self):
        self.fa_config["extensive_fuzzing"] = True
        fault_analysis_ext = FaultAnalysis(self.mr_config, self.tree_config, self.fa_config, self.prefix, self.benchmark)
        # Get the file where extensive fuzzing faults (iteration 0) will be stored
        faults_file = log_file(
            self.benchmark, self.prefix,  self.mr_config["repair_method"], self.tree_config['model_type'], self.tree_config['max_depth'], 
            self.tree_config['num_trees'], self.tree_config['lr'], iteration=0, extensive_fuzzing=True
        )
        previous_run = False
        if os.path.exists(faults_file):
            # parse previously detected faults for extensive fuzzing
            parsing_logs = fault_analysis_ext.parse_and_add_faults(
                self.original_policy, faults_file, preserve_non_inputs=True
            )
            if parsing_logs["new_faults"]:
                previous_run = True
        if not previous_run:
            # UNCOMMENT TO RUN EXTENSIVE FUZZING IF NO PREVIOUS RUN FOUND 
            # print(f"Warning: No faults from extensive fuzzing found. Running fuzzing.")
            # # run fault analysis with extensive fuzzing
            # faults_file, __ = fault_analysis_ext.find_faults(iterations=0) 
            # parsing_logs = fault_analysis_ext.parse_and_add_faults(
            #     self.original_policy, faults_file, preserve_non_inputs=True
            # )
            print(f"Warning: No faults from extensive fuzzing found.")
        faults_extensive_fuzzing = fault_analysis_ext.get_faults()
        print(f"Found/Loaded {len(faults_extensive_fuzzing[0])} faults from extensive fuzzing")
        return faults_extensive_fuzzing
    

    def get_last_policy_iteration(self, repair_method):
        mr_config_ext = self.mr_config.copy()
        mr_config_ext["repair_method"] = repair_method
        res_file = get_result_file_name(self.tree_config, mr_config_ext, self.fa_config)
        if not os.path.exists(res_file):
            raise FileNotFoundError(f"Result file not found: {res_file}")
        ver_status, final_iteration = read_plaja_result(res_file, self.benchmark, self.prefix)
        print(f"Final iteration detected (from result file): {final_iteration}")
        return final_iteration

    def run_percent_fixed_analysis(self, repair_method, final_iteration):
        # Run fault analysis on the provided faults Fx, Aa, Fy using the final policy at final_iteration
        states, not_faults = self.fault_analysis.check_policy_not_fault_on_states("percent_fixed", final_iteration, repair_method)
        # print(f"States checked for Percent Fixed analysis: {states}, Not Faults: {not_faults}")
        if states == "---":
            print(f"PlaJA check OOR.")
            self.log[repair_method] = {
                "percent_fixed": float('nan'),
                "num_fixed": 0,
                "num_total": 0
            }
            return
        percent_fixed = (not_faults / states) * 100 if states > 0 else 0.0
        print(f"Percent Fixed Faults at iteration {final_iteration}: {percent_fixed:.2f}% ({not_faults}/{states})")
        self.log[self.mr_config["repair_method"]] = {
            "percent_fixed": round(percent_fixed, 2), # round to 2 decimals
            "num_fixed": not_faults,
            "num_total": states
        }

    def run(self, run_for_no_ver=False):
        # --- 1. Find final iteration using read_plaja_result  ---
        self.original_policy = self.ensemble_learner.retrieve_policy(self.mr_config["repair_method"], 0) # This is the original policy!
        faults_extensive_fuzzing = self.load_extensive_fuzzing_faults() # Get faults from extensive fuzzing for the original policy
        faults_ext = faults_extensive_fuzzing[0].iloc[:, :]
        faults_ext['forbidden_action'] = faults_extensive_fuzzing[2]
        print(f"#Faults (extensive fuzzing): {len(faults_extensive_fuzzing[0])}")

        # if no faults from extensive fuzzing, we cannot run percent fixed analysis, so we exit
        if len(faults_extensive_fuzzing[0]) == 0:
            print("No faults from extensive fuzzing, cannot run Percent Fixed analysis. Exiting.")
            # write an entry with 0% fixed and 0 faults to avoid missing data points in the results         
            import numpy as np   
            self.log["milp-l1"] = {
                "percent_fixed": np.nan,
                "num_fixed": 0,
                "num_total": 0
            }
            self.log["retraining-augment"] = {
                "percent_fixed": np.nan,
                "num_fixed": 0,
                "num_total": 0
            }
            self.log["retraining-loss"] = {
                "percent_fixed": np.nan,
                "num_fixed": 0,
                "num_total": 0
            }
            self.results_processor.write_results(self.log, prefix=self.prefix, benchmark=self.benchmark)
            return
        # We always remove duplicates 
        assert faults_ext.duplicated().sum() == 0, "Duplicate faults within the extensive fuzzing faults dataframe?!" 
        faults_ext["applicable_actions"] = faults_extensive_fuzzing[1]
        faults_ext["applicable_actions"] = faults_ext["applicable_actions"].apply(tuple)
        MERGE_COLS = [c for c in faults_ext.columns if c != "applicable_actions"]
        initial_size = len(faults_ext)

        # UNCOMMENT TO FILTER OUT FAULTS FOUND IN DEBUGGING LOOP 
        # (i.e., keep only faults from extensive fuzzing that are disjoint from debugging loop faults)
        for repair_method in REPAIR_METHODS:
            # --- 2. Load faults from debugging loop ---
            analyze_loop = AnalyzeLoop(self.prefix, self.benchmark, self.mr_config, self.tree_config)
            analyze_loop.load_debugging_loop_faults(repair_method)
            faults_debugging_loop = analyze_loop.fault_analysis.get_faults()
            print(f"Loaded {len(faults_debugging_loop[0])} faults from debugging loop")
            assert faults_extensive_fuzzing[0].shape[1] == faults_debugging_loop[0].shape[1], \
            "Faults dataframes have different number of columns. Something is wrong!"

            # TMP log separately #Faults for penalty trees (main log is wrong)
            # --> not needed anymore as the issue was fixed, kept this for consistency with KR26 run
            # if repair_method == "penalty":
            #     rf =f"{RESULT_DIR}/penalty_trees_faults.txt"
            #     with open(rf, "a") as f:                    
            #         f.write(f"\n{self.benchmark}_{self.prefix} {len(faults_debugging_loop[0])}")
                    # exit()

            faults_loop = faults_debugging_loop[0].iloc[:, :]
            faults_loop['forbidden_action'] = faults_debugging_loop[2]
            print(f"#Faults (debugging loop) {repair_method}: {len(faults_debugging_loop[0])}")
            assert faults_loop.duplicated().sum() == 0, "Duplicate faults within the debugging loop faults dataframe?!" 

            faults_loop["applicable_actions"] = faults_debugging_loop[1]
            faults_loop["applicable_actions"] = faults_loop["applicable_actions"].apply(tuple)

            # when we check for duplicates, need to ignore some columns
            ignore_final_cols = faults_debugging_loop[0].shape[1] - filter_data_columns(
                        self.benchmark, self.prefix
            )
            
            # need to exclude non-input columns from the merge to find disjoint faults
            # otherwise we might keep faults that are actually the same but have different non-input columns 
            comparison_columns = [col for col in MERGE_COLS if col not in ["forbidden_action"]]
            comparison_columns = [col for col in comparison_columns[:-ignore_final_cols]] + ["forbidden_action"] 
            # left merge to find which rows in faults_ext appear in faults_loop
            # TO ALSO CONSIDER NON-INPUT COLUMNS IN THE MERGE, JUST MERGE OVER ALL COLS
            merged = pd.merge(
                faults_ext,
                faults_loop[comparison_columns],  
                on=comparison_columns,  # not all MERGE_COLS!
                how="left",
                indicator=True,
            )

            # keep only rows that exist only in faults_ext
            faults_ext = merged[merged["_merge"] == "left_only"]

            # optional: drop the merge indicator column
            faults_ext = faults_ext.drop(columns="_merge")


        final_size = len(faults_ext)
        assert final_size <= initial_size, "Something went wrong with finding disjoint faults. More disjoint faults than original faults?!"


        state_cols = [ c for c in faults_ext.columns if c not in ["forbidden_action", "applicable_actions"]]

        Fx = faults_ext[state_cols].copy()
        Fy = faults_ext["forbidden_action"].copy()
        Aa = faults_ext["applicable_actions"].tolist()
        print(f"\nAfter removing debugging loop faults, {final_size} disjoint faults remain for Percent Fixed analysis ({initial_size - final_size} faults removed, {final_size} faults remain).\n")

        if not run_for_no_ver:
            for repair_method in REPAIR_METHODS:
                print(f"\n *** Running Percent Fixed Faults analysis for repair method: {repair_method} ***")
                final_iteration = self.get_last_policy_iteration(repair_method)
                print(f"Final iteration for {repair_method}: {final_iteration}")
                
                self.mr_config["repair_method"] = repair_method
                self.fault_analysis = FaultAnalysis(self.mr_config, self.tree_config, self.fa_config, self.prefix, self.benchmark)
                self.fault_analysis.set_faults(Fx, Aa, Fy)
                self.fault_analysis.save_faulty_states_as_json("percent_fixed")
                self.run_percent_fixed_analysis(repair_method, final_iteration)
        else:
            # specifically run %Fixed on the 1st policy where we call verifier in our debugging loop
            # (need to do this separately as we log in a different file...)
            for repair_method in REPAIR_METHODS:
                print(f"\n *** Running Percent Fixed Faults analysis for repair method: {repair_method} (NO VERIFICATION MODE) ***")
                nover_it = find_first_verified_policy(self.benchmark, self.prefix, repair_method)
                if not nover_it:
                    print(f"No verified policy found for {repair_method}. Logging NaN.")
                    self.log[repair_method] = {
                        "percent_fixed": float('nan'),
                        "num_fixed": float('nan'),
                        "num_total": float('nan')
                    }
                    continue
                print(f"First verification iteration for {repair_method}: {nover_it}")

                self.mr_config["repair_method"] = repair_method
                self.fault_analysis = FaultAnalysis(self.mr_config, self.tree_config, self.fa_config, self.prefix, self.benchmark)
                self.fault_analysis.set_faults(Fx, Aa, Fy)
                self.fault_analysis.save_faulty_states_as_json("percent_fixed")
                self.run_percent_fixed_analysis(repair_method, nover_it)


        print(self.log)
        self.results_processor.write_results(self.log, prefix=self.prefix, benchmark=self.benchmark) 
        
