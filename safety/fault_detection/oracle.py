# Calling PlaJa to collect faults and check whether policy decision is safe on given states
from utils.utils import *
from utils.plaja_interface import *
from fault_detection.helpers.faults import *
import json 

def collect_faults(
        prefix, benchmark, repair_method, tconfig, iteration, fconfig, seed=SEED, 
        evaluation_set=False, train_test_split=0.5
    ):
    """
    Sample num_starts randomly sampled start states from the start condition. Use biased policy run sampling to generate unsafe paths. Then process these unsafe paths backwards to get faults.
    """
    # HACK: A generic method of setting train_test_split and evaluation_set based on enumerability of state space
    train_test_split = 0.5 if not fconfig["enumerable"] else 1.0
    evaluation_set = False if not fconfig["enumerable"] else evaluation_set

    # HACK for blocksworld (otherwise not detected as enumerable for some reason)
    if get_game(benchmark) == "blocksworld":
        num_starts = 4139
    elif fconfig["extensive_fuzzing"]:
        num_starts = 1e6 # tmp: test with 100k starts when doing extensive fuzzing
    else:
        num_starts = NUM_STARTS

    model_file = get_model_file(prefix, benchmark)
    compact = fconfig["enumerable"] if not fconfig["extensive_fuzzing"] else True
    property_file = get_property_file(prefix, benchmark, compact, fconfig['num_starts'])
    interface_file = get_interface_file(prefix, benchmark)
    ensemble_file = get_ensemble_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                      tconfig['max_depth'], tconfig['num_trees'], 
                                      tconfig['lr'], iteration)
    if not os.path.exists(ensemble_file):
        raise FileNotFoundError(f"Ensemble file {ensemble_file} does not exist.")
    log_file_path = log_file(benchmark, prefix, repair_method, tconfig['model_type'], tconfig['max_depth'],
                                tconfig['num_trees'], tconfig['lr'], iteration, extensive_fuzzing=fconfig["extensive_fuzzing"])
    stats_file_path = stat_file(benchmark, prefix, repair_method, tconfig['model_type'], tconfig['max_depth'],
                                    tconfig['num_trees'], tconfig['lr'], iteration, extensive_fuzzing=fconfig["extensive_fuzzing"])
    for path in [log_file_path, stats_file_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sample = "inc" if not fconfig["extensive_fuzzing"] else "sample"
    use_evaluation = "--evaluation-set " if evaluation_set else ""
    generic_command = f"{PLAJA_DIR} --model-file {model_file} --seed {seed} --engine FAULT_ANALYSIS --max-time {fconfig['fa_total_time']} --ensemble-interface {interface_file} --ensemble {ensemble_file} --initial-state-enum {sample} --oracle-use finding --oracle-mode infinite --infinite-oracle {ORACLE_TYPE} --applicability-filtering 1 --fault-experiment-mode aggregate --search-dir backward --seconds-per-query {fconfig['fa_time_per_query']} --num-starts {num_starts} --distance-func avoid --save-intermediate-stats --stats-file {stats_file_path} --cache-fault-search --retain-fault-cache --retraining-train-test-split {train_test_split}"
    generic_command += f" --additional-properties {property_file} --prop 1 {use_evaluation}" if (property_file != None) else f" --prop 0 {use_evaluation}"
    if fconfig["extensive_fuzzing"] and (
        prefix in ["non_det_with_park", "non_det_no_park", "swap_unsafe", "CI", "CA"] 
    ):
        generic_command += f" --num-paths-per-start 10 "
    if fconfig["fuzzing"] == "random":
        command = generic_command + f"--rand-prob 1 --sampling-lookahead -1 >> {log_file_path}" # hardcoded to random at the moment
    elif fconfig["fuzzing"] == "probabalistic":
        command = generic_command + f"--rand-prob 0 --sampling-lookahead -1 --use-prob-sampling >> {log_file_path}" # hardcoded to random at the moment
    else:
        assert fconfig["fuzzing"] == "greedy"
        command = generic_command + f"--rand-prob 0 --sampling-lookahead -1 >> {log_file_path}" # hardcoded to random at the moment
    t = time.time()
    oor = False
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        # preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                        # more memory for fault analysis:
                        preexec_fn=limit_memory_preexec(FA_PLAJA_MEM_MIB),
                        # headroom over PlaJA's --max-time: let PlaJA stop itself
                        # and write its final stats instead of being SIGKILLed
                        timeout=fconfig.get('fa_total_time') + 600
                        )
    except subprocess.TimeoutExpired:
        print(f"\t\t- Fault collection timed out after {time.time()-t:.3f} s.")
        oor = "timeout"
    except subprocess.CalledProcessError as e:
        if e.returncode == 134:
            print(f"\t\t- Fault collection timed out after {time.time()-t:.3f} s.")
            oor = "timeout"
        elif e.returncode == 137:
            print(f"\t\t- PlaJa OOM after {time.time()-t:.3f} s.")
            oor = "oom"
        elif e.returncode == 22:
            print(f"\t\t- PlaJa OOM after {time.time()-t:.3f} s ",
                  # f"(due to imposed {DEFAULT_PLAJA_MEM_MIB / 1024} GB memory limit).")
                    f"(due to imposed {FA_PLAJA_MEM_MIB / 1024} GB memory limit).")
            oor = "oom"
        else:
            print("Command failed!")
            print("Exit code:", e.returncode)
            print("Output:", e.stdout)
            print("Error output:", e.stderr)

    # print(f"Command executed successfully: {command}")
    print(f"\t\t- fault detection completed in {time.time()-t:.3f} s.")
    return log_file_path, stats_file_path, oor


# count-faults experiment helper functions
def get_json_path_for_faulty_states(prefix, benchmark, subdir, repair_method, tconfig):
    return f"{FAULT_DIR}/{subdir}/{prefix}_{benchmark}_{repair_method}_{tconfig['max_depth']}D_{tconfig['num_trees']}T_{tconfig['lr']}LR.json"

def check_policy_on_states_plaja(prefix, benchmark, subdir, repair_method, tconfig, check_policy_not_fault=True, iteration=0):
    model_file = get_model_file(prefix, benchmark)
    property_file = get_property_file(prefix, benchmark, True, 0)
    interface_file = get_interface_file(prefix, benchmark)
    problem = get_problem(benchmark)
    _lr = f"_{tconfig['lr']}LR"

    ensemble_file = get_ensemble_file(benchmark, prefix, repair_method, tconfig['model_type'], tconfig['max_depth'], tconfig['num_trees'], tconfig['lr'], iteration)
    faulty_states_file = get_json_path_for_faulty_states(prefix, benchmark, subdir, repair_method, tconfig)
    
    _prefix = 'not_fault' if check_policy_not_fault else 'safe'
    log_file = f"{FAULT_DIR}/{subdir}/{_prefix}_{prefix}_{benchmark}_{repair_method}_{tconfig['max_depth']}D_{tconfig['num_trees']}T_{tconfig['lr']}LR.txt"
    
    for path in [log_file]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    
    count_unsafe = "" if check_policy_not_fault else " --count-unsafe-states"
    command = f"{PLAJA_DIR} --model-file {model_file} --seed 10 --engine FAULT_ANALYSIS --max-time 43200 --ensemble-interface {interface_file} --ensemble {ensemble_file} --initial-state-enum inc --oracle-use policy_action_fault_on_state --oracle-mode infinite --infinite-oracle PI3 --applicability-filtering 1 --fault-experiment-mode aggregate --search-dir backward --seconds-per-query 100 --num-starts 10000 --distance-func avoid --cache-fault-search --retain-fault-cache --additional-properties {property_file} --prop 1 --faulty-states-filepath {faulty_states_file}{count_unsafe} >> {log_file}"
    print("Log file:", log_file)
    t = time.time()
    oor = False
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        # preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                        # more memory for fault analysis:
                        preexec_fn=limit_memory_preexec(FA_PLAJA_MEM_MIB),
                        timeout=43200 + 600
                        )
    except subprocess.TimeoutExpired:
        print(f"\t\t- Fault collection timed out after {time.time()-t:.3f} s.")
        oor = "timeout"
    except subprocess.CalledProcessError as e:
        if e.returncode == 134:
            print("HERE")
            exit()
            print(f"\t\t- Fault collection timed out after {time.time()-t:.3f} s.")
            oor = "timeout"
        elif e.returncode == 137:
            print(f"\t\t- PlaJa OOM after {time.time()-t:.3f} s.")
            oor = "oom"
        elif e.returncode == 22:
            print(f"\t\t- PlaJa OOM after {time.time()-t:.3f} s ",
                  # f"(due to imposed {DEFAULT_PLAJA_MEM_MIB / 1024} GB memory limit).")
                    f"(due to imposed {FA_PLAJA_MEM_MIB / 1024} GB memory limit).")
            oor = "oom"
        else:
            print("Command failed!")
            print("Exit code:", e.returncode)
            print("Output:", e.stdout)
            print("Error output:", e.stderr)

    # print(f"Command executed successfully: {command}")
    print(f"\t\t- fault detection completed in {time.time()-t:.3f} s.")
    return log_file, oor

class FaultAnalysis:
    def __init__(self, mr_config, tree_config, fconfig, prefix, benchmark):
        self.mr_config = mr_config # set the correct configuration
        self.tree_config = tree_config
        self.fconfig = fconfig
        self.prefix = prefix
        self.benchmark = benchmark
        self.faulty_states, self.possible_actions, self.faulty_actions = \
            pd.DataFrame(), [], pd.Series()   # initially there's no faults

        
    def set_faults(self, Fx, Aa, Fy):
        self.faulty_states = Fx
        self.possible_actions = Aa        
        self.faulty_actions = Fy
        
    def get_faults(self):
        return self.faulty_states, self.possible_actions, self.faulty_actions
    
    def save_faulty_states_as_json(self, sub_dir : str):
        filename = get_json_path_for_faulty_states(self.prefix, self.benchmark, sub_dir, self.mr_config["repair_method"], self.tree_config)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(self.benchmark.startswith(name) for name in ["bouncing_ball", "stopping_car", "inverted_pendulum"]):
            states = [[int(x) if isinstance(x, int) or (isinstance(x, float) and x.is_integer()) else float(x) for x in row]
                      for row in self.faulty_states.itertuples(index=False)]
        else:
            # HACK for control benchmarks
            # --> parse the first number as a int and the rest as floats 
            states = []
            for row in self.faulty_states.itertuples(index=False):
                state = []
                for i, x in enumerate(row):
                    if i == 0:
                        state.append(int(x))
                    else:
                        state.append(float(x))
                states.append(state)

        # # states = [[int(x) if isinstance(x, int) else float(x) for x in row]
    # for row in self.faulty_states.drop_duplicates().itertuples(index=False)
#      for row in self.faulty_states.itertuples(index=False)
# ]
        seen = set()
        for state in states:
            t = tuple(state)
            # assert t not in seen, f"❌ Duplicate state found: {state}"
            seen.add(t)

        with path.open("w") as f:
            f.write('{\n  "states": [\n')
            for i, state in enumerate(states):
                line = json.dumps(state)
                if i < len(states) - 1:
                    f.write(f"    {line},\n")
                else:
                    f.write(f"    {line}\n")
            f.write("  ]\n}\n")
        return 
    
    def check_policy_not_fault_on_states(self, subdir, iteration=0, repair_method=None):
        if repair_method is None:
            repair_method = self.mr_config["repair_method"]
        logfile, oor = check_policy_on_states_plaja(
            self.prefix, self.benchmark, subdir, repair_method, self.tree_config, check_policy_not_fault=True, iteration=iteration
        )
        if oor:
            return "---", "---"
        num_states, num_not_faults = parse_policy_action_not_fault_logs(logfile)
        return num_states, num_not_faults
    
    def check_policy_safe_on_states(self, subdir, iteration=0):
        logfile, oor = check_policy_on_states_plaja(
            self.prefix, self.benchmark, subdir, self.mr_config["repair_method"], self.tree_config, check_policy_not_fault=False, iteration=iteration
        )
        if oor:
            return "---", "---"
        num_states, num_unsafe = parse_policy_safe_logs(logfile)
        return num_states, num_unsafe


    def find_faults(self, iterations): # Returns fixed policy (at_optimized)
        """
        Method for fault analysis using fuzzing

        Args:
            iterations (int): Tells us which policy needs to be loaded
        Given:
            mr_config: Which repair method we are using for correct indexing of the policy
            tree_config: Tree ensemble size
            fconfig: Fault Analysis and Verification configurations
        """
        log = {"runtime": 0, "repair_status": None}
        fa_start_time = time.time()
        faults_path, __, oor = collect_faults(self.prefix, self.benchmark, 
                                              self.mr_config["repair_method"], 
                                              self.tree_config, iterations, self.fconfig)
        log["runtime"] += time.time() - fa_start_time
        log["repair_status"] = oor
        return faults_path, log
    

    def parse_and_add_faults(
            self, policy, faults_path, sanity_check=True, preserve_non_inputs=False, duplicates_ok=False 
    ):
        """
        Parse faults and add them to create the new set of faults for repair
        """
        
        log = {"runtime": 0, "new_faults": False, "#faults": 0}
        time_other = time.time()
        fx, aa, fy = parse_faults(
            faults_path, self.benchmark, self.prefix, preserve_non_inputs
        )
        # print(f"Parsed {len(fx)} faults from {faults_path} with shape {fx.shape}.")
        if not fx.empty:
            # Check and add new faults to the existing ones
            if sanity_check:
                fx, aa, fy = manual_faults_check(policy, fx, fy, aa)
            log["#faults"] = len(fx)
            # normally we stack all faults together BUT for penalty trees or debug reasons, 
            # we might want to only keep each iteration's faults
            no_stacking = True if self.mr_config["repair_method"] == "penalty" else False
            # when retraining a new model at each iteration, 
            # faults can be repeated across iterations if previous model was not safe
            # instead, MILP repair gives a hard guarantee
            duplicates_ok = True if (
                self.mr_config["repair_method"].startswith("retraining") or duplicates_ok
            ) else False
            if preserve_non_inputs:
                ignore_final_cols = fx.shape[1] - filter_data_columns(
                    self.benchmark, self.prefix
                )
            
            self.faulty_states, self.possible_actions, self.faulty_actions, nd = \
                attach_faults(self.faulty_states, self.possible_actions, 
                              self.faulty_actions, fx, aa, fy, policy, 
                              no_stacking, duplicates_ok, 
                              ignore_final_cols=ignore_final_cols if preserve_non_inputs else 0
                )
            
            if sanity_check:
                print(f"\n\tb) Analyzing forbidden state-value pairs:")
                print(f"\t\t- detected faults: {len(self.faulty_states)}, "\
                        f"of which {len(fx)-(nd)} from current iteration " \
                        f"(+{nd} duplicates).")

            if len(fx)-nd == 0:
                if sanity_check:
                    print(f"\t\t--> no new faults found (all {len(fx)} duplicates)!")
            else:
                log["new_faults"] = True
        log["runtime"] += time.time() - time_other

        return log
    
    def load_cached_faults(self, repair_method, iteration):
        depth = self.tree_config['max_depth']
        num_trees = self.tree_config['num_trees']
        lr = self.tree_config['lr']

        # hack for comapring with old debugging loop faults and old naming scheme
        # if self.benchmark.startswith("one_way_line"):
        #     benchmark = self.benchmark.replace("one_way_line", "mod_linetrack")
        # elif self.benchmark.startswith("two_way_line"):
        #     benchmark = self.benchmark.replace("two_way_line", "race_linetrack")
        # else:
        #     benchmark = self.benchmark
        benchmark = self.benchmark
        # if self.prefix == "non_det_icy_drop_park":      prefix = "non_det_with_park"
        # elif self.prefix == "non_det_fail_dec":           prefix = "non_det_no_park"
        # else:
        #     prefix = self.prefix
        prefix = self.prefix
        # print(f"Loading faults from the old debugging loop for {benchmark} / {prefix}...")
        if prefix:
            filepath = f"{CACHED_FAULTS_DIR}/{repair_method}/{get_problem(benchmark)}/{benchmark}_{prefix}_{depth}depth_{num_trees}trees_{lr}LR_it{iteration}.txt"
        else:
            filepath = f"{CACHED_FAULTS_DIR}/{repair_method}/{get_problem(benchmark)}/{benchmark}_{depth}depth_{num_trees}trees_{lr}LR_it{iteration}.txt"

        return filepath

    def load_cached_verification_faults(self, repair_method, iteration):
        depth = self.tree_config['max_depth']
        num_trees = self.tree_config['num_trees']
        lr = self.tree_config['lr']
        if self.prefix:
            filepath = f"{CACHED_VER_DIR}/{repair_method}/{get_problem(self.benchmark)}/{self.benchmark}_{self.prefix}_{depth}depth_{num_trees}trees_{lr}LR_it{iteration}.txt"
        else:
            filepath = f"{CACHED_VER_DIR}/{repair_method}/{get_problem(self.benchmark)}/{self.benchmark}_{depth}depth_{num_trees}trees_{lr}LR_it{iteration}.txt"
        return filepath
    
    def consolidate_faults(self):
        """Using pandas to consolidate faults. 
           This returns all faulty actions found over several iterations 
           for the same faulty state.
        """
        if self.faulty_states.empty:
            return self.faulty_states, self.possible_actions, self.faulty_actions
        
        # Combine everything into one DataFrame
        df = self.faulty_states.copy()
        df['possible_action'] = self.possible_actions
        df['faulty_action'] = self.faulty_actions
        
        # Group by all state columns
        grouped = df.groupby(list(self.faulty_states.columns))
        
        # Aggregate: collect unique actions
        def collect_unique(series):
            return list(set(series))
        
        result = grouped.agg({
            'possible_action': collect_unique,
            'faulty_action': collect_unique
        }).reset_index()
        
        # Split back into components
        faulty_states = result[self.faulty_states.columns]
        possible_actions = result['possible_action'].tolist()
        faulty_actions = pd.Series(result['faulty_action'].tolist())
        
        return faulty_states, possible_actions, faulty_actions
    
    def test_faults_hold(self, policy):
        # faulty_states, possible_actions, faulty_actions = self.consolidate_faults()
        # mismatches, log = manual_faults_check(policy, self.faulty_states, self.faulty_actions, 
        #                           self.possible_actions, sanity_check = False)
        # return log["#faults"], log["#matches"], mismatches
        log = manual_faults_check(policy, self.faulty_states, self.faulty_actions, 
                                  self.possible_actions, sanity_check = False)
        return log["#faults"], log["#matches"]
        