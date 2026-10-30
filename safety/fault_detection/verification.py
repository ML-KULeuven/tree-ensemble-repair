# Call plaja to verify a policy. Run fault analysis on the unsafe path if verification fails. Parse the logs and return the faults found.

from abc import ABC, abstractmethod
from utils.utils import *
from utils.plaja_interface import *
from fault_detection.helpers.parse_ver_logs import *
from fault_detection.helpers.faults import *

def run_verification_via_fault_analysis(prefix, benchmark, repair_method, iteration, tconfig, fconfig):
    """
    Verify a given policy using PA CEGAR. Then run fault analysis on the unsafe path we find.
    """
    model_file = get_model_file(prefix, benchmark)
    property_file = get_property_file(prefix, benchmark, compact=True)
    interface_file = get_interface_file(prefix, benchmark)
    ensemble_file = get_ensemble_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                      tconfig['max_depth'], tconfig['num_trees'], 
                                      tconfig['lr'], iteration)
    if not os.path.exists(ensemble_file):
        raise FileNotFoundError(f"Ensemble file {ensemble_file} does not exist.")
    log_file_path = verification_log_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                      tconfig['max_depth'], tconfig['num_trees'], 
                                      tconfig['lr'], iteration)
    stats_file_path = verification_stat_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                      tconfig['max_depth'], tconfig['num_trees'], 
                                      tconfig['lr'], iteration)
    for path in [log_file_path, stats_file_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    verification_parameters = "--check-for-pa-terminal-states true --terminal-as-preds true --set-pa-goal-objective-terminal true --split-point bs --selection-refinement all --entailment-mode entailment-only --check-policy-spuriousness false --incremental-search true --ensemble-sat veritas"
    oracle_parameters = f"--initial-state-enum sample --oracle-use finding --oracle-mode infinite --infinite-oracle {ORACLE_TYPE} --applicability-filtering 1 --fault-experiment-mode aggregate --search-dir backward --seconds-per-query {fconfig['fa_time_per_query']} --num-starts 1 --distance-func avoid --rand-prob 0 --sampling-lookahead -1 --cache-fault-search --retain-fault-cache"
    # command = f"{PLAJA_DIR} --model-file {model_file} --use-cegar --seed {iteration} --prop 0 --engine FAULT_ANALYSIS --max-time {fconfig['verification_total_time']} --ensemble-interface {interface_file} --ensemble {ensemble_file} --save-intermediate-stats --stats-file {stats_file_path} {oracle_parameters} {verification_parameters} >> {log_file_path}"
    command = f"{PLAJA_DIR} --model-file {model_file} --use-cegar --seed {iteration} --additional-properties {property_file} --prop 1 --engine FAULT_ANALYSIS --max-time {fconfig['verification_total_time']} --ensemble-interface {interface_file} --ensemble {ensemble_file} --save-intermediate-stats --stats-file {stats_file_path} {oracle_parameters} {verification_parameters} >> {log_file_path}"

    t = time.time()
    oor = False
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        # preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                        preexec_fn=limit_memory_preexec(FA_PLAJA_MEM_MIB),
                        # headroom over PlaJA's --max-time (see oracle.py)
                        timeout=fconfig.get('verification_total_time') + 600)
    except subprocess.TimeoutExpired:
        oor = "TIMEOUT"
        print(f"\t- Verification TIMEOUT after {time.time()-t:.3f} s.")
    except subprocess.CalledProcessError as e:
        if e.returncode in [134, 137]:
            oor = "TIMEOUT" if np.isclose(time.time()-t, fconfig['verification_total_time'],
                    atol=60) else "OOM"
            print(f"\t- Verification {oor} after {time.time()-t:.3f} s.")
            print(e)
        elif e.returncode == 22:
            oor = "OOM"
            print(f"\t- Verification OOM after {time.time()-t:.3f} s ",
                  # f"(due to imposed {DEFAULT_PLAJA_MEM_MIB / 1024} GB memory limit).")
                  f"(due to imposed {FA_PLAJA_MEM_MIB / 1024} GB memory limit).")
        else:
            print("Command failed!")
            print("Exit code:", e.returncode)
            print("Output:", e.stdout)
            print("Error output:", e.stderr)

    # print(f"Command executed successfully: {command}")
    print(f"\t- Verification completed in {time.time()-t:.3f} s.")
    return log_file_path, stats_file_path, oor


def create_verifier_factory(verification_method, **kwargs):
    if verification_method == "plaja":
        return PPAVerifier(**kwargs)
    else:
       raise ValueError(
            f"Unknown fixer type: {verification_method}. "
            f"Available: ['plaja']"
        ) 

class Verifier:
    def __init__(self, mr_config, tree_config, fconfig, prefix, benchmark):
        self.mr_config = mr_config # set the correct configuration
        self.tree_config = tree_config
        self.fa_config = fconfig
        self.prefix = prefix
        self.benchmark = benchmark
        
    def set_faults(self, Fx, Aa, Fy):
        self.faulty_states = Fx
        self.possible_actions = Aa
        self.faulty_actions = Fy
        
    def get_faults(self):
        return self.faulty_states, self.possible_actions, self.faulty_actions
    
    @abstractmethod
    def verify_policy(self, iterations): # Returns fixed policy (at_optimized)
        """
        Abstract base method for verification followed by fault fixing

        Args:
            iterations (int): Tells us which policy needs to be loaded
        
        Given:
            mr_config: Which repair method we are using for correct indexing of the policy
            tree_config: Tree ensemble size
            fconfig: Fault Analysis and Verification configurations
        """
        pass
    
    def parse_verification_logs_and_add_faults(self, iterations, verification_log, ver_status):
        log = {
            "ver_status": ver_status,
            "runtime": 0,
            "ver_faults": False,
            "repair_status": None
        }
        # check verification result
        if ver_status == "SAFE":
            # [S1] Scenario no. 1: model is proved safe
            print(f"\n\n[{iterations}-RES] Model PROVABLY safe: hooray!!! ***")
            return log

        elif ver_status == "TIMEOUT":
            # [S2] Scenario no. 2: model seems safe but verification times out
            print(f"\n\n[{iterations}-RES] Model PROBABLY safe but verification timed out.")
            return log

        elif ver_status == "OOM":
            # [S3] Scenario no. 3: model seems safe but verification runs out of memory
            print(f"\n\n[{iterations}-RES] Model PROBABLY safe but verification OOM.")
            return log
        else:
            # model is unsafe according to verification!
            # check if verification also found new concrete faults, and attach them
            parse_time = time.time()
            fx, aa, fy = parse_faults(verification_log, self.benchmark, self.prefix)
            no_stacking = True if self.mr_config["repair_method"] == "penalty" else False
            if not fx.empty:
                self.faulty_states, self.possible_actions, self.faulty_actions = \
                    attach_verification_fault(None, fx, aa, fy, self.faulty_states, 
                                              self.possible_actions, self.faulty_actions,
                                                no_stacking=no_stacking
                    )
                log["ver_faults"] = True

            log["runtime"] += time.time() - parse_time
            
            if not log.get("ver_faults", False):
                # [S5] Scenario no. 5: model unsafe but no new faults found
                #      --> no safe policy can exist!
                print(f"\n\n[{iterations}-RES] Model presumed safe but verification found unsafety. " \
                        f"Detected unsafe path without concrete ways to fix it " \
                        f"--> no safe policy can exist: terminating...")
                log["ver_status"] = "UNSAFE"
                return log

            # [S4] Scenario no. 4: model seemed safe but verification finds unsafety
            print(f"\n\n[{iterations}-RES] Model presumed safe but found unsafe ", \
                f"({len(fx)} new concrete fault found): restarting...")
            return log                
    
class PPAVerifier(Verifier):
    def verify_policy(self, iterations):
        logs = {"runtime": 0, "status": None}
        ver_time = time.time()
        faults_dir, __, oor = run_verification_via_fault_analysis(self.prefix, self.benchmark, 
                                                                        self.mr_config["repair_method"],
                                                                        iterations, self.tree_config, self.fa_config)
        ver_status = parse_verification_log(faults_dir) if not oor else oor
        logs["runtime"] += time.time() - ver_time
        logs["status"] = ver_status if not oor else oor
        return faults_dir, logs