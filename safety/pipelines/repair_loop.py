# This is our entire model repair loop
# Learn a tree ensemble policy. Then, an inner loop to run fault analysis and repair until no more faults are found. Then, an outer loop which attempts to run verification and certify the policy as safe.

from utils.utils import *
from fault_detection import *
from fault_fixing_algorithms.fault_fixer import FaultFixer
from fault_fixing_algorithms.fixer_factory import create_fault_fixer
from policy.tree_ensemble import EnsembleLearner
from fault_detection.oracle import FaultAnalysis
from fault_detection.verification import Verifier, create_verifier_factory
from fault_fixing_algorithms.baseline import Baseline
from results_processors.processor import Processor
from results_processors.processor_factory import create_results_processor
from fault_detection.helpers.faults import attach_faults, parse_faults


class WalltimeExceeded(Exception):
    """Raised by the SIGTERM handler when the job's walltime wrapper fires."""
    pass

try:
    from gurobipy import GurobiError
except ImportError:     # environments without gurobi (e.g. results parsing)
    class GurobiError(Exception):
        pass

class RepairLoop:
    def __init__(self, prefix, benchmark, mr_config, tree_config, fa_config, debug, debug_iteration):
        self.prefix = prefix
        self.benchmark = benchmark
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.fa_config = fa_config
        self.debug = debug
        self.debug_iteration = debug_iteration
        self.log = {
            "#its": 0,                          # total number of iterations
            "#faults": 0,                       # total number of faults found
            "cumulated_faults": 0,              # manual sum of faults found across iterations 
                                                # (specific for penalty trees where we do not stack faults)
            "#trees": 0,                        # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "ver_faults": False,                # new fault found with verification?
            "#ver_its": 0,                      # number of verification iterations
            "ver_status": None,                       
            # "retrain_lambda": 1.0,              # lambda used in retraining-loss
            "retrained_safe": True,            # start with retrained model being safe
            "runtimes": {
                "tot": 0.0,
                "performance_origin": 0.0,
                "performance_fa_only": 0.0,
                "performance_final": 0.0,
                "fa": 0.0,
                "fix": 0.0,                     # Time for just fixing the faults
                "repair": 0.0,                  # Time for repair loop as a whole (processing faults + fixing them)
                "ver": 0.0,
                "other": 0.0,
                "penalty_trees": 0.0,
                "performance_penalty_trees": 0.0
            },
            # "repair_specs": {} # TODO
            "perf_teacher": (np.nan, np.nan, np.nan, np.nan),             
            "perf_origin": (np.nan, np.nan, np.nan, np.nan),
            "perf_fa_only": (np.nan, np.nan, np.nan, np.nan),
            "perf_final": (np.nan, np.nan, np.nan, np.nan),
            "perf_penalty_trees": (np.nan, np.nan, np.nan, np.nan),
            "orig_fidelity": np.nan,
            "final_fidelity": np.nan
        }
        self.results_processor: Processor = \
            create_results_processor("repair", mr_config=self.mr_config, 
                                     tree_config=self.tree_config, 
                                     fa_config=self.fa_config
            )
        self.tot_time = time.time()
        # with penalty trees repair, need to track all faults to count them correctly 
        # --> fault analysis does NOT stack faults across iterations, so count is off
        self.all_faults = {
            "x": pd.DataFrame(),
            "aa": [],
            "y": pd.Series(dtype=object)
        }
        
    def check_fidelity(self) -> float:
        if len(self.training_states) == 0:
            return "bad_data"
        yhat = np.argmax(self.policy.predict(self.training_states), axis=1)
        accuracy = accuracy_score(self.training_actions, yhat)
        return round(accuracy, 3)

        
    def initialize_policy(self, cache = True, unsafe_approximation = False):
        """Initialize the run by loading tree ensemble policy or
           training a new policy through imitation learning from NN teacher data. 
           If NN teacher is unavailable, then the teacher is trained using RL.
        Args:
            cache (bool, optional): 
              Load preexisting policy if it exists or train a new one. Defaults to True.
            unsafe_approximation (bool, optional): 
              Should be set to true on benchmarks where computing unsafety exactly
              will run OOM (blocksworld). Defaults to False.
        """
        print(f"\n(i) Loading/training original tree ensemble model...")
        time_training = time.time()
        self.ensemble_learner = EnsembleLearner(self.mr_config, self.tree_config, 
                                                self.fa_config, self.prefix, 
                                                self.benchmark, self.debug
                                )
        self.training_states, self.training_actions, teacher_performance, self.policy = \
            self.ensemble_learner.learn(cache=cache)
        if self.policy is None:
            return
        self.log["perf_teacher"] = teacher_performance 
        self.log["runtimes"]["other"] = time.time() - time_training
        if self.mr_config["run_baseline"]:
            self.original_policy = self.policy.copy()
            
        print("\n(ii) Evaluating original model performance...")
        eval_time = time.time()
        performance_original, _ = self.ensemble_learner.evaluate(0, unsafe_approximation)
        self.unsafe_approximation = unsafe_approximation
        self.log["perf_origin"] = performance_original
        self.log["runtimes"]["performance_origin"] += time.time() - eval_time
        
        self.log["orig_fidelity"] = self.check_fidelity()
        return
    
    def fuzz_and_fault_analysis(self):
        """
        Running fuzzing and fault analysis on the current iteration
        Returns:
            _type_: (Bool, Bool): 
              First represents whether the repair loop should be terminated.
              Second represents whether new faults were found.
        """
        print(f"\n\ta) Running fault detection with PlaJa: ")
        # detect faults
        faults_file, fa_logs = self.fault_analysis.find_faults(self.log["#its"])
        self.log["runtimes"]["fa"] += fa_logs["runtime"]
        if fa_logs["repair_status"]:
            self.next_policy = None
            self.log["repair_status"] = fa_logs["repair_status"]
            return True, False
            
        # parse detected faults
        parsing_logs = self.fault_analysis.parse_and_add_faults(self.policy, faults_file)
        self.log["runtimes"]["other"] += parsing_logs["runtime"]
        # keep track of tot. faults across iterations 
        if self.mr_config["repair_method"] == "penalty":
            # penalty trees repair doesn't stack faults across iterations in FaultAnalysis
            # --> we manually accumulate faults in self.all_faults for correct logging
            if parsing_logs["new_faults"]:
                self.all_faults["x"], self.all_faults["aa"], self.all_faults["y"], __ = \
                    attach_faults(
                        self.all_faults["x"].reset_index(drop=True), 
                        self.all_faults["aa"], 
                        self.all_faults["y"].reset_index(drop=True),
                        self.fault_analysis.faulty_states.reset_index(drop=True), 
                        self.fault_analysis.possible_actions, 
                        self.fault_analysis.faulty_actions.reset_index(drop=True),
                        at=None
                    )
            self.log["cumulated_faults"] = len(self.all_faults["x"])
        new_faults_found = parsing_logs["new_faults"]
        return False, new_faults_found
    
    def verify_and_fault_analysis(self):
        """ 
        Run verification followed by fault analysis if the policy was found to be unsafe
        """
        print(f"\nVerifying repaired policy...") 
        self.log["#ver_its"] += 1
        verification_log_file, ver_logs = self.verifier.verify_policy(self.log["#its"])
        self.log["runtimes"]["ver"] += ver_logs["runtime"]
        # update faults with the most recent faults
        self.verifier.set_faults(*self.fault_analysis.get_faults())
        parsed_logs = self.verifier.parse_verification_logs_and_add_faults(
            self.log["#its"], verification_log_file, ver_logs["status"]
        )
        self.fault_analysis.set_faults(*self.verifier.get_faults()) 
        self.log["runtimes"]["other"] += parsed_logs["runtime"]
        self.log["ver_status"] = parsed_logs["ver_status"]
        self.log["ver_faults"] = parsed_logs["ver_faults"]
        self.log["repair_status"] = parsed_logs["repair_status"] \
            if parsed_logs["repair_status"] else self.log["repair_status"]
        # if no new faults found by verifier, we stop
        to_break = not parsed_logs["ver_faults"] 
        if not to_break and self.mr_config["repair_method"] == "penalty":
            # we did find a new fault with verification
            # parse and add this new fault to all_faults for correct logging
            fx, aa, fy = parse_faults(verification_log_file, self.benchmark, self.prefix)
            self.all_faults["x"], self.all_faults["aa"], self.all_faults["y"], __ = \
                attach_faults(
                    self.all_faults["x"].reset_index(drop=True), 
                    self.all_faults["aa"], 
                    self.all_faults["y"].reset_index(drop=True),    
                    fx.reset_index(drop=True), aa, fy.reset_index(drop=True), at=None
                )
            self.log["cumulated_faults"] = len(self.all_faults["x"])
        return to_break
    
    def log_repair_status(self):
        if self.log["repair_status"] in ["infeasible", "timeout"]:
            # [S7] Scenario no. 7: optimization problem is infeasible/times out
            print(f"\n\n[{self.log['#its']}-RES] Repair ended with status: {self.log['repair_status']}.")

        elif self.log["repair_status"] == "oom":
            # [S8] Termination scenario no. 8: out of memory in PlaJA
            print(f"\n\n[{self.log['#its']}-RES] PlaJA ran out of memory.")

        elif self.log["repair_status"] == "BUG":
            # [SB] Bug scenario
            print(f"\n\n[{self.log['#its']}-RES] There was some BUG in the loop.")
        
        elif self.log["repair_status"] in ['safe', 'optimal', 'suboptimal'] and not self.fa_config['run_verification']:
            # [S9] Scenario no. 9: no verification, just return inner loop positive result
            print(f"\n\n[{self.log['#its']}-RES] Model seems safe (but without exact verification!)")

        elif self.log["repair_status"] == "maxit":
            # [S10] Scenario no. 10: model not safe even for inner loop, but we go on
            print(f"\n\n[{self.log['#its']}-RES] Model still unsafe - max inner iterations reached.")
            # also force verification status to MAXIT (if we are running verification)
            if self.fa_config['run_verification']:
                self.log["ver_status"] = "MAXIT"  
            
        elif self.log["repair_status"] == "unsafe":
            # AAAI26 rebuttal - added to avoid useless inner loop runs
            # print(f"\n\n[{log['#its']}-RES] Model still unsafe - loop terminated.")
            # break
            print(f"\n\n[{self.log['#its']}-RES] Model still unsafe.")
        
        else:
            raise ValueError(f"Unknown result status: {self.log['repair_status']}")
    
    def repair_policy(self):
        """
            Run the correct fault fixing algorithm
        """
        mr_time = time.time()
        # Set faults as all faults found till now
        self.fault_fixer.set_faults(*self.fault_analysis.get_faults()) 
        # Fix faults
        self.next_policy, repair_logs = self.fault_fixer.fix_faults(self.policy)
        # non-repair bookkeeping done inside the fixer (infeasibility detection,
        # feasibility-tree learning, post-solve diagnostics) is booked as "other"
        other = repair_logs.get("runtime_other", 0.0)
        self.log["runtimes"]["repair"] += (time.time() - mr_time) - other
        self.log["runtimes"]["other"] += other
        self.log["runtimes"]["fix"] += repair_logs["runtime"]
        self.log["retrained_safe"] = repair_logs["retrained_safe"] \
            if self.mr_config["repair_method"].startswith("retraining") else False
        self.log["repair_status"] = repair_logs["repair_status"]
        if "#trees" in repair_logs.keys():
            self.log["#trees"] += repair_logs["#trees"]
        return self.next_policy is not None
            
    def run(self, cache, unsafe_approximation):
        """Run the debugging loop; on walltime (SIGTERM) write a partial results line.
        NOTE: after SIGTERM there are only ~2 minutes before SIGKILL
        (timeout --kill-after), so the walltime path must not start new PlaJA runs."""
        try:
            self._run_loop(cache, unsafe_approximation)
        except WalltimeExceeded:
            self._write_abort_results("walltime")
            return
        except MemoryError:
            self._write_abort_results("oom-repair")
            return
        except GurobiError as e:
            if "memory" not in str(e).lower():
                raise
            self._write_abort_results("oom-repair")
            return
        try:
            self.final_evaluations_and_logging() # On loop termination, get final statistics
        except (WalltimeExceeded, MemoryError):
            # abort during the final evaluations: keep the loop's statuses,
            # write the line with whatever performance numbers we already have
            self._write_abort_results(None)

    def _write_provisional_results(self):
        """Write a PENDING results line with everything known before verification.
        It is replaced by the final line (see RepairLoopResults.write_results)."""
        import copy
        snapshot = copy.deepcopy(self.log)
        snapshot["runtimes"]["tot"] = time.time() - self.tot_time
        total_faults, __, __ = self.fault_analysis.get_faults()
        snapshot["#faults"] = snapshot["cumulated_faults"] \
            if self.mr_config["repair_method"].startswith("penalty") else len(total_faults)
        runtime = {k: int(v) for k, v in snapshot["runtimes"].items()}
        self.results_processor.write_results(snapshot, provisional=True, runtime=runtime,
                                             prefix=self.prefix, benchmark=self.benchmark)

    def _write_abort_results(self, status):
        """Write the results line after an abort (walltime SIGTERM or out-of-memory).
        status=None keeps the statuses the loop already established."""
        print(f"\n(!) Job aborted ({status or 'post-loop'}) -- writing partial results line.")
        if status is not None:
            self.log["repair_status"] = status
            if self.fa_config["run_verification"] and not self.log["ver_status"]:
                self.log["ver_status"] = "WALLTIME" if status == "walltime" else "N/A"
        self.log["runtimes"]["tot"] = time.time() - self.tot_time
        fa = getattr(self, "fault_analysis", None)
        if fa is not None:
            total_faults, __, __ = fa.get_faults()
            self.log["#faults"] = self.log["cumulated_faults"] \
                if self.mr_config["repair_method"].startswith("penalty") else len(total_faults)
        runtime = {k: int(v) for k, v in self.log["runtimes"].items()}
        self.results_processor.write_results(self.log, runtime=runtime,
                                             prefix=self.prefix, benchmark=self.benchmark)

    def _run_loop(self, cache, unsafe_approximation):
        self.initialize_policy(cache, unsafe_approximation)
        if self.policy is None: # We are running an invalid configuration
            print("We reach here")
            self.log["perf_origin"] = ("no data", "no data", "no data", "no data")
            runtime = {k: int(v) for k, v in self.log["runtimes"].items()}
            self.results_processor.write_results(self.log, runtime=runtime, prefix=self.prefix, benchmark=self.benchmark) # Write None to CSV file
            exit()

        presumed_safe = False         # if True, time to run verification/exit
        
        self.fault_analysis : FaultAnalysis = FaultAnalysis(self.mr_config, 
                                                            self.tree_config, 
                                                            self.fa_config, 
                                                            self.prefix, 
                                                            self.benchmark
                                            )
        
        self.fault_fixer : FaultFixer = \
            create_fault_fixer(
                self.mr_config["repair_method"], 
                training_data_states=self.training_states, 
                training_data_actions=self.training_actions, 
                mr_config=self.mr_config, 
                tree_config=self.tree_config, 
                prefix=self.prefix, 
                benchmark=self.benchmark
        )

        if self.fa_config["run_verification"]:
            self.verifier : Verifier = \
                create_verifier_factory(
                    self.fa_config["verifier"], 
                    mr_config=self.mr_config, 
                    tree_config=self.tree_config, 
                    fconfig=self.fa_config, 
                    prefix=self.prefix, 
                    benchmark=self.benchmark
                )

        while self.log["#its"] < self.mr_config["iterations"]:
            
            # [T4] termination scenario no. 4: max iterations reached
            if self.log["#its"] == self.mr_config['iterations']-1:
                print(f"\n\tMAX ITERATIONS reached: terminating...\n")
                self.log["repair_status"] = "maxit"
                # we also align verification status
                if self.fa_config['run_verification']:
                    self.log["ver_status"] = "MAXIT"
                break
            
            print(f"\n\n*** Starting loop iteration {self.log['#its']} ***")
            # (1) Load policy
            time_other = time.time()
            self.policy = self.ensemble_learner.load(self.log["#its"]) 
            self.log["runtimes"]["other"] += time.time() - time_other
            
            # (2) Run fault analysis if policy was not safe in previous iteration
            if not presumed_safe:
                to_break, new_faults_found = self.fuzz_and_fault_analysis()
                if to_break: break
                
                # (3) Repair model if:
                #   - new faults found through FA in this iteration
                #   - new fault found through verification in previous iteration
                #   - we are retraining and previous iteration did not train a safe model
                if (new_faults_found or self.log["ver_faults"] or 
                    (self.mr_config["repair_method"].startswith("retraining") and 
                     not self.log["retrained_safe"])
                    ):
                    repaired = self.repair_policy()
                    if not repaired:
                        # [T1] termination scenario no. 1: repair failed (OOM/timeout)
                        print(f"\t\t--> Repair failed. \n")
                        break
                else:
                    # [T3] termination scenario no. 3: model seems safe (i.e., no new faults found)
                    print(f"\t\t--> no new faults found. \n")
                    if not self.log["repair_status"]:  self.log["repair_status"] = "safe" 
                    presumed_safe = True
                    self.next_policy = self.policy.copy()
                        
                # reset flag for new faults found via verification
                self.log["ver_faults"] = False

            elif self.fa_config["run_verification"]:
                # (4) verify model if it was presumed safe in previous iteration
                presumed_safe = False
                # evaluate policy performance if it's the first verification attempt
                if self.log["#ver_its"] == 0: 
                    print(f"\n(iii) Evaluating model performance after FA done.")
                    perf_time = time.time()
                    perf_mid, uf_approx = self.ensemble_learner.evaluate(
                        self.log["#its"], self.unsafe_approximation
                    )
                    if uf_approx:
                        self.unsafe_approximation = uf_approx
                    self.log["runtimes"]["performance_fa_only"] += time.time() - perf_time
                    self.log["perf_fa_only"] = perf_mid
                    # snapshot the results line before verification: an uncatchable
                    # kill during verification then still leaves a (PENDING) record
                    self._write_provisional_results()
                
                # now we verify
                to_break = self.verify_and_fault_analysis()
                if to_break:
                    # verification did not find an unsafe path + fault
                    break
                # [T5] termination scenario no. 5: verification budget exhausted
                #      (faults found by the verifier are logged but NOT repaired)
                max_ver_its = self.fa_config.get("max_ver_its", 0)
                if max_ver_its and self.log["#ver_its"] >= max_ver_its:
                    print(f"\n\n[{self.log['#its']}-RES] Verification found new faults, but the "
                          f"verification budget ({max_ver_its}) is exhausted: terminating.")
                    break
            else:
                # either model is safe or we ran into memory/max iteration issues. Time to exit.
                self.log_repair_status() 
                break
            
            self.log["#its"] += 1
                
            # whenever we proceed to next iteration, we export current version of the model
            time_other = time.time()
            if self.debug_iteration < self.log["#its"]: # Presumes that we have already saved earlier models
                self.ensemble_learner.write_policy_to_file(self.next_policy, self.log["#its"])
            self.log["runtimes"]["other"] += time.time() - time_other
    
    def final_evaluations_and_logging(self):
        # further check for final model performance (if no plaja OOM/gurobi timeout)
        last_saved_it = max(0, self.log["#its"] - 1) 
        if not self.log["repair_status"] in ["oom", "infeasible", "timeout", "BUG", "no-nn-found"]:
            print("\n(iv) Evaluating final model performance...")
            perf_time = time.time()
            perf_final, __ = self.ensemble_learner.evaluate(last_saved_it, self.unsafe_approximation)
            self.log["perf_final"] = perf_final
            self.log["runtimes"]["performance_final"] += time.time() - perf_time
            self.log["final_fidelity"] = self.check_fidelity()

        self.log["runtimes"]["tot"] = time.time() - self.tot_time
        
        # Collect final set of faults
        total_faults, _, total_faulty_actions = self.fault_analysis.get_faults()
        if self.mr_config["repair_method"] == "penalty":
            # for penalty trees, we do not stack faults across iterations, 
            # so we use the manual cumulated_faults count instead
            self.log["#faults"] = self.log["cumulated_faults"]
        else:
            self.log["#faults"] = len(total_faults) 
        
        # Optionally, run penalty trees baseline with final set of faults from loop
        if self.mr_config['run_baseline'] and self.log["repair_status"] not in ["oom", "infeasible", "timeout", "BUG"]:
            print("\n[!] Running penalty trees baseline with final set of faults...")
            baseline = Baseline(self.mr_config, self.tree_config, self.fa_config, self.prefix, self.benchmark, last_saved_it)
            baseline_logs = baseline.run_baseline(self.original_policy, total_faults, total_faulty_actions)
            self.log["runtimes"]["penalty_trees"] += baseline_logs["runtime"]
            self.log["perf_penalty_trees"] = baseline_logs["perf_penalty_trees"]
            self.log["runtimes"]["performance_penalty_trees"] += baseline_logs["runtime_performance_penalty_trees"]
            
        # turn all runtimes into integers
        runtime = {k: int(v) for k, v in self.log["runtimes"].items()}
        # prettier summary print
        print("\n\n=================== FINAL SUMMARY ===================")
        print(f"Original performance:        {self.log['perf_origin']}")
        print(f"Performance after FA only:   {self.log['perf_fa_only']}")
        print(f"Final performance:           {self.log['perf_final']}")
        if self.mr_config['run_baseline']:
            print(f"Final performance (penalty trees baseline): {self.log['perf_penalty_trees']}")
        print(f"Total #faults found:         {self.log['#faults']}")
        print(f"Total #feasibility trees:    {self.log['#trees']}")
        print(f"Final repair status:         {self.log['repair_status']}")
        print(f"Final verification status:   {self.log['ver_status']}")
        print(f"Total runtime (s):           {runtime['tot']}")
        print(f"Total #iterations:           {self.log['#its']}")
        print(f"Total #verification it.:     {self.log['#ver_its']}")
        print(f"Fidelity: Original ({self.log['orig_fidelity']}) Final ({self.log['final_fidelity']})")
        print("=====================================================\n")
        
        self.results_processor.write_results(self.log, runtime=runtime, prefix=self.prefix, benchmark=self.benchmark) # Write results to CSV file
                

                  
                    
                
 
