from fault_fixing_algorithms.fault_fixer import FaultFixer
from fault_fixing_algorithms.penalty_tree import *
from utils.utils import *

class Baseline:
    def __init__(self, mr_config, tree_config, fa_config, prefix, benchmark, last_saved_it):
        self.fixer = PenaltyTreeFixer({}, {}, mr_config, tree_config, prefix, benchmark)
        self.last_saved_it = last_saved_it
        self.fa_config = fa_config
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.prefix = prefix
        self.benchmark = benchmark

    
    def add_faults(self, Fx, Fy):
        self.fixer.set_faults(Fx, [], Fy)
        
    def fix_faults(self, policy):
        fixed_policy, logs = self.fixer.fix_faults(policy)
        return fixed_policy, logs
        
    def run_baseline(self, policy, Fx, Fy):
        self.add_faults(Fx, Fy)
        fixed_policy, logs = self.fix_faults(policy)
        penalty_model_path = get_ensemble_file(self.benchmark, self.prefix, self.mr_config["repair_method"],
                                               self.tree_config['model_type'], self.tree_config["max_depth"], 
                                               self.tree_config["num_trees"], self.tree_config["lr"], 
                                               # log["#its"], penalty=True)
                                                self.last_saved_it, penalty=True)

        with open(penalty_model_path, "w") as f:
            f.write(fixed_policy.to_json())
        perf_time = time.time()
        perf_final_baseline, __ = evaluate_ensemble_performance(self.prefix, self.benchmark, 
                                                                self.mr_config["repair_method"],
                                                                self.tree_config, self.last_saved_it, # log["#its"], 
                                                                self.fa_config, penalty=True)
        logs["runtime_performance_penalty_trees"] = time.time() - perf_time
        logs["perf_penalty_trees"] = perf_final_baseline
        return logs
    