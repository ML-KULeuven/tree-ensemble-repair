from fault_fixing_algorithms.fault_fixer import FaultFixer, safety_report, remove_conflicting_rows
from utils.utils import *
from fault_fixing_algorithms.helper_functions.retraining import *

class FALossFixer(FaultFixer):
    def __init__(self, training_data_states, training_data_actions, mr_config, tree_config, prefix, benchmark):
        # initialize parent class
        super().__init__(training_data_states, training_data_actions, mr_config, tree_config, prefix, benchmark)
        # create an empty dictionary for weight of the faults
        self.fault_weights = {}
        self.next_size_increase = 0
        
    def fix_faults(self, policy): # Returns fixed policy (at_optimized)
        """
        Method for fault fixing using FALoss
        """
        
        repair_log = {
            "#its": 1,                          # total number of iterations
            "retrained_safe": False,                        # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "runtime": 0.0,                     # repair runtime
        }
        
        # clean original training set from rows identified as faults
        cleaned_training_states, cleaned_training_actions = remove_conflicting_rows(
            self.training_states, self.training_actions, self.faulty_states, self.faulty_actions
        )
        
        # not sure what needs to be used, putting both here for safety
        try:
            tot_trees = self.tree_config["#trees"] # Specify number of trees
        except KeyError:
            tot_trees = self.tree_config["num_trees"]
        
        # if new faults found, add them with initial weight. Otherwise use existing weights
        for row in self.faulty_states.to_numpy():
            self.fault_weights.setdefault(tuple(row), self.mr_config["retraining_weight_boost"])
            
        ret_t0 = time.time()
        fixed_policy, __, still_unsafe_flag = retrain_ensemble(
            cleaned_training_states, cleaned_training_actions, 
            self.faulty_states, self.faulty_actions, self.possible_actions, 
            self.fault_weights,
            params=self.tree_config,
            # prev_bst=None,  # or pass previous booster if you want incremental growth
            num_boosting_rounds=tot_trees, #tree_config.get("num_trees", 20),
            retrain_method="retraining-loss",
            violator_boost=self.mr_config["retraining_weight_boost"],
            benchmark=self.benchmark,
        )
        ret_t1 = time.time()
        repair_log["runtime"] += ret_t1 - ret_t0 # hack: retraining time logged as gurobi time

        print("\t[DEBUG] Fault weights distribution:",
            dict(zip(*np.unique(list(self.fault_weights.values()), return_counts=True))))

        # check resulting model safety
        still_unsafe_report = safety_report(policy, fixed_policy, self.faulty_states, self.faulty_actions, self.possible_actions)

        # debug
        # if still_unsafe_flag != still_unsafe_report:
        #     import xgboost as xgb
        #     print(f"[DEBUG] Safety mismatch detected ({self.benchmark}, {self.prefix}): su={still_unsafe_flag}, still_unsafe={still_unsafe_report}")
            # Compare predictions for all faults
            # print("[DEBUG] Comparing Booster vs AddTree predictions for fault set...")
            # raw_booster = bst.predict(xgb.DMatrix(self.faulty_states.values), output_margin=True).reshape(-1, self.tree_config["num_class"])
            # raw_addtree = [fixed_policy.eval(fx)[0] for fx in self.faulty_states.values]
            # for i, (fx, fy, aa) in enumerate(zip(self.faulty_states.values, self.faulty_actions, self.possible_actions)):
            #     booster_pred = raw_booster[i]
            #     addtree_pred = raw_addtree[i]
            #     booster_argmax = aa[np.argmax(booster_pred[aa])]
            #     addtree_argmax = addtree_pred[aa].argmax()
            #     if booster_argmax != addtree_argmax:
            #         print(f"[DEBUG] Fault {i}: State={fx}")
            #         print(f"  Forbidden label: {fy}, Applicable actions: {aa}")
            #         print(f"  Booster logits: {booster_pred}, argmax(applicable)={booster_argmax}")
            #         print(f"  AddTree logits: {addtree_pred}, argmax(applicable)={addtree_argmax}")
        # assert still_unsafe_flag == still_unsafe_report, \
        #     "Mismatch in safety status after retraining."
        # we make this a soft assertion and just consider not safe
        if still_unsafe_flag != still_unsafe_report:
            print("[DEBUG] Safety mismatch detected ({}, {}): su={}, still_unsafe={}".format(
                self.benchmark, self.prefix, still_unsafe_flag, still_unsafe_report))
            print(f"\t--> considering retrained model as unsafe.")

        # ---------------------------------------------------------------------------
        # # Optionally grow ensemble size when retraining isn't working (specs in mr_config)
        # max_weight = max(self.fault_weights.values())

        # # Compute threshold: every N boosts for top weight (defined in config), increase size
        # boost_factor = self.mr_config["retraining_weight_boost"]
        # boost_steps = self.mr_config["retraining_#increase"]
        # trigger_threshold = boost_factor ** (self.next_size_increase * boost_steps)

        # if max_weight >= trigger_threshold:
        #     # Increase number of trees
        #     added_trees = self.mr_config["retraining_size_increase"]
        #     tot_trees += added_trees
        #     self.tree_config["num_trees"] = tot_trees  # keep metadata consistent

        #     print(
        #         f"\t(!) Added {added_trees} trees after {next_size_increase * boost_steps} boosts; "
        #         f"new size = {tot_trees} (threshold = {trigger_threshold:.1f})"
        #     )

        #     self.next_size_increase += 1

        #     # Optionally increase depth (up to max allowed)
        #     if self.tree_config["max_depth"] < mr_config["retraining_max_depth"]:
        #         self.tree_config["max_depth"] += 1
        #         print(f"\t    + increased max depth to {self.tree_config['max_depth']}.")
        # ---------------------------------------------------------------------------
        
        # neither of the two flags has to be true for safety
        if not still_unsafe_report and not still_unsafe_flag:
            print("\t--> All faults fixed!")
            repair_log["retrained_safe"] = True
            repair_log["repair_status"] = "optimal"
        else:
            print("\t(!) Retrained model still unsafe on collected faults.")
            repair_log["retrained_safe"] = False
            repair_log["repair_status"] = "unsafe"
        return fixed_policy, repair_log
    
    def fix_faults_hard(self, policy): # Returns fixed policy (at_optimized)
        """
        Identical to fix_faults, BUT it forces the method to try fixing all faults 
        by accumulating weights over multiple retraining iterations.
        """

        repair_log = {
            "#its": 0,                          # total number of iterations
            "retrained_safe": False,            # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "runtime": 0.0,                     # repair runtime
        }
        
        # clean original training set from rows identified as faults
        cleaned_training_states, cleaned_training_actions = remove_conflicting_rows(
            self.training_states, self.training_actions, self.faulty_states, self.faulty_actions
        )

        # not sure what needs to be used, putting both here for safety
        try:
            tot_trees = self.tree_config["#trees"] # Specify number of trees
        except KeyError:
            tot_trees = self.tree_config["num_trees"]

        # initialize weight for all faults
        for row in self.faulty_states.to_numpy():
            self.fault_weights.setdefault(tuple(row), self.mr_config["retraining_weight_boost"])
        
        still_unsafe_flag = True
        while still_unsafe_flag:
            repair_log["#its"] += 1
            print(f"\n\t[INFO] Hard retraining iteration {repair_log['#its']}...")

            ret_t0 = time.time()
            fixed_policy, __, still_unsafe_flag = retrain_ensemble(
                cleaned_training_states, cleaned_training_actions, 
                self.faulty_states, self.faulty_actions, self.possible_actions, 
                self.fault_weights,
                params=self.tree_config,
                # prev_bst=None,  # or pass previous booster if you want incremental growth
                num_boosting_rounds=tot_trees, #tree_config.get("num_trees", 20),
                retrain_method="retraining-loss",
                violator_boost=self.mr_config["retraining_weight_boost"],
                benchmark=self.benchmark,
            )
            ret_t1 = time.time()
            repair_log["runtime"] += ret_t1 - ret_t0 # hack: retraining time logged as gurobi time

            print("\t[DEBUG] Fault weights distribution:",
                dict(zip(*np.unique(list(self.fault_weights.values()), return_counts=True))))

            # check resulting model safety
            still_unsafe_report = safety_report(policy, fixed_policy, self.faulty_states, self.faulty_actions, self.possible_actions)

            # we make this a soft assertion and just consider not safe
            if still_unsafe_flag != still_unsafe_report:
                print("[DEBUG] Safety mismatch detected ({}, {}): su={}, still_unsafe={}".format(
                    self.benchmark, self.prefix, still_unsafe_flag, still_unsafe_report))
                print(f"\t--> considering retrained model as unsafe.")

            if not still_unsafe_report and not still_unsafe_flag:
                print("\t--> All faults fixed!")
                repair_log["retrained_safe"] = True
                repair_log["repair_status"] = "optimal"
            else:
                print("\t(!) Retrained model still unsafe on collected faults.")
                repair_log["retrained_safe"] = False
                repair_log["repair_status"] = "unsafe"

            # if the highest weight has reached the max, break to avoid infinite loop
            from fault_fixing_algorithms.helper_functions.retraining import MAX_FAULT_WEIGHT
            max_weight = max(self.fault_weights.values())
            if max_weight >= MAX_FAULT_WEIGHT:
                print(f"\t(!) Maximum fault weight {MAX_FAULT_WEIGHT} reached, stopping retraining.")
                break
        return fixed_policy, repair_log

