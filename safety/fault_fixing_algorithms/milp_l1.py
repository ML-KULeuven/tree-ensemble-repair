from fault_fixing_algorithms.fault_fixer import FaultFixer, check_safety
from utils.utils import *
from fault_fixing_algorithms.helper_functions.leaf_refinement import *
from fault_fixing_algorithms.helper_functions.enforce_feasibility import *
from policy.tree_ensemble_helpers import is_discrete_domain

class MILPL1Fixer(FaultFixer):
    def fix_faults(self, policy): # Returns fixed policy (at_optimized)
        """
        Repair with the MILP objective restricted to L1 leaf changes only.
        """
        repair_log = {
            "#its": 1,                          # total number of iterations
            "#trees": 0,                        # total number of added trees to make MILP feasible/retraining safe
            "repair_status": None,              # repair result status
            "runtime": 0.0,                     # repair runtime: MILP solve only
            # bookkeeping that is NOT repair work (summed into "runtime_other",
            # which the pipeline books under the "other" runtime):
            "runtime_infeasibility": 0.0,       # detecting infeasibility (IIS)
            "runtime_feasibility_trees": 0.0,   # decoupling OCs / learning fault-separator trees
            "runtime_diagnostics": 0.0,         # post-solve sanity checks / reporting
            "runtime_other": 0.0,
        }

        def _finalize(log):
            log["runtime_other"] = (log["runtime_infeasibility"]
                                    + log["runtime_feasibility_trees"]
                                    + log["runtime_diagnostics"])
            return log

        # i. add feasibility trees to avoid having clashing faults with same OC
        len_at_before = len(policy)
        t_tree = time.time()
        policy = decouple_ocs(policy, self.faulty_states, self.faulty_actions, self.possible_actions,
                              discrete=is_discrete_domain(self.benchmark))
        repair_log["runtime_feasibility_trees"] += time.time() - t_tree
        if len_at_before != len(policy):
            repair_log["#trees"] += len(policy) - len_at_before
            print(f"\t[DEBUG] Feasibility trees added: {len(policy) - len_at_before}, "
            f"This iteration = {repair_log['#trees']}")
            
        fault_counts = self.faulty_actions.value_counts().to_dict()
        print(f"\t\t- fault counts: {fault_counts}")
        # check how many leaves are visited by forbidden states
        faulty_leaves = count_faulty_leaves(policy, self.faulty_states)

        # ii. Create W/Phi matrices to reproduce ensemble output with matrix multiplicat.
        xtrain, _, _, _ = train_test_split(self.training_states, self.training_actions, 
                                           test_size=TEST_SIZE, random_state=SEED, 
                                           shuffle=True)
        print("\n\tc) Optimizing leaf values with MILP-L1 (Gurobi):")
        W, Phi = build_W_phi(xtrain, policy)
        # print("\n\t\t- weight matrix built")

        # iii. Frame optimization problem where we:
        #       - minimize the change in weights W (i.e., leaf values of the ensemble)
        #       - enforce new constraints with forbidden state-value pairs
        m, new_weights = get_linear_problem(policy, W, Phi, self.faulty_states, self.faulty_actions, 
                                            faulty_leaves, self.mr_config['fl_only'],
                                            self.possible_actions, self.mr_config['gur_tol'],
                                            verbose=False)
        # print("\t\t- linear problem created")

        # iv. Solve the optimization problem, extract optimized leaf values
        gur_time = time.time()
        gurobi_status, optimized_weights = solve_linear_problem(m, W, new_weights)
        repair_log["runtime"] += time.time() - gur_time

        if gurobi_status in ["infeasible", "timeout"]:
            
            # try solving infeasibility by adding new trees
            if gurobi_status == "infeasible":
                
                # TMP experiment: add splits only
                len_at_before = len(policy)
                policy, W, Phi, faulty_leaves, gurobi_status, optimized_weights, repair_log = \
                    enforce_feasibility(m, policy, xtrain, self.faulty_states, self.faulty_actions, 
                                        self.possible_actions, repair_log, self.mr_config['fl_only'], 
                                        self.mr_config['gur_tol'],
                                        discrete=is_discrete_domain(self.benchmark))
                repair_log["#trees"] += len(policy) - len_at_before
                
            # [T2] termination scenario no. 2: problem times out or still infeasible 
            if gurobi_status in ["timeout", "infeasible"]:
                repair_log["repair_status"] = gurobi_status
                fixed_policy = None
                return None, _finalize(repair_log)

        # v. Check how many weights have been changed (and how many are in faulty leaves)
        t_diag = time.time()
        check_modified_weights(W, optimized_weights, faulty_leaves, self.mr_config['fl_only'])
        repair_log["runtime_diagnostics"] += time.time() - t_diag

        # vi. Update leaf values of the ensemble  
        fixed_policy = policy.copy()
        base_score = [policy.get_base_score(l) for l in range(policy.num_leaf_values())]
        fixed_policy = set_new_leaf_vals_mc(fixed_policy, base_score, optimized_weights)
        repair_log["repair_status"] = gurobi_status
        
        # vii. Assert and guarantee repair works:
        t_diag = time.time()
        check_safety(policy, fixed_policy, self.faulty_states, self.faulty_actions, self.possible_actions)
        repair_log["runtime_diagnostics"] += time.time() - t_diag
        return fixed_policy, _finalize(repair_log)

