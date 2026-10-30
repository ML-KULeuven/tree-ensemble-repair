import click
import os
from pipelines.repair_loop import RepairLoop, WalltimeExceeded
from pipelines.count_faults import AnalyzeLoop
from utils.utils import (
    get_num_outputs, get_fixed_property_file, count_start_states, get_game,
    FA_TOTAL_TIME, FA_TIME_PER_QUERY, VERIFICATION_TIME, RESULT_DIR
)    

@click.command()
@click.argument('benchmark', default='one_way_line_64_64')
@click.option('--prefix', '-p', default="")
@click.option('--model_type', '-m', default='xgb', type=click.Choice(['xgb', 'rf']))
@click.option('--max_depth', '-d', default=5, type=int)
@click.option('--num_trees', '-t', default=20, type=int)
@click.option('--lr', '-l', default=0.4, type=float)
@click.option('--iterations', '-i', default=1000, type=int)
@click.option('--repair_method', '-r', default="milp-l1",
              type=click.Choice(['milp-l1', 'penalty', 'retraining-augment', 'retraining-loss', "milp"]),
              help="milp: generic MILP with fidelity; milp-l1: L1 leaf changes only.")
@click.option('--lambda_slack', '-ls', default=1, type=float) # Used only in milp: weight of L1 leaf changes relative to fidelity loss.
@click.option('--num_training_samples', '-n', default=1000, type=int) # Used only in milp to decide how many training samples to use
@click.option('--only_goal_reaching_data','-og', is_flag=True, default=False, help="Preprocess the training data to only use goal reaching traces? If set to false, then use data we had already cached (which has all traces).")
@click.option('--penalty_trees', '-pt', default=False, type=bool) # learn penalty trees on the set of faults identified during repair as a baseline for comparison
@click.option('--fuzzing', '-fu', default="random", 
              type=click.Choice(['random', 'probabilistic', 'greedy'])) # fuzzing type
@click.option('--run_verification', '-rv', default=False, type=bool) # exactly verify final policy
@click.option('--verifier', '-ver', default="plaja", type=click.Choice(['plaja', 'ic3']))
@click.option('--max_verification_its', '-mv', default=0, type=int,
              help="Stop after this many verification runs even if they find new faults (0 = unlimited). "
                   "E.g. -rv True -mv 1: fuzz/repair until no faults, evaluate, verify once, stop.")
@click.option('--debug', is_flag=True, default=False, help="Enable debug mode")
@click.option('--debug_iteration', '-di', default=0, type=int, help="Iteration to run in debug mode (default: 0)")
@click.option('--experiment_mode', '-exp', default="repair", 
              type=click.Choice(['repair', 'count_faults', 'percent_fixed'])) # which experiment to run
def main(benchmark, prefix, model_type, max_depth, num_trees, lr, iterations, repair_method, lambda_slack, num_training_samples, only_goal_reaching_data,
            penalty_trees, fuzzing, run_verification, verifier, max_verification_its, debug, debug_iteration, experiment_mode):
    
    tree_config = {
        "model_type": model_type,
        "max_depth": max_depth,
        "num_trees": num_trees,
        "lr": lr if model_type == "xgb" else None,
        "ablation": False,
        "num_class": get_num_outputs(benchmark, prefix),
        "only_goal_data": only_goal_reaching_data
    }

    # assert repair_method not in ["penalty", "shielding"], \
    #     "Use milp-l1 + penalty trees baseline for repair via shielding."
    if repair_method == "milp":
        repair_method = f"milp_{lambda_slack}lambda_{num_training_samples}samples"
    
    only_goal_suffix = "_only-goal-data" if only_goal_reaching_data else "_all-data"
    repair_method += only_goal_suffix
    mr_config = {   
        "fl_only": True,                    # only optimize faulty leaves
        "gur_tol": 1e-2,                    # gurobi tolerance --> default 1e-2 
        "iterations": iterations,
        "safe_its": 1,                      # stop as soon as FA does not identify new faults
        "repair_method": repair_method,
        "run_baseline": penalty_trees,
        # below: retraining-specific params (TBD)
        "retraining_weight_boost": 2.0,        # per-iteration boost for violators in retraining
        # "retraining_size_increase": 10,      # per-iteration increase in ensemble size (#trees)
        # "retraining_#increase": 5,           # how many boosts to top weight before increasing size
        #                                      # --> 2^5, 2^10, 2^15, ...
        # "retraining_max_size": 50,           # maximum #trees when retraining
        # "retraining_max_depth": 8            # maximum depth when retraining
    }

    num_starts = 20000                      # we always use 20k start states (10k train, 10k eval)
    fa_config = {
        "num_starts": num_starts,                       # we always use 20k start states 
        "fa_total_time": FA_TOTAL_TIME,                 # how much time to spend in PlaJA in total
        "fa_time_per_query": FA_TIME_PER_QUERY,         # how much time per query in fault analysis
        "run_verification": run_verification,
        "verification_total_time": VERIFICATION_TIME,   # how much time to spend in verification in total
        "verifier": verifier,  
        "max_ver_its": max_verification_its,            # 0 = keep verifying/repairing until proved safe
        "enumerable": False,
        "fuzzing": fuzzing, 
        # "evaluation_method": eval_method,
        "extensive_fuzzing": False, 
    }

    # For blocksworld and beluga, the start states are enumerable so set repair states == evaluation states
    if get_game(benchmark) == "blocksworld" or get_game(benchmark) == "beluga":
        fa_config['enumerable'] = True
        uf_approx = True # For efficiency, we know that blocksworld calculating exact %Unsafe is computationally infeasible so directly skip to approximation
    else:
        uf_approx = False
    
    # main experiment (debugging loop)
    if experiment_mode == "repair":
        repair_loop = RepairLoop(
            prefix, benchmark, mr_config, tree_config, fa_config, debug, debug_iteration
        )
        repair_loop.run(cache = True, unsafe_approximation=uf_approx)
    # count faults ablation: 
    #   how many new faults are introduced through repair?
    elif experiment_mode == "count_faults":
        analyze_loop = AnalyzeLoop(prefix, benchmark, mr_config, tree_config)
        analyze_loop.run()
    # %Fixed metric: is repair fixing unseen faults?
    #  - additional safety metric to %Unsafe
    #  - computed on final debugging loop policy
    elif experiment_mode == "percent_fixed":
        from pipelines.percent_fixed import PercentFixedFaults
        percent_fixed_loop = PercentFixedFaults(
            prefix, benchmark, mr_config, tree_config, fa_config
        )
        percent_fixed_loop.run(run_for_no_ver=False)
   

def _sigterm_to_exception(signum, frame):
    # the job's walltime sends SIGTERM;
    # convert it into an exception so the repair loop can write a results line
    # before the follow-up SIGKILL
    raise WalltimeExceeded("SIGTERM received (job walltime)")


def _apply_job_memory_limit():
    #  Memory allocations fail as catchable MemoryError/GurobiError instead of the
    # cgroup OOM-killer's uncatchable SIGKILL
    mem_mib = os.environ.get("JOB_MEM_MIB")
    if mem_mib:
        import resource
        limit = int(mem_mib) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(f"[init] RLIMIT_AS set to {int(mem_mib)} MiB (JOB_MEM_MIB)")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _sigterm_to_exception)
    _apply_job_memory_limit()
    main()
  