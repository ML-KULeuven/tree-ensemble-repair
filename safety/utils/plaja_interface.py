import os
import subprocess
from utils.utils import *
import signal

# Helper to run a subprocess with a POSIX virtual-memory limit (RLIMIT_AS).
# --> constrains how much address space a child process (and its descendants) may allocate. 
def limit_memory_preexec(limit_mb):
    """Return a function suitable for subprocess preexec_fn that sets RLIMIT_AS.

    limit_mb: int or None. If None, no limits are set.
    """
    def _inner():
        if limit_mb is None:
            return
        import resource     # will fail on non-POSIX systems (who cares)
        limit_bytes = int(limit_mb) * 1024 * 1024

        # never exceed the inherited hard limit: if the parent python process runs
        # under its own RLIMIT_AS (JOB_MEM_MIB), asking for more would make
        # setrlimit fail and kill the child at spawn
        __, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit_bytes = min(limit_bytes, hard)

        # set both soft and hard limits
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return _inner

# PlaJA memory limits (MiB), overridable via environment. Keep these BELOW the
# scheduler's memory allowance (condor request_memory): the rlimit makes PlaJA
# fail with a catchable exit code, the cgroup OOM-killer is an uncatchable SIGKILL.
DEFAULT_PLAJA_MEM_MIB = int(os.environ.get("DEFAULT_PLAJA_MEM_MIB", 12 * 1024))  # 12 GB
FA_PLAJA_MEM_MIB = int(os.environ.get("FA_PLAJA_MEM_MIB", 96 * 1024))  # larger for fault collection

ORACLE_TYPE = "PI3" # "Tarjan"  
NUM_STARTS = 10000 # property files contain 20k start states but we TMP use a subset


def performance_cache_signature(command, input_paths):
    """Match completed statistics to their command and unchanged inputs."""
    return {"command": command, "inputs": {
        path: [os.stat(path).st_size, os.stat(path).st_mtime_ns]
        for path in input_paths}}


########################################### POLICY PERFORMANCE METRIC COMPUTATION #########################################################
def simulate_ensemble_for_policy_performance(prefix, benchmark, repair_method, tconfig, 
                                        iteration, fconfig, seed=SEED, evaluation_set=False, 
                                        train_test_split=0.5, penalty=False, approx=False):
    """
    Run a tree ensemble policy on num_starts randomly sampled start states from the start condition.
    """
    # HACK: A generic method of setting train_test_split and evaluation_set 
    # based on enumerability of state space
    train_test_split = 1.0 if fconfig["enumerable"] else 0.5
    evaluation_set = False if fconfig["enumerable"] else evaluation_set

    # HACK for blocksworld (otherwise not detected as enumerable for some reason)
    if get_game(benchmark) == "blocksworld":
        num_starts = 4139
    else:
        num_starts = NUM_STARTS

    model_file = get_model_file(prefix, benchmark)
    # LTS benchmarks have deterministic transitions under the fixed policy.
    # Repeating the same trace does not add information.
    with open(model_file) as model_stream:
        paths_per_start = 1 if json.load(model_stream).get("type") == "lts" else 100
    property_file = get_property_file(prefix, benchmark, fconfig["enumerable"], fconfig['num_starts'])
    interface_file = get_interface_file(prefix, benchmark)
    if tconfig['ablation']:
        ensemble_file = get_ablation_ensemble_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                      tconfig['max_depth'], tconfig['num_trees'], 
                                      tconfig['lr'], iteration, penalty) 
                                      # HACK: iteration here should be called with faultFrac
    else:
        ensemble_file = get_ensemble_file(benchmark, prefix, repair_method, tconfig['model_type'], 
                                        tconfig['max_depth'], tconfig['num_trees'], 
                                        tconfig['lr'], iteration, penalty)
    # print(f"\t- Ensemble file: {ensemble_file}")
    if not os.path.exists(ensemble_file):
        raise FileNotFoundError(f"Ensemble file {ensemble_file} does not exist.")
    
    stats_files = []
    # NOTE we now run for policy performance with both evaluation methods
    # --> "envelope": exhaustive %unsafe
    # --> "some_paths": approximate %unsafe 
    t0 = time.time()
    oor = False
    eval_methods = ["envelope", "some_paths"] if not approx else ["some_paths"]
    for eval_method in eval_methods:
        log_file_path = perf_log_file(benchmark, prefix, repair_method, tconfig['model_type'], tconfig['max_depth'],
                                    tconfig['num_trees'], tconfig['lr'], iteration, evaluation_set, tconfig['ablation'], eval_method)
        stats_file_path = perf_stat_file(benchmark, prefix, repair_method, tconfig['model_type'], tconfig['max_depth'],
                                        tconfig['num_trees'], tconfig['lr'], iteration, evaluation_set, tconfig['ablation'], eval_method)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        sample = "inc"
        use_evaluation = "--evaluation-set" if evaluation_set else ""
        # generic_command = f"{PLAJA_DIR} --model-file {model_file} --seed {seed} --additional-properties {property_file} --prop 1 --engine FAULT_ANALYSIS --max-time {fconfig['fa_total_time']} --ensemble-interface {interface_file} --stats-file {stats_file_path} --save-intermediate-stats --ensemble {ensemble_file} --initial-state-enum {sample} --oracle-use finding --oracle-mode infinite --infinite-oracle {ORACLE_TYPE} --applicability-filtering 1 --fault-experiment-mode ignore --search-dir backward --seconds-per-query {fconfig['fa_time_per_query']} --num-starts {fconfig['num_starts']} --distance-func avoid --rand-prob 1 --max-steps-per-start 1010 --retraining-train-test-split {train_test_split} {use_evaluation}"
        generic_command = f"{PLAJA_DIR} --model-file {model_file} --seed {seed} --additional-properties {property_file} --prop 1 --engine FAULT_ANALYSIS --max-time {fconfig['fa_total_time']} --ensemble-interface {interface_file} --stats-file {stats_file_path} --save-intermediate-stats --ensemble {ensemble_file} --initial-state-enum {sample} --oracle-use finding --oracle-mode infinite --infinite-oracle {ORACLE_TYPE} --applicability-filtering 1 --fault-experiment-mode ignore --search-dir backward --seconds-per-query {fconfig['fa_time_per_query']} --num-starts {num_starts} --distance-func avoid --rand-prob 1 --max-steps-per-start 1010 --retraining-train-test-split {train_test_split} {use_evaluation}"
        match eval_method:
            case "envelope":
                command = generic_command + f" --evaluation-method policy_envelope >> {log_file_path}"
            case "all_paths":
                command = generic_command + f" --evaluation-method all_policy_paths >> {log_file_path}"
            case "some_paths":
                command = generic_command + f" --evaluation-method sample_policy_paths --num-paths-per-start {paths_per_start} >> {log_file_path}"
        completion_path = stats_file_path + ".complete.json"
        signature = performance_cache_signature(command, [model_file, property_file,
                                                           interface_file, ensemble_file])
        if fconfig.get("reuse_performance") and os.path.exists(completion_path):
            try:
                with open(completion_path) as cache_stream:
                    receipt = json.load(cache_stream)
                stat = os.stat(stats_file_path)
                if (receipt["signature"] == signature and
                        receipt["stats"] == [stat.st_size, stat.st_mtime_ns]):
                    stats_files.append(stats_file_path)
                    print(f"\t- Reusing completed {eval_method} statistics: {stats_file_path}")
                    continue
            except (OSError, ValueError, KeyError):
                pass
        for path in [log_file_path, stats_file_path, completion_path]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        t = time.time()
        oor = False
        try:     
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                            # headroom over PlaJA's --max-time (see oracle.py)
                            timeout=fconfig.get('fa_total_time') + 600)
            stats_files.append(stats_file_path)
            stat = os.stat(stats_file_path)
            with open(completion_path, "w") as cache_stream:
                json.dump({"signature": signature,
                           "stats": [stat.st_size, stat.st_mtime_ns]}, cache_stream)
            print(f"\t- Policy performance computation with evaluation method '{eval_method}' completed in {time.time()-t:.3f} s.")
        except subprocess.TimeoutExpired as e:
            print(f"\t- Policy performance computation with evaluation method '{eval_method}' timed out after {time.time()-t:.3f} s.")
            oor = "timeout"
        except subprocess.CalledProcessError as e:
            if e.returncode == 134:
                print(f"\t- Policy performance computation with evaluation method '{eval_method}' timed out after {time.time()-t:.3f} s.")
                oor = "timeout"
            elif e.returncode == 137:
                print(f"\t- Policy performance computation with evaluation method '{eval_method}' OOM after {time.time()-t:.3f} s.")
                oor = "oom"
            elif e.returncode == 135 or e.returncode == 1:
                print(f"\t- Policy performance computation with evaluation method '{eval_method}' failed with signal {e.returncode} "\
                    f" after {time.time()-t:.3f} s.")
                oor = True
            elif e.returncode == 22:
                print(f"\t- Policy performance computation with evaluation method '{eval_method}' OOM after {time.time()-t:.3f} s ",
                      f"(due to imposed {DEFAULT_PLAJA_MEM_MIB / 1024} GB memory limit).")
                oor = "oom"
            else:
                print("Command failed!")
                print("Exit code:", e.returncode)
                print("Output:", e.stdout)
                print("Error output:", e.stderr)
    # print(f"Command executed successfully: {command}")
    print(f"\t- Policy performance computation completed in {time.time()-t0:.3f} s.")
    # return log_file_path, stats_file_path, oor
    return stats_files, oor
