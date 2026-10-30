#! /usr/bin/env python3

import os
import os.path
import glob
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Number of jobs to run at once. Defaults to the CPU count, override with the
# N_WORKERS environment variable (e.g. `N_WORKERS=8 python3 submit_script.py`).
N_WORKERS = 16


benchmarks = [
    ("det", "two_way_line_40_25_16_16"),
    ("det", "two_way_line_30_20_64_64"),
    ("det", "two_way_line_20_10_crazy_long"),
    ("non_det_with_park", "two_way_line_30_20_64_64"),
    ("non_det_with_park", "two_way_line_40_25_64_64"),
    ("non_det_with_park", "two_way_line_20_10_256_256"),
    ("non_det_no_park", "two_way_line_30_20_16_16"),
    ("non_det_with_park", "two_way_line_17_10_128_128"),
    ("det", "two_way_line_17_10_16_16"),
    ("non_det_no_park", "two_way_line_16_16"),
    ("non_det_no_park", "two_way_line_40_25_16_16"),
    ("non_det_with_park", "two_way_line_128_128"),
    ("det", "one_way_line_40_25_16_16"),
    ("non_det_no_park", "two_way_line_20_10_16_16"),
    ("non_det_no_park", "two_way_line_17_10_16_16"),
    ("det", "two_way_line_64_64"),
    ("non_det_no_park", "one_way_line_40_25_crazy_long"),
    ("non_det_no_park", "one_way_line_30_20_crazy_long"),
    ("swap_unsafe", "beluga_3_2_256_256"),
    ("non_det_no_park", "one_way_line_crazy_long"),
    ("swap_unsafe", "beluga_5_2_256_256"),
    ("", "bouncing_ball_0.05_0.05_32_32"),
    ("", "bouncing_ball_0.01_0.01_32_32"),
    ("", "bouncing_ball_32_32"),
    ("non_det_with_park", "one_way_line_40_25_64_64"),
    ("det", "one_way_line_64_64"),
    ("non_det_no_park", "one_way_line_20_10_128_128"),
    ("non_det_with_park", "one_way_line_20_10_32_32"),
    ("det", "one_way_line_20_10_crazy_long"),
    ("swap_unsafe", "beluga_4_2_128_128"),
    ("CI", "blocksworld_8_7_kinda_long"),
    ("", "stopping_car_32_32"),
    ("non_det_no_park", "one_way_line_17_10_crazy_long"),
    ("swap_unsafe", "beluga_6_2_256_256"),
    ("CA", "blocksworld_8_7_kinda_long"),
    ("non_det_with_park", "one_way_line_32_32"),
    ("det", "one_way_line_17_10_crazy_long"),
    ("", "inverted_pendulum_32_32"),
    ("non_det_with_park", "one_way_line_30_20_64_64"),
    ("det", "one_way_line_30_20_64_64"),
    ("", "linetrack_64_64"),
    ("non_det_with_park", "one_way_line_17_10_128_128"),
]

exp_dir = os.path.dirname(os.path.realpath(__file__))
exe_path = f"{exp_dir}/execute_ensemble.sh"
log_dir = os.path.join(exp_dir, "logs")

time_limit = 60*60*48 # 48 hours
MEMORY = 64


def get_data(prefix, benchmark, depth, num_trees, repair_method, data, lda, n_sample):
    """
    Run a teacher policy on num_starts randomly sampled start states from the start condition. Save the <state, action> decisions made by the nn policy.
    """
    if prefix == "":
        command = f"{exe_path} {benchmark} {depth} {num_trees} {repair_method} {data} {lda} {n_sample}"
    command = f"{exe_path} {prefix} {benchmark} {depth} {num_trees} {repair_method} {data} {lda} {n_sample}"
    return command

# Hyperparameters
# repair_methods = ["penalty", "milp-l1", "milp", "retraining-loss", "retraining-augment"]
repair_methods = ["penalty", "milp-l1", "milp", "retraining-loss", "retraining-augment"]
depths = [5]
trees = [20]
Lambdas = [0.1, 0.5, 1, 2, 10, 100]
Num_Samples = [100_000]
training_data = ["all-data"]


def run_job(name, command):
    """Execute a single job command, writing its output to logs/<name>.log."""
    log_path = os.path.join(log_dir, f"{name}.log")
    print(f"[START] {name} -> {log_path}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            command, shell=True, cwd=exp_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
    status = "DONE" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"[{status}] {name}")
    return name, result.returncode


def main():
    # Create output directory
    jobs = [] # This holds the commands to be executed. Use whatever system you want to execute them (e.g., slurm, local, etc.)
    job_names = []
    for prefix, benchmark in benchmarks:
        for depth in depths:
            for tree in trees:
                for repair_method in repair_methods:
                    for data in training_data:
                        lambdas, num_samples = [], []
                        if repair_method != "milp":
                            lambdas = [1]
                            num_samples = [100_000]
                        else:
                            lambdas = Lambdas
                            num_samples = Num_Samples
                        for lda in lambdas:
                            for n_sample in num_samples:
                                perf_command = get_data(prefix, benchmark, depth, tree, repair_method, data, lda, n_sample)
                                jobs.append(perf_command)
                                job_names.append(f"{prefix}_{benchmark}_{depth}_{tree}_{repair_method}_{data}_{lda}_{n_sample}")

    # Run the jobs in parallel, at most N_WORKERS at a time.
    os.makedirs(log_dir, exist_ok=True)
    print(f"Running {len(jobs)} jobs with {N_WORKERS} workers, logs in {log_dir}")
    failures = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(run_job, name, command): name
            for name, command in zip(job_names, jobs)
        }
        for future in as_completed(futures):
            name, returncode = future.result()
            if returncode != 0:
                failures.append(name)

    print(f"\nFinished {len(jobs)} jobs, {len(failures)} failed.")
    for name in failures:
        print(f"  FAILED: {name}")


if __name__ == "__main__":
    main()