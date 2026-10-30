# Tree Ensemble Repair through Constrained Optimization

In this folder, we use our MILP-based method to repair tree ensembles for the property of **multistep safety**. Our tree ensembles are policies in planning domains.
The repair loop searches for faults (states where the policy picks an unsafe action) with PlaJA, repairs them using MILP, and repeats until no new faults are found. The policy is then evaluated and, optionally, verified.

Below are all repair methods we use, together with the names used in the paper (our method is `milp`).
| `-r` value | paper name | what it does |
|---|---|---|
| `milp` | MILP Repair | MILP over leaf values: fixes all faults, objective minimizes leaf value changes and training accuracy (weight `-ls`) |
| `milp-l1` | MILP-L1 | MILP that only minimises leaf changes |
| `retraining-augment` | MH | baseline that retrains on data augmented with the faults |
| `retraining-loss` | FALoss | baseline that retrains using a fault-aware loss |
| `penalty` | PT | baseline that adds penalty trees on top of the policy (shield) |

## Setup

1. **Python environment** (Python 3.12):

   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

   The MILP methods need a Gurobi licence (`GRB_LICENSE_FILE`).

2. **Benchmarks.** Download `benchmarks.zip` from this [TODO link], place it in this folder and unzip it:

   ```bash
   unzip benchmarks.zip      # creates safety/benchmarks/
   ```

   or point `PLAJA_BENCH_DIR` at an existing benchmark directory.

3. **PlaJA** is used for fault search, policy evaluation and verification. 
Build it so that the binary is at `third_party/plaja/build/PlaJA`, or set `PLAJA_BIN` to the binary.

## Running one experiment

```bash
python3 start_experiment.py -r milp -ls 0.001 -n 100000 -p det one_way_line_64_64
```

The last argument is the benchmark, `-p` its variant (prefix). Useful options:

| option | meaning |
|---|---|
| `-r` | repair method, see the table above |
| `-ls`, `-n` | MILP only: lambda and number of training samples in the objective |
| `-d`, `-t` | tree depth and number of trees (default 5 and 20) |
| `-rv True` | verify the policy after the loop |
| `-mv 1` | with `-rv True`: verify once, then stop, even if new faults are found |

The paper results use `-rv True -mv 1`: repair until fuzzing finds no new faults,
evaluate the policy, verify it once, and stop. `python3 start_experiment.py --help`
lists all options.

**Resource limits** are read from the environment, in MiB:
`JOB_MEM_MIB` caps the Python process, `FA_PLAJA_MEM_MIB` caps PlaJA during fault
search and verification, and `DEFAULT_PLAJA_MEM_MIB` caps PlaJA during evaluation.

## Running all benchmarks

`submit_script.py` runs every combination of benchmark, repair method and
hyperparameters, several at a time. Edit the lists at the top of the file
(`benchmarks`, `repair_methods`, `Lambdas`, ...) and the number of parallel jobs
`N_WORKERS`, then run:

```bash
python3 submit_script.py         # logs go to logs/<job name>.log
```

Each job calls `execute_ensemble.sh`, which calls `start_experiment.py`. That script
runs without verification (`-rv False`); change it to `-rv True -mv 1` for the
paper setting.

## Reproducing the paper's figures and tables

```bash
python3 make_neurips_results.py
```

This reads the results files and writes to `neurips_results/`:

| file | content |
|---|---|
| `rainclouds_eval.pdf`, `rainclouds_train.pdf` | %Safe and %Goal per method, on the evaluation and repair states |
| `performance_eval.tex`, `performance_train.tex` | %Goal and %Safe per benchmark |
| `loop_statistics.tex` | number of faults and loop iterations |
| `runtime.tex` | fault search, repair and total time |
| `loop_outcomes.tex` | whether each method's loop converged and was verified |
| `sources.json` | for every cell, where its value came from |

## Repository layout

| path | content |
|---|---|
| `start_experiment.py` | entry point for one experiment |
| `pipelines/` | the repair loop and two auxiliary experiments (`-exp count_faults`, `-exp percent_fixed`) |
| `fault_detection/` | fault search and verification with PlaJA |
| `fault_fixing_algorithms/` | the repair methods; the MILP is in `helper_functions/milp/leaf_refinement.py` |
| `policy/` | training and loading tree-ensemble policies |
| `results_processors/` | writing and parsing the results files |
| `utils/` | paths, PlaJA interface, shared helpers |
| `plotting/` | shared plot style and the raincloud figures |
| `make_neurips_results.py`, `neurips_results/` | paper figures and tables |
