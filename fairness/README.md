# Tree Ensemble Repair through Constrained Optimization

In this folder, we use our MILP-based method to repair tree ensembles for the property of **individual fairness**, intended as the absence of causal discrimination (as in Calzavara et al., 2023): two inputs that differ only in a sensitive attribute must receive the same prediction. Fairness is verified exhaustively with [veritas](https://github.com/laudv/veritas).

For each dataset and fold, the pipeline:

1. trains an XGBoost or Random Forest model;
2. compresses it with **LOP**, so that Veritas can provably verify it;
3. runs our **iterative MILP repair loop**: (a) find all counterexamples through verification (i.e., all pairs of inputs that only differ in the sensitive attribute and receive different labels), (b) optimize the leaf values so that every pair gets the same label while preserving balanced accuracy on the training set, and (c) repeat (a)-(b) until the model is provably fair;
4. compares against two fairness-aware training methods: **FATT** (fairness-aware decision trees, Ranzato et al., CIKM 2021) and **BuDRO** (individually fair gradient boosting, Vargo et al., ICLR 2021).

Every dataset is split into 5 folds (60% train / 20% validation / 20% test). 

---

## Installation

Tested with Python 3.11 on Linux.

**1. Python packages.** In a fresh virtual environment:

```bash
pip install -r requirements.txt
```

This includes [veritas](https://github.com/laudv/veritas) (tree ensemble verifier) and [prada](https://github.com/laudv/prada) (quick access to tabular datasets). Note that `gurobipy` needs a [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (free for academic use).

**2. tree_compress (LOP).** Clone it next to this repository:

```bash
git clone https://github.com/TimoM99/tree_compress.git
cd tree_compress && git checkout 08da17e && cd ..
pip install -e ./tree_compress
```

**3. Baselines.** Clone them next to this repository:

```bash
# FATT
git clone https://github.com/fatt21/fatt.git
cd fatt && make bin/silva bin/meta-silvae && cd ..

# BuDRO
git clone https://github.com/UMich-FATML/BuDRO.git
```

**4. Paths.** Set the paths in [`paths.py`](paths.py):

| Variable | Meaning |
|---|---|
| `PRADA_DATA_DIR` | cache to download and store datasets (see [Data](#data)) |
| `CACHE_DIR` | cache for storing trained models and result files |
| `FATT_DIR` | FATT clone, with the built `bin/` binaries (default: `../fatt`) |
| `BUDRO_DIR` | BuDRO clone (default: `../BuDRO`) |
| `PARALLEL_BIN` | [GNU parallel](https://www.gnu.org/software/parallel/), used by the `run*.sh` scripts (falls back to `parallel` on your `PATH`) |

## Data

All datasets are downloaded through [prada](https://github.com/laudv/prada) and cached into `PRADA_DATA_DIR` (set in `paths.py`). 

The sensitive attribute of each dataset is already defined in `SENSITIVE_COLS` in [`utils.py`](utils.py).

## Running the experiments

Each step of the pipeline is one `main.py` experiment (`-e`); `run.sh` runs it on all datasets × 5 folds in parallel. Pass `--cache` to store models and results into `CACHE_DIR`.

```bash
# 1. Train original models and compress them with LOP
./run.sh -e orig --cache [-m xgb|rf]
./run.sh -e lop  --cache [-m xgb|rf]

# 2. Run MILP repair 
./run.sh -e milp --cache [-m xgb|rf]

# 3. Run baselines
./run.sh -e fatt --cache 
./run.sh -e budro --cache 
```

To run a single fold using XGB:

```bash
python main.py CompasTwoYears --fold 0 -e orig --cache   # train
python main.py CompasTwoYears --fold 0 -e lop  --cache   # compress
python main.py CompasTwoYears --fold 0 -e milp --cache   # repair
```

## Ablation and generalization studies

```bash
./run_ablation.sh -e 1 --cache     # sensitivity to the regularization weight lambda
./run_ablation.sh -e 2 --cache     # composite vs accuracy-only vs L1-only MILP objective
./run.sh -e milp-b   --cache       # alternative MILP enforcing accuracy through hard constraints
./run_generalization.sh            # check repair generalization to unseen counterexamples

```

All of them reuse the models of step 1.

## Collecting results

```bash
python collect_results.py -m xgb              # main tables and figures
python collect_ablation_results.py
python collect_budget_results.py              # hard budget vs basic MILP
python collect_generalization_results.py
python collect_runtime_analysis.py
```

Tables are printed to the terminal (and as LaTeX); figures are written to `figures/`.
