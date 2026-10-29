# Tree Ensemble Repair through Constrained Optimization

 This work proposes a **MILP-based repair** method for tree ensembles (XGBoost / Random Forest).

We apply it to two settings, one per folder. Each folder has its own README with setup and
run instructions:

| Folder | Property enforced | README |
|---|---|---|
| [`fairness/`](fairness/) | **Individual fairness** — the prediction must not change when the protected attribute is flipped | [fairness/README.md](fairness/README.md) |
| [`safety/`](safety/) | **Multistep safety** — a tree ensemble policy must not reach an unsafe state | [safety/README.md](safety/README.md) |

Both settings share the same core idea: verify the model, collect the inputs that violate
the property, and solve a MILP over the leaf values that repairs violations while
preserving predictive performance.

