# Tree Ensemble Repair through Constrained Optimization

This work proposes a **MILP-based repair** method for tree ensembles (XGBoost / Random Forest).

Given a trained ensemble, our MILP repair loop:
- finds property violations as concrete counterexamples through verification;
- modifies the ensemble's leaf values through a MILP so that the repaired model satisfies the property on all given counterexamples (while not degrading predictive performance);
- iterates the two earlier steps until no more counterexamples are found by the verifier. 

This codebase applies it to tree ensembles that must adhere to two distinct properties, fairness and safety. Each folder has its own README with setup and run instructions:

| Folder | Property enforced | README |
|---|---|---|
| [`fairness/`](fairness/) | **Individual fairness** — the prediction must not change when the protected attribute is flipped | [fairness/README.md](fairness/README.md) |
| [`safety/`](safety/) | **Multistep safety** — a tree ensemble policy must not reach an unsafe state | [safety/README.md](safety/README.md) |


## License

This code is released under the Apache License 2.0 (see [LICENSE](LICENSE)).

## Citation

Paper: [TODO: link to the paper]

If you use this code, please cite:

```bibtex
@inproceedings{cascioli2026treerepair,
  title     = {Tree-Ensemble Repair Through Constrained Optimization},
  author    = {Cascioli, Lorenzo and Jain, Chaahat and Steinmetz, Marcel and Hoffmann, J{\"o}rg and Davis, Jesse},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
