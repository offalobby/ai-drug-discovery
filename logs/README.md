# Run logs

Raw stdout/stderr from the runs that produced `outputs/`, kept as provenance.

| File | Command |
|---|---|
| `logs_train.txt` | `python -m src.train` (first run; produced the locked test result) |
| `logs_train_primary_rerun.txt` | `python -m src.train` rerun with the same seed — reproduced every metric exactly |
| `logs_train_starter.txt` | `python -m src.train --config-set starter_package` (secondary replication) |
| `logs_solubility.txt` | `python -m src.solubility` |
| `logs_robustness.txt` | `python -m src.robustness` (split-sensitivity analysis) |

The DeepChem warnings about PyTorch/TensorFlow and about training weights are expected: only the
splitters and the scikit-learn wrapper are used, and `RandomForestClassifier` ignores the per-sample
weights DeepChem would pass.
