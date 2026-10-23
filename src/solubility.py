"""Experiment B: measured aqueous solubility (ESOL / Delaney) regression.

Protocol (predeclared):
  1. Clean: canonical SMILES, drop invalid; average consistent duplicates, drop inconsistent ones.
  2. DeepChem ScaffoldSplitter 80/10/10.
  3. Candidates: mean-only baseline, ridge regression on Morgan fingerprints, Random Forest on
     Morgan fingerprints, plus Random Forest on RDKit descriptors and on fingerprints+descriptors.
  4. Select by validation MAE; refit on train+valid; one held-out test evaluation per candidate,
     with the validation-selected model reported as the primary result.

Targets stay in measured log10(mol/L) units: no target normalization is applied.
Run from the repository root:  python -m src.solubility
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import chemistry, data, metrics, plots
from .train import fit_deepchem

log = logging.getLogger("solubility")


def candidate_models(seed: int, n_jobs: int) -> dict:
    """name -> (feature set, estimator factory). Predeclared; do not edit after viewing test results."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    def rf(**kw):
        return lambda: RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=n_jobs, **kw)

    return {
        "mean_baseline": ("fingerprint", lambda: DummyRegressor(strategy="mean")),
        "ridge_fp_alpha1": ("fingerprint", lambda: Ridge(alpha=1.0)),
        "ridge_fp_alpha10": ("fingerprint", lambda: Ridge(alpha=10.0)),
        "ridge_fp_alpha100": ("fingerprint", lambda: Ridge(alpha=100.0)),
        "rf_fp_leaf1": ("fingerprint", rf(min_samples_leaf=1, max_features="sqrt")),
        "rf_fp_leaf1_allfeatures": ("fingerprint", rf(min_samples_leaf=1, max_features=1.0)),
        "rf_descriptors": ("descriptors", rf(min_samples_leaf=1, max_features=1.0)),
        "rf_fp_plus_descriptors": ("fingerprint+descriptors", rf(min_samples_leaf=1, max_features=0.3)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/delaney-processed.csv")
    parser.add_argument("--url", default=data.ESOL_URL)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default=data.ESOL_TARGET)
    parser.add_argument("--radius", type=int, default=chemistry.DEFAULT_RADIUS)
    parser.add_argument("--fp-size", type=int, default=chemistry.DEFAULT_FP_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    source = data.ensure_download(Path(args.csv), args.url)
    raw = pd.read_csv(args.csv)
    clean, audit = data.clean_regression_dataset(raw, args.smiles_column, args.target_column)
    log.info("Audit: %s", audit)
    smiles = clean["smiles"].tolist()
    y = clean["target"].to_numpy()

    train_idx, valid_idx, test_idx = data.deepchem_scaffold_split(smiles, y, seed=args.seed)
    split = np.empty(len(clean), dtype=object)
    split[train_idx], split[valid_idx], split[test_idx] = "train", "valid", "test"
    clean.assign(scaffold=[chemistry.murcko_scaffold(s) for s in smiles], split=split).to_csv(
        out / "solubility_split_assignments.csv", index=False)
    split_summary = [{"split": n, "n": int(len(i)), "target_mean": float(y[i].mean()),
                      "target_std": float(y[i].std())}
                     for n, i in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]]

    fp = chemistry.featurize(smiles, args.radius, args.fp_size).astype(float)
    desc = chemistry.descriptor_matrix(smiles)
    features = {"fingerprint": fp, "descriptors": desc, "fingerprint+descriptors": np.hstack([fp, desc])}
    candidates = candidate_models(args.seed, args.n_jobs)

    rows = []
    for name, (feature_set, factory) in candidates.items():
        X = features[feature_set]
        model = fit_deepchem(factory(), X[train_idx], y[train_idx])
        pred = predict(model, X[valid_idx])
        s = metrics.regression_summary(y[valid_idx], pred)
        rows.append({"model": name, "features": feature_set, "valid_mae": s["mae"],
                     "valid_rmse": s["rmse"], "valid_r2": s["r2"]})
        log.info("Validation %-26s MAE=%.3f RMSE=%.3f R2=%.3f", name, s["mae"], s["rmse"], s["r2"])
    selected = min(rows, key=lambda r: r["valid_mae"])["model"]
    log.info("Selected by validation MAE: %s", selected)

    # Locked test evaluation: each candidate refit once on train+valid
    trval_idx = np.concatenate([train_idx, valid_idx])
    test_preds = {}
    for row in rows:
        feature_set, factory = candidates[row["model"]]
        X = features[feature_set]
        model = fit_deepchem(factory(), X[trval_idx], y[trval_idx])
        pred = predict(model, X[test_idx])
        test_preds[row["model"]] = pred
        s = metrics.regression_summary(y[test_idx], pred)
        row.update(test_mae=s["mae"], test_rmse=s["rmse"], test_r2=s["r2"],
                   selected=row["model"] == selected)
        if row["model"] == selected:
            final_model = model
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "solubility_validation_trials.csv", index=False)

    y_test = y[test_idx]
    selected_pred = test_preds[selected]
    test_summary = metrics.regression_summary(y_test, selected_pred)
    test_summary["bootstrap_95ci"] = metrics.bootstrap_regression(y_test, selected_pred, seed=args.seed)
    baseline_summary = metrics.regression_summary(y_test, test_preds["mean_baseline"])
    abs_err_sel = np.abs(y_test - selected_pred)
    abs_err_base = np.abs(y_test - test_preds["mean_baseline"])
    rng = np.random.default_rng(args.seed)
    diffs = [np.mean(abs_err_base[i] - abs_err_sel[i])
             for i in (rng.integers(0, len(y_test), len(y_test)) for _ in range(2000))]
    test_summary["mae_improvement_vs_mean_baseline"] = {
        "point": float(baseline_summary["mae"] - test_summary["mae"]),
        "bootstrap_95ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }
    log.info("TEST %s: MAE=%.3f RMSE=%.3f R2=%.3f", selected, test_summary["mae"], test_summary["rmse"],
             test_summary["r2"])

    test_smiles = [smiles[i] for i in test_idx]
    pd.DataFrame({"smiles": test_smiles, "measured_logS": y_test, "predicted_logS": selected_pred,
                  "abs_error": abs_err_sel, "mean_baseline_pred": test_preds["mean_baseline"],
                  "MW": desc[test_idx, 0], "logP": desc[test_idx, 1]}).sort_values(
        "abs_error", ascending=False).to_csv(out / "solubility_test_predictions.csv", index=False)

    joblib.dump({"model": final_model.model, "selected": selected, "features": candidates[selected][0],
                 "radius": args.radius, "fp_size": args.fp_size,
                 "descriptor_names": chemistry.DESCRIPTOR_NAMES, "trained_on": "train+valid"},
                out / "artifacts" / "solubility_model.joblib", compress=3)
    plots.regression_figure(y_test, selected_pred, test_summary, out / "plots" / "solubility_scatter.png",
                            title=f"ESOL held-out test: {selected}")
    plots.regression_comparison_figure(rows, out / "plots" / "solubility_models.png")

    data.write_json({
        "experiment": "ESOL measured aqueous solubility regression",
        "target_units": "log10(mol/L), measured; no target normalization",
        "protocol": {"split": "DeepChem ScaffoldSplitter 0.8/0.1/0.1", "selection_metric": "validation MAE",
                     "featurization": {"radius": args.radius, "fp_size": args.fp_size,
                                       "descriptors": chemistry.DESCRIPTOR_NAMES},
                     "seed": args.seed, "final_fit": "train+valid"},
        "environment": data.environment_info(),
        "data_source": source,
        "data_audit": audit,
        "splits": split_summary,
        "trials": rows,
        "selected_model": selected,
        "test": test_summary,
        "test_mean_baseline": baseline_summary,
        "runtime_seconds": round(time.time() - t0, 1),
    }, out / "solubility_metrics.json")
    log.info("Done in %.1fs", time.time() - t0)


def predict(model, X) -> np.ndarray:
    import deepchem as dc
    return np.asarray(model.predict(dc.data.NumpyDataset(X=X))).reshape(len(X))


if __name__ == "__main__":
    main()
