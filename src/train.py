"""Experiment A: HIV assay-activity classification and retrospective virtual-screening enrichment.

Protocol (predeclared):
  1. Clean: canonical SMILES, drop invalid, dedupe, exclude conflicting labels.
  2. DeepChem ScaffoldSplitter 80/10/10.
  3. Morgan radius 2, 2048 bits, chirality.
  4. Fit three Random Forest configurations on train; select by validation average precision.
  5. Refit the winner on train+valid; evaluate exactly once on the held-out test split.

Run from the repository root:  python -m src.train
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

log = logging.getLogger("train")

# Predeclared candidate configurations. Do not edit after looking at test results.
# "primary" is the registered protocol; "starter_package" reproduces the three candidates from the
# original starter code and is reported as a disclosed secondary replication.
RF_CONFIG_SETS = {
    "primary": {
        "rf_200trees_fulldepth_leaf1": dict(n_estimators=200, max_depth=None, min_samples_leaf=1),
        "rf_500trees_fulldepth_leaf3": dict(n_estimators=500, max_depth=None, min_samples_leaf=3),
        "rf_500trees_depth30_leaf1": dict(n_estimators=500, max_depth=30, min_samples_leaf=1),
    },
    "starter_package": {
        "rf_150trees_depth18_leaf1": dict(n_estimators=150, max_depth=18, min_samples_leaf=1),
        "rf_200trees_depth24_leaf2": dict(n_estimators=200, max_depth=24, min_samples_leaf=2),
        "rf_250trees_fulldepth_leaf3": dict(n_estimators=250, max_depth=None, min_samples_leaf=3),
    },
}
RF_CONFIGS = {name: cfg for configs in RF_CONFIG_SETS.values() for name, cfg in configs.items()}
RF_COMMON = dict(class_weight="balanced_subsample", max_features="sqrt")


def make_rf(name: str, seed: int, n_jobs: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**RF_CONFIGS[name], **RF_COMMON, random_state=seed, n_jobs=n_jobs)


def make_logistic(seed: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000,
                              random_state=seed)


def fit_deepchem(estimator, X, y):
    """Wrap a scikit-learn estimator as a DeepChem SklearnModel and fit on a NumpyDataset."""
    import deepchem as dc
    model = dc.models.SklearnModel(estimator)
    model.fit(dc.data.NumpyDataset(X=X, y=y))
    return model


def positive_scores(model, X) -> np.ndarray:
    """Ranking score for the active class (predict_proba column 1; not a calibrated probability)."""
    import deepchem as dc
    pred = np.asarray(model.predict(dc.data.NumpyDataset(X=X)))
    pred = pred.reshape(len(X), -1)
    return pred[:, -1] if pred.shape[1] > 1 else pred[:, 0]


def summarize_split(name, idx, labels):
    y = labels[idx]
    return {"split": name, "n": int(len(idx)), "actives": int(y.sum()), "prevalence": float(y.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/HIV.csv")
    parser.add_argument("--url", default=data.HIV_URL, help="Download source if --csv is missing")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="HIV_active")
    parser.add_argument("--radius", type=int, default=chemistry.DEFAULT_RADIUS)
    parser.add_argument("--fp-size", type=int, default=chemistry.DEFAULT_FP_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--config-set", choices=sorted(RF_CONFIG_SETS), default="primary")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    configs = RF_CONFIG_SETS[args.config_set]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Data and audit
    source = data.ensure_download(Path(args.csv), args.url)
    raw = pd.read_csv(args.csv)
    clean, audit = data.clean_binary_dataset(raw, args.smiles_column, args.label_column)
    if "activity" in raw.columns:
        audit["source_activity_class_counts"] = raw["activity"].value_counts().to_dict()
    log.info("Audit: %s", {k: v for k, v in audit.items() if k != "invalid_smiles_examples"})
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy()

    # 2. Scaffold split
    train_idx, valid_idx, test_idx = data.deepchem_scaffold_split(smiles, labels, seed=args.seed)
    split = np.empty(len(clean), dtype=object)
    split[train_idx], split[valid_idx], split[test_idx] = "train", "valid", "test"
    scaffolds = [chemistry.murcko_scaffold(s) for s in smiles]
    assignments = clean.assign(scaffold=scaffolds, split=split)
    assignments.to_csv(out / "split_assignments.csv", index=False)
    split_summary = [summarize_split(n, i, labels) for n, i in
                     [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]]
    for name, idx in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        entry = next(s for s in split_summary if s["split"] == name)
        entry["unique_scaffolds"] = int(assignments.loc[idx, "scaffold"].nunique())
    shared = (set(assignments.loc[train_idx, "scaffold"]) & set(assignments.loc[test_idx, "scaffold"])) - {""}
    audit["unique_scaffolds_total"] = int(assignments["scaffold"].nunique())
    audit["acyclic_molecules_empty_scaffold"] = int((assignments["scaffold"] == "").sum())
    audit["nonempty_scaffolds_shared_train_test"] = len(shared)
    log.info("Splits: %s", split_summary)

    # 3. Features
    X = chemistry.featurize(smiles, args.radius, args.fp_size)
    X_train, y_train = X[train_idx], labels[train_idx]
    X_valid, y_valid = X[valid_idx], labels[valid_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    # 4. Validation-based selection among predeclared RF configurations
    trials = []
    for name in configs:
        start = time.time()
        model = fit_deepchem(make_rf(name, args.seed, args.n_jobs), X_train, y_train)
        s = positive_scores(model, X_valid)
        summary = metrics.classification_summary(y_valid, s)
        trials.append({
            "model": name, **configs[name], **RF_COMMON,
            "valid_average_precision": summary["average_precision"],
            "valid_roc_auc": summary["roc_auc"],
            "valid_ef@1%": summary["enrichment@1%"]["enrichment_factor"],
            "valid_ef@5%": summary["enrichment@5%"]["enrichment_factor"],
            "valid_ef@10%": summary["enrichment@10%"]["enrichment_factor"],
            "fit_seconds": round(time.time() - start, 1),
            "candidate_for_selection": True,
        })
        log.info("Validation %s: AP=%.4f EF@5%%=%.2f", name, summary["average_precision"],
                 summary["enrichment@5%"]["enrichment_factor"])

    # Comparators reported alongside (not eligible for selection under the registered protocol)
    logreg = fit_deepchem(make_logistic(args.seed), X_train, y_train)
    lr_valid = metrics.classification_summary(y_valid, positive_scores(logreg, X_valid))
    mw_valid_scores = chemistry.descriptor_matrix([smiles[i] for i in valid_idx])[:, 0]
    mw_valid = metrics.classification_summary(y_valid, mw_valid_scores)
    for name, summary in [("logistic_regression_C0.1_balanced", lr_valid),
                          ("heuristic_molecular_weight_desc", mw_valid)]:
        trials.append({"model": name, "valid_average_precision": summary["average_precision"],
                       "valid_roc_auc": summary["roc_auc"],
                       "valid_ef@1%": summary["enrichment@1%"]["enrichment_factor"],
                       "valid_ef@5%": summary["enrichment@5%"]["enrichment_factor"],
                       "valid_ef@10%": summary["enrichment@10%"]["enrichment_factor"],
                       "candidate_for_selection": False})
    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(out / "validation_trials.csv", index=False)
    candidates = trials_df[trials_df["candidate_for_selection"]]
    selected = candidates.sort_values("valid_average_precision", ascending=False).iloc[0]["model"]
    log.info("Selected by validation AP: %s", selected)

    # 5. Refit on train+valid and evaluate once on the locked test split
    trval_idx = np.concatenate([train_idx, valid_idx])
    final = fit_deepchem(make_rf(selected, args.seed, args.n_jobs), X[trval_idx], labels[trval_idx])
    test_scores = positive_scores(final, X_test)
    test_summary = metrics.classification_summary(y_test, test_scores)
    test_summary["bootstrap"] = {
        f"ef@{int(f * 100)}%": metrics.bootstrap_ef(y_test, test_scores, f, args.n_boot, args.seed)
        for f in (0.01, 0.05, 0.10)
    }
    log.info("TEST %s: AP=%.4f ROC-AUC=%.4f EF@5%%=%.2f", selected, test_summary["average_precision"],
             test_summary["roc_auc"], test_summary["enrichment@5%"]["enrichment_factor"])

    # Comparators on test (refit on train+valid as well)
    logreg_final = fit_deepchem(make_logistic(args.seed), X[trval_idx], labels[trval_idx])
    lr_test_scores = positive_scores(logreg_final, X_test)
    test_desc = chemistry.descriptor_matrix([smiles[i] for i in test_idx])
    comparators = {
        "logistic_regression_C0.1_balanced": metrics.classification_summary(y_test, lr_test_scores),
        "heuristic_molecular_weight_desc": metrics.classification_summary(y_test, test_desc[:, 0]),
    }

    # Error analysis: similarity of each test molecule to the final training pool
    ref_fps = chemistry.bit_vectors([smiles[i] for i in trval_idx], args.radius, args.fp_size)
    test_smiles = [smiles[i] for i in test_idx]
    test_sim = chemistry.max_tanimoto_to_reference(
        chemistry.bit_vectors(test_smiles, args.radius, args.fp_size), ref_fps)
    ranked = pd.DataFrame({
        "smiles": test_smiles, "scaffold": [scaffolds[i] for i in test_idx], "label": y_test,
        "score": test_scores, "logreg_score": lr_test_scores, "MW": test_desc[:, 0],
        "logP": test_desc[:, 1], "max_tanimoto_to_train": test_sim,
    }).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(out / "test_ranked.csv", index=False)
    error_analysis = analyze_errors(ranked, out)

    joblib.dump({"model": final.model, "radius": args.radius, "fp_size": args.fp_size,
                 "selected_config": selected, "trained_on": "train+valid"},
                out / "artifacts" / "model.joblib", compress=3)
    (out / "artifacts" / "all_benchmark_smiles.txt").write_text("\n".join(smiles) + "\n", encoding="utf-8")

    plots.classification_figure(y_test, test_scores, test_summary, out / "plots" / "test_curves.png",
                                comparators={"Logistic regression": lr_test_scores,
                                             "MW heuristic": test_desc[:, 0]})
    plots.validation_figure(trials_df, out / "plots" / "validation_trials.png")
    plots.similarity_figure(error_analysis["by_similarity"], out / "plots" / "test_by_similarity.png")

    results = {
        "experiment": "HIV assay activity classification (MoleculeNet HIV)",
        "protocol": {
            "featurization": {"type": "Morgan", "radius": args.radius, "fp_size": args.fp_size,
                              "include_chirality": True},
            "split": "DeepChem ScaffoldSplitter 0.8/0.1/0.1 (deterministic, largest scaffold sets to train)",
            "selection_metric": "validation average precision",
            "config_set": args.config_set,
            "candidates": {k: {**v, **RF_COMMON} for k, v in configs.items()},
            "seed": args.seed, "final_fit": "train+valid", "test_evaluations": 1,
        },
        "environment": data.environment_info(),
        "data_source": source,
        "data_audit": audit,
        "splits": split_summary,
        "validation_trials": trials_df.to_dict(orient="records"),
        "selected_model": selected,
        "test": test_summary,
        "test_comparators": comparators,
        "error_analysis": error_analysis,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    data.write_json(results, out / "metrics.json")
    log.info("Done in %.1fs; results in %s", time.time() - t0, out)
    return results


SIMILARITY_BINS = [0.0, 0.3, 0.4, 0.5, 0.7, 1.0001]


def analyze_errors(ranked: pd.DataFrame, out: Path) -> dict:
    """Performance by nearest-neighbour similarity, plus the most confident errors."""
    ranked = ranked.copy()
    ranked["similarity_bin"] = pd.cut(ranked["max_tanimoto_to_train"], SIMILARITY_BINS, right=False)
    by_bin = []
    for interval, group in ranked.groupby("similarity_bin", observed=True):
        row = {"bin": f"[{interval.left:.1f}, {min(interval.right, 1.0):.1f})", "n": int(len(group)),
               "actives": int(group["label"].sum())}
        if 0 < group["label"].sum() < len(group):
            s = metrics.classification_summary(group["label"], group["score"], fractions=(0.05,))
            row.update(roc_auc=s["roc_auc"], average_precision=s["average_precision"],
                       prevalence=s["prevalence"], ef_at_5pct=s["enrichment@5%"]["enrichment_factor"])
        by_bin.append(row)

    k = metrics.budget_size(len(ranked), 0.05)
    top = ranked.head(k)
    false_pos = ranked[ranked["label"] == 0].head(25)
    false_neg = ranked[ranked["label"] == 1].sort_values("score").head(25)
    false_pos.to_csv(out / "test_top_false_positives.csv", index=False)
    false_neg.to_csv(out / "test_lowest_scored_actives.csv", index=False)
    top_hits = top[top["label"] == 1]
    return {
        "by_similarity": by_bin,
        "median_similarity_test_to_train": float(ranked["max_tanimoto_to_train"].median()),
        "top5pct_unique_scaffolds": int(top["scaffold"].nunique()),
        "top5pct_hits_unique_scaffolds": int(top_hits["scaffold"].nunique()),
        "top5pct_median_similarity": float(top["max_tanimoto_to_train"].median()),
        "top5pct_hits_median_similarity": float(top_hits["max_tanimoto_to_train"].median()) if len(top_hits) else None,
        "spearman_score_vs_similarity": float(ranked["score"].corr(ranked["max_tanimoto_to_train"], method="spearman")),
        "actives_scored_zero": int(((ranked["label"] == 1) & (ranked["score"] == 0)).sum()),
        "molecules_scored_zero": int((ranked["score"] == 0).sum()),
    }


if __name__ == "__main__":
    main()
