"""Post-hoc sensitivity analysis: how much does HIV test enrichment depend on the partition?

Run after `python -m src.train`. Uses the RF configuration already selected on validation data
(read from outputs/metrics.json) with no re-tuning, and re-evaluates it on:
  * randomized scaffold splits (scaffold sets assigned in random order), several seeds
  * plain random splits (no scaffold grouping), several seeds, to show leakage optimism

This is a sensitivity analysis, not a replacement for the single locked test result.
    python -m src.robustness
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import chemistry, data, metrics, plots
from .train import fit_deepchem, make_rf, positive_scores

log = logging.getLogger("robustness")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = Path(args.output_dir)

    locked = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    config = locked["selected_model"]
    fp = locked["protocol"]["featurization"]
    assignments = pd.read_csv(out / "split_assignments.csv", keep_default_na=False)
    smiles = assignments["smiles"].tolist()
    labels = assignments["label"].to_numpy()
    X = chemistry.featurize(smiles, fp["radius"], fp["fp_size"])

    rows = [{"split_scheme": "DeepChem scaffold\n(locked result)", "seed": None,
             **_row(locked["test"])}]
    for seed in args.seeds:
        tr, va, te = data.randomized_scaffold_split(smiles, 0.8, 0.1, seed=seed)
        rows.append({"split_scheme": "Randomized\nscaffold split", "seed": seed,
                     **_evaluate(X, labels, np.concatenate([tr, va]), te, config, args)})
        log.info("randomized scaffold seed=%d: %s", seed, rows[-1])
    for seed in args.seeds:
        perm = np.random.default_rng(seed).permutation(len(smiles))
        cut = int(0.9 * len(smiles))
        rows.append({"split_scheme": "Random split\n(no scaffold grouping)", "seed": seed,
                     **_evaluate(X, labels, perm[:cut], perm[cut:], config, args)})
        log.info("random seed=%d: %s", seed, rows[-1])

    df = pd.DataFrame(rows)
    df.assign(split_scheme=df["split_scheme"].str.replace("\n", " ")).to_csv(out / "robustness.csv", index=False)
    summary = (df.assign(split_scheme=df["split_scheme"].str.replace("\n", " "))
                 .groupby("split_scheme", sort=False)[["roc_auc", "average_precision", "ef@1%", "ef@5%", "ef@10%"]]
                 .agg(["mean", "std", "min", "max"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    data.write_json({"selected_config": config, "seeds": args.seeds,
                     "summary": summary.reset_index().to_dict(orient="records")}, out / "robustness.json")
    plots.robustness_figure(df, out / "plots" / "robustness_ef5.png")
    print(summary.round(3).to_string())


def _evaluate(X, labels, train_idx, test_idx, config, args):
    model = fit_deepchem(make_rf(config, args.model_seed, args.n_jobs), X[train_idx], labels[train_idx])
    return _row(metrics.classification_summary(labels[test_idx], positive_scores(model, X[test_idx])))


def _row(summary):
    return {"n_test": summary["n"], "test_actives": summary["n_actives"], "roc_auc": summary["roc_auc"],
            "average_precision": summary["average_precision"],
            "ef@1%": summary["enrichment@1%"]["enrichment_factor"],
            "ef@5%": summary["enrichment@5%"]["enrichment_factor"],
            "ef@10%": summary["enrichment@10%"]["enrichment_factor"]}


if __name__ == "__main__":
    main()
