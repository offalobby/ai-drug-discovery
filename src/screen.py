"""Rank an external, unlabeled molecule library with the trained HIV activity model.

Scores are virtual prioritization only: a high-ranked molecule is NOT an experimentally confirmed hit.

    python -m src.screen --library data/sample_library.csv
    python -m src.screen --library data/clintox.csv.gz --output outputs/screened_clintox.csv

Only load model artifacts you produced yourself: joblib files can execute arbitrary code.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import DataStructs

from . import chemistry, data

log = logging.getLogger("screen")


def prepare_library(df: pd.DataFrame, benchmark: set, smiles_column="smiles", name_column="name",
                    keep_overlap=False, stereo_insensitive=True):
    """Canonicalize, drop invalid and duplicate structures, and (by default) benchmark overlaps.

    With `stereo_insensitive`, a library molecule also counts as overlapping when it matches a
    benchmark structure after stereochemistry is removed (e.g. a fully specified drug whose
    benchmark record has no stereo labels).
    """
    if stereo_insensitive:
        benchmark = set(benchmark) | {chemistry.stereo_free_smiles(s) for s in benchmark}
    counts = {"input_rows": int(len(df))}
    lib = pd.DataFrame({"raw_smiles": df[smiles_column]})
    lib["name"] = (df[name_column].astype(str) if name_column in df.columns
                   else [f"LIB{i + 1:05d}" for i in range(len(df))])
    lib["smiles"] = [chemistry.canonical_smiles(s) for s in lib["raw_smiles"]]
    counts["invalid_removed"] = int(lib["smiles"].isna().sum())
    lib = lib.dropna(subset=["smiles"])
    # Dummy atoms (*) denote polymers/attachment points rather than discrete molecules.
    dummy = lib["smiles"].str.contains("*", regex=False)
    counts["dummy_atom_structures_removed"] = int(dummy.sum())
    lib = lib[~dummy]
    before = len(lib)
    lib = lib.drop_duplicates(subset="smiles", keep="first")
    counts["duplicates_removed"] = int(before - len(lib))
    lib["in_benchmark"] = lib["smiles"].isin(benchmark)
    if stereo_insensitive:
        lib["in_benchmark"] |= lib["smiles"].map(chemistry.stereo_free_smiles).isin(benchmark)
    counts["benchmark_overlap"] = int(lib["in_benchmark"].sum())
    if not keep_overlap:
        lib = lib[~lib["in_benchmark"]]
    counts["ranked"] = int(len(lib))
    return lib.reset_index(drop=True), counts


def diverse_shortlist(ranked: pd.DataFrame, fps, n: int, max_similarity: float) -> list:
    """Greedy pick down the ranking, skipping molecules too similar to anything already picked."""
    picked = []
    for i in range(len(ranked)):
        if len(picked) >= n:
            break
        if picked and max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in picked])) >= max_similarity:
            continue
        picked.append(i)
    return picked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--model", default="outputs/artifacts/model.joblib")
    parser.add_argument("--benchmark-smiles", default="outputs/artifacts/all_benchmark_smiles.txt")
    parser.add_argument("--split-assignments", default="outputs/split_assignments.csv",
                        help="Used to compute similarity to the model's training molecules")
    parser.add_argument("--keep-overlap", action="store_true", help="Keep molecules present in the benchmark")
    parser.add_argument("--exact-stereo-overlap", action="store_true",
                        help="Match benchmark overlap on isomeric SMILES only (default also ignores stereo)")
    parser.add_argument("--output", default="outputs/screened_library.csv")
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--shortlist-max-similarity", type=float, default=0.6)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    bundle = joblib.load(args.model)
    benchmark = set(Path(args.benchmark_smiles).read_text(encoding="utf-8").split())
    raw = pd.read_csv(args.library)
    lib, counts = prepare_library(raw, benchmark, args.smiles_column, args.name_column, args.keep_overlap,
                                  stereo_insensitive=not args.exact_stereo_overlap)
    log.info("Library filtering: %s", counts)
    if lib.empty:
        log.warning("No molecules left to rank (all invalid, duplicated, or in the benchmark).")
        return None

    X = chemistry.featurize(lib["smiles"], bundle["radius"], bundle["fp_size"])
    lib["score"] = bundle["model"].predict_proba(X)[:, 1]

    fps = chemistry.bit_vectors(lib["smiles"], bundle["radius"], bundle["fp_size"])
    split = pd.read_csv(args.split_assignments)
    train_smiles = split.loc[split["split"].isin(["train", "valid"]), "smiles"]
    lib["max_tanimoto_to_training"] = chemistry.max_tanimoto_to_reference(
        fps, chemistry.bit_vectors(train_smiles, bundle["radius"], bundle["fp_size"]))
    desc = pd.DataFrame([chemistry.descriptors(s) for s in lib["smiles"]])
    lib = pd.concat([lib, desc], axis=1)
    lib["scaffold"] = [chemistry.murcko_scaffold(s) for s in lib["smiles"]]

    order = np.argsort(-lib["score"].to_numpy(), kind="mergesort")
    lib = lib.iloc[order].reset_index(drop=True)
    fps = [fps[i] for i in order]
    lib.insert(0, "rank", np.arange(1, len(lib) + 1))
    picked = diverse_shortlist(lib, fps, args.shortlist_size, args.shortlist_max_similarity)
    lib["diverse_shortlist"] = False
    lib.loc[picked, "diverse_shortlist"] = True
    lib["novel_vs_benchmark"] = ~lib["in_benchmark"]
    lib = lib.drop(columns=["in_benchmark", "raw_smiles"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lib.to_csv(output, index=False)
    summary = {
        "library": args.library, "library_sha256": data.sha256(Path(args.library)),
        "model_config": bundle.get("selected_config"), "filtering": counts,
        "score_quantiles": {q: float(lib["score"].quantile(q)) for q in (0.5, 0.9, 0.99)},
        "top10": lib.head(10)[["rank", "name", "smiles", "score", "max_tanimoto_to_training", "MW", "logP",
                               "passes_rule_of_five"]].to_dict(orient="records"),
        "top10_unique_scaffolds": int(lib.head(10)["scaffold"].nunique()),
        "diverse_shortlist": lib.loc[picked, ["rank", "name", "score"]].to_dict(orient="records"),
        "top10_median_similarity_to_training": float(lib.head(10)["max_tanimoto_to_training"].median()),
        "library_median_similarity_to_training": float(lib["max_tanimoto_to_training"].median()),
    }
    data.write_json(summary, output.with_suffix(".summary.json"))
    log.info("Wrote %d ranked molecules to %s", len(lib), output)
    print(lib.head(10)[["rank", "name", "score", "MW", "logP", "max_tanimoto_to_training"]].to_string(index=False))
    return lib


if __name__ == "__main__":
    main()
