"""Post-hoc error analysis of the locked HIV test predictions (no model refitting).

Reads outputs/test_ranked.csv and the raw HIV.csv to ask:
  * Are recovered hits mostly "confirmed active" (CA) or "moderately active" (CM) in the source?
  * Which scaffold families dominate the false positives?
  * How do computed properties of the top-ranked set differ from the rest of the pool?

    python -m src.analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import chemistry, data, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/HIV.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)

    ranked = pd.read_csv(out / "test_ranked.csv", keep_default_na=False)
    raw = pd.read_csv(args.csv)
    raw["smiles_canonical"] = [chemistry.canonical_smiles(s) for s in raw["smiles"]]
    activity = raw.dropna(subset=["smiles_canonical"]).set_index("smiles_canonical")["activity"]
    ranked["source_activity"] = ranked["smiles"].map(activity)

    k = metrics.budget_size(len(ranked), args.fraction)
    top = ranked.head(k)
    rest = ranked.iloc[k:]
    pct = int(round(args.fraction * 100))

    def class_counts(df):
        return {c: int(n) for c, n in df["source_activity"].value_counts().items()}

    false_pos = top[top["label"] == 0]
    fp_scaffolds = (false_pos.assign(scaffold=false_pos["scaffold"].replace("", "(acyclic)"))
                    .groupby("scaffold").size().sort_values(ascending=False))
    fp_ring_systems = false_pos["smiles"].map(_ring_signature).value_counts().head(10)

    props = ["MW", "logP"]
    result = {
        "budget_fraction": args.fraction,
        "n_selected": k,
        "test_activity_classes": class_counts(ranked),
        f"top{pct}pct_activity_classes": class_counts(top),
        "recall_by_class": {
            c: float((top["source_activity"] == c).sum() / max((ranked["source_activity"] == c).sum(), 1))
            for c in ("CA", "CM")
        },
        f"top{pct}pct_false_positives": int(len(false_pos)),
        "false_positive_top_scaffolds": fp_scaffolds.head(10).to_dict(),
        "false_positive_common_ring_signatures": fp_ring_systems.to_dict(),
        "property_medians": {
            f"top{pct}pct": top[props].median().round(2).to_dict(),
            "rest_of_test": rest[props].median().round(2).to_dict(),
            f"top{pct}pct_hits": top.loc[top["label"] == 1, props].median().round(2).to_dict(),
            f"top{pct}pct_false_positives": false_pos[props].median().round(2).to_dict(),
        },
        "similarity_medians": {
            f"top{pct}pct_hits": float(top.loc[top["label"] == 1, "max_tanimoto_to_train"].median()),
            f"top{pct}pct_false_positives": float(false_pos["max_tanimoto_to_train"].median()),
            "missed_actives": float(rest.loc[rest["label"] == 1, "max_tanimoto_to_train"].median()),
        },
        "contains_metal_or_unusual_element": {
            f"top{pct}pct_false_positives": int(false_pos["smiles"].map(_has_unusual_element).sum()),
            f"top{pct}pct_hits": int(top.loc[top["label"] == 1, "smiles"].map(_has_unusual_element).sum()),
        },
    }
    result["benchmark_stereo_variant_audit"] = _stereo_variant_audit(out / "split_assignments.csv")
    data.write_json(result, out / "error_analysis.json")
    print(pd.Series(result).to_string())


def _stereo_variant_audit(split_path: Path) -> dict:
    """Benchmark structures that are distinct isomeric SMILES but identical once stereo is removed."""
    split = pd.read_csv(split_path, keep_default_na=False)
    split["flat"] = split["smiles"].map(chemistry.stereo_free_smiles)
    groups = split.groupby("flat")
    multi = groups.filter(lambda g: len(g) > 1).groupby("flat")
    return {
        "stereo_variant_groups": int(multi.ngroups),
        "molecules_in_those_groups": int(multi.size().sum()) if multi.ngroups else 0,
        "groups_spanning_train_and_test": int(sum({"train", "test"} <= set(g["split"]) for _, g in multi)),
        "groups_with_conflicting_labels": int(sum(g["label"].nunique() > 1 for _, g in multi)),
    }


ORGANIC = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}


def _has_unusual_element(smiles: str) -> bool:
    mol = chemistry.mol_from_smiles(smiles)
    return any(atom.GetSymbol() not in ORGANIC for atom in mol.GetAtoms())


def _ring_signature(smiles: str) -> str:
    """Coarse family label from ring counts (total and aromatic)."""
    d = chemistry.descriptors(smiles)
    return f"rings={d['rings']},aromatic={d['aromatic_rings']}"


if __name__ == "__main__":
    main()
