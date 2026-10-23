"""Dataset download, auditing/cleaning, scaffold splitting and run provenance."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import canonical_smiles, murcko_scaffold

log = logging.getLogger(__name__)

HIV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ESOL_TARGET = "measured log solubility in mols per litre"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_download(path: Path, url: str) -> dict:
    """Download `url` to `path` if missing; return source provenance (URL, hash, timestamp)."""
    path = Path(path)
    downloaded = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s -> %s", url, path)
        urllib.request.urlretrieve(url, path)
        downloaded = True
    return {
        "path": str(path.as_posix()),
        "url": url,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "downloaded_this_run": downloaded,
        "file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def environment_info() -> dict:
    import deepchem
    import rdkit
    import scipy
    import sklearn

    try:
        import os
        cpu_count = os.cpu_count()
    except Exception:  # pragma: no cover
        cpu_count = None
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": cpu_count,
        "rdkit": rdkit.__version__,
        "deepchem": deepchem.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def clean_binary_dataset(df: pd.DataFrame, smiles_column: str, label_column: str):
    """Canonicalize, drop invalid/nonbinary rows, deduplicate, and drop label conflicts.

    Returns (clean_df with columns [smiles, label, n_source_rows], audit dict). Structures whose
    duplicate records disagree on the label are excluded entirely rather than resolved.
    """
    audit = {"raw_rows": int(len(df))}
    work = df[[smiles_column, label_column]].rename(columns={smiles_column: "raw_smiles",
                                                              label_column: "label"})
    missing = work["raw_smiles"].isna() | work["label"].isna()
    audit["missing_smiles_or_label"] = int(missing.sum())
    work = work[~missing].copy()

    labels = pd.to_numeric(work["label"], errors="coerce")
    nonbinary = ~labels.isin([0, 1])
    audit["nonbinary_labels"] = int(nonbinary.sum())
    work = work[~nonbinary].copy()
    work["label"] = labels[~nonbinary].astype(int)

    work["smiles"] = [canonical_smiles(s) for s in work["raw_smiles"]]
    invalid = work["smiles"].isna()
    audit["invalid_smiles"] = int(invalid.sum())
    audit["invalid_smiles_examples"] = work.loc[invalid, "raw_smiles"].head(10).tolist()
    work = work[~invalid].copy()
    audit["parsed_valid_rows"] = int(len(work))

    grouped = work.groupby("smiles")["label"].agg(["nunique", "size", "first"])
    conflicting = grouped[grouped["nunique"] > 1]
    consistent = grouped[grouped["nunique"] == 1]
    audit["unique_canonical_structures"] = int(len(grouped))
    audit["structures_with_multiple_records"] = int((grouped["size"] > 1).sum())
    audit["redundant_duplicate_rows_removed"] = int((consistent["size"] - 1).sum())
    audit["conflicting_label_structures_removed"] = int(len(conflicting))
    audit["conflicting_label_rows_removed"] = int(conflicting["size"].sum())

    clean = pd.DataFrame({
        "smiles": consistent.index,
        "label": consistent["first"].astype(int).values,
        "n_source_rows": consistent["size"].astype(int).values,
    }).reset_index(drop=True)
    audit["clean_molecules"] = int(len(clean))
    audit["clean_actives"] = int(clean["label"].sum())
    audit["clean_active_fraction"] = float(clean["label"].mean())
    return clean, audit


def clean_regression_dataset(df: pd.DataFrame, smiles_column: str, target_column: str,
                             max_duplicate_range: float = 0.5):
    """Canonicalize and deduplicate a continuous-target dataset.

    Duplicate structures are averaged when their measurements agree within `max_duplicate_range`
    target units and excluded otherwise.
    """
    audit = {"raw_rows": int(len(df))}
    work = df[[smiles_column, target_column]].rename(columns={smiles_column: "raw_smiles",
                                                               target_column: "target"})
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    missing = work["raw_smiles"].isna() | work["target"].isna()
    audit["missing_smiles_or_target"] = int(missing.sum())
    work = work[~missing].copy()
    work["smiles"] = [canonical_smiles(s) for s in work["raw_smiles"]]
    invalid = work["smiles"].isna()
    audit["invalid_smiles"] = int(invalid.sum())
    work = work[~invalid].copy()
    audit["parsed_valid_rows"] = int(len(work))

    grouped = work.groupby("smiles")["target"].agg(["mean", "size", "min", "max"])
    spread = grouped["max"] - grouped["min"]
    inconsistent = grouped[(grouped["size"] > 1) & (spread > max_duplicate_range)]
    kept = grouped.drop(inconsistent.index)
    audit["unique_canonical_structures"] = int(len(grouped))
    audit["structures_with_multiple_records"] = int((grouped["size"] > 1).sum())
    audit["duplicate_rows_averaged_away"] = int((kept["size"] - 1).sum())
    audit["inconsistent_duplicate_structures_removed"] = int(len(inconsistent))
    clean = pd.DataFrame({"smiles": kept.index, "target": kept["mean"].values,
                          "n_source_rows": kept["size"].astype(int).values}).reset_index(drop=True)
    audit["clean_molecules"] = int(len(clean))
    audit["target_mean"] = float(clean["target"].mean())
    audit["target_std"] = float(clean["target"].std())
    audit["target_min"] = float(clean["target"].min())
    audit["target_max"] = float(clean["target"].max())
    return clean, audit


def deepchem_scaffold_split(smiles: list, y: np.ndarray, frac_train=0.8, frac_valid=0.1,
                            frac_test=0.1, seed=42):
    """DeepChem ScaffoldSplitter indices (train, valid, test).

    DeepChem assigns scaffold sets largest-first, so training gets the common scaffolds and the
    test split is dominated by rare/singleton scaffolds. The seed is accepted but has no effect.
    """
    import deepchem as dc

    dataset = dc.data.NumpyDataset(X=np.zeros((len(smiles), 1)), y=np.asarray(y), ids=np.asarray(smiles))
    splitter = dc.splits.ScaffoldSplitter()
    train, valid, test = splitter.split(dataset, frac_train=frac_train, frac_valid=frac_valid,
                                        frac_test=frac_test, seed=seed)
    return np.array(train, dtype=int), np.array(valid, dtype=int), np.array(test, dtype=int)


def randomized_scaffold_split(smiles: list, frac_train=0.8, frac_valid=0.1, seed=0):
    """Scaffold-grouped split with scaffold sets assigned in random order (sensitivity analysis)."""
    scaffolds = pd.Series([murcko_scaffold(s) for s in smiles])
    groups = list(scaffolds.groupby(scaffolds).groups.values())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n = len(smiles)
    train, valid, test = [], [], []
    for gi in order:
        members = list(groups[gi])
        if len(train) + len(members) <= frac_train * n:
            train += members
        elif len(valid) + len(members) <= frac_valid * n:
            valid += members
        else:
            test += members
    return np.array(train, dtype=int), np.array(valid, dtype=int), np.array(test, dtype=int)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value)}")
