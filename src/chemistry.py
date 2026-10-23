"""RDKit helpers: SMILES parsing/canonicalization, Morgan fingerprints, descriptors, scaffolds."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Invalid SMILES are counted and reported by the cleaning code; silence per-molecule RDKit spam.
RDLogger.DisableLog("rdApp.*")

DEFAULT_RADIUS = 2
DEFAULT_FP_SIZE = 2048


def mol_from_smiles(smiles) -> Optional[Chem.Mol]:
    """Parse a SMILES string; return None for missing, empty or unparseable input."""
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(smiles) -> Optional[str]:
    """RDKit canonical isomeric SMILES, or None if the input cannot be parsed."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def stereo_free_smiles(smiles) -> Optional[str]:
    """Canonical SMILES with stereochemistry removed (for stereo-insensitive identity checks)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


@lru_cache(maxsize=8)
def _morgan_generator(radius: int, fp_size: int):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size, includeChirality=True
    )


def fingerprint(smiles: str, radius: int = DEFAULT_RADIUS, fp_size: int = DEFAULT_FP_SIZE) -> np.ndarray:
    """Binary Morgan (ECFP-like) fingerprint with chirality, as a uint8 vector of length fp_size."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return _morgan_generator(radius, fp_size).GetFingerprintAsNumPy(mol).astype(np.uint8)


def featurize(smiles_list: Iterable[str], radius: int = DEFAULT_RADIUS,
              fp_size: int = DEFAULT_FP_SIZE) -> np.ndarray:
    """Stack fingerprints into an (n_molecules, fp_size) uint8 matrix."""
    rows = [fingerprint(s, radius, fp_size) for s in smiles_list]
    if not rows:
        return np.zeros((0, fp_size), dtype=np.uint8)
    return np.vstack(rows)


def bit_vectors(smiles_list: Iterable[str], radius: int = DEFAULT_RADIUS, fp_size: int = DEFAULT_FP_SIZE):
    """RDKit ExplicitBitVect fingerprints (same definition as `fingerprint`) for Tanimoto similarity."""
    gen = _morgan_generator(radius, fp_size)
    return [gen.GetFingerprint(mol_from_smiles(s)) for s in smiles_list]


def max_tanimoto_to_reference(query_fps, reference_fps) -> np.ndarray:
    """For each query fingerprint, the maximum Tanimoto similarity to any reference fingerprint."""
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)) for fp in query_fps])


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES (empty string for acyclic molecules)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "rotatable_bonds", "rings",
                    "aromatic_rings", "heavy_atoms", "fraction_csp3"]


def descriptors(smiles: str) -> dict:
    """Computed (not measured) RDKit physicochemical descriptors plus an illustrative Lipinski flag."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    values = {
        "MW": Descriptors.MolWt(mol),
        "logP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    violations = sum([values["MW"] > 500, values["logP"] > 5, values["HBD"] > 5, values["HBA"] > 10])
    values["lipinski_violations"] = int(violations)
    # Illustrative only: Ro5 is a heuristic, not an efficacy/safety/ADMET prediction.
    values["passes_rule_of_five"] = bool(violations <= 1)
    return values


def descriptor_matrix(smiles_list: Iterable[str]) -> np.ndarray:
    """(n_molecules, len(DESCRIPTOR_NAMES)) float matrix of RDKit descriptors."""
    return np.array([[descriptors(s)[k] for k in DESCRIPTOR_NAMES] for s in smiles_list], dtype=float)
