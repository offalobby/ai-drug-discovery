import math

import numpy as np
import pandas as pd
import pytest

from src import chemistry, data, metrics
from src.screen import prepare_library


# --- chemistry -------------------------------------------------------------------------------

def test_canonical_smiles_merges_equivalent_notations():
    assert chemistry.canonical_smiles("OCC") == chemistry.canonical_smiles("CCO") == "CCO"
    assert chemistry.canonical_smiles("C1=CC=CC=C1") == "c1ccccc1"


@pytest.mark.parametrize("bad", ["not_a_smiles", "", None, float("nan"), "C1CC"])
def test_invalid_smiles_return_none(bad):
    assert chemistry.canonical_smiles(bad) is None


def test_fingerprint_shape_dtype_and_determinism():
    fp = chemistry.fingerprint("CCO", radius=2, fp_size=2048)
    assert fp.shape == (2048,) and fp.dtype == np.uint8
    assert 0 < fp.sum() < 2048
    assert np.array_equal(fp, chemistry.fingerprint("OCC"))
    assert chemistry.featurize(["CCO", "c1ccccc1"]).shape == (2, 2048)


def test_fingerprint_encodes_chirality():
    r = chemistry.fingerprint("C[C@H](N)C(=O)O")
    s = chemistry.fingerprint("C[C@@H](N)C(=O)O")
    assert not np.array_equal(r, s)


def test_fingerprint_rejects_invalid():
    with pytest.raises(ValueError):
        chemistry.fingerprint("not_a_smiles")


def test_descriptors_known_values():
    d = chemistry.descriptors("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert d["MW"] == pytest.approx(180.16, abs=0.05)
    assert d["HBD"] == 1 and d["passes_rule_of_five"]
    assert chemistry.descriptor_matrix(["CCO"]).shape == (1, len(chemistry.DESCRIPTOR_NAMES))


def test_scaffold_and_similarity():
    assert chemistry.murcko_scaffold("Cc1ccccc1") == "c1ccccc1"
    assert chemistry.murcko_scaffold("CCO") == ""
    fps = chemistry.bit_vectors(["CCO", "c1ccccc1"])
    sims = chemistry.max_tanimoto_to_reference(fps, fps)
    assert np.allclose(sims, 1.0)


# --- metrics ---------------------------------------------------------------------------------

def test_handbook_worked_example_gives_ef_2():
    # N=4000, A=100, budget 5% -> k=200; 10 actives in the top 200 -> EF = 2.0
    y = np.zeros(4000, dtype=int)
    y[:100] = 1
    scores = np.zeros(4000)
    scores[:10] = 1.0          # 10 actives ranked at the top
    scores[100:290] = 0.5      # 190 inactives fill the rest of the top 200
    e = metrics.enrichment(y, scores, 0.05)
    assert e["n_selected"] == 200
    assert e["hits"] == 10
    assert e["expected_random_hits"] == pytest.approx(5.0)
    assert e["enrichment_factor"] == pytest.approx(2.0)
    assert e["precision"] == pytest.approx(0.05)
    assert 0 < e["random_p_value"] < 0.05


def test_ties_at_boundary_use_expectation_and_ignore_row_order():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0])
    # k=2: one sure active, then 1 slot shared by 4 tied molecules containing 1 active -> 0.25
    assert metrics.hits_at_k(y, scores, 2) == pytest.approx(1.25)
    perm = np.random.default_rng(0).permutation(10)
    assert metrics.hits_at_k(y[perm], scores[perm], 2) == pytest.approx(1.25)


def test_random_scores_have_ef_near_one_on_average():
    rng = np.random.default_rng(1)
    efs = []
    for _ in range(200):
        y = (rng.random(2000) < 0.05).astype(int)
        efs.append(metrics.enrichment(y, rng.random(2000), 0.05)["enrichment_factor"])
    assert np.mean(efs) == pytest.approx(1.0, abs=0.1)


def test_perfect_ranking_and_budget_size():
    y = np.array([1] * 5 + [0] * 95)
    e = metrics.enrichment(y, -np.arange(100), 0.05)
    assert e["hits"] == 5 and e["enrichment_factor"] == pytest.approx(20.0)
    assert metrics.budget_size(4000, 0.05) == 200
    assert metrics.budget_size(4001, 0.05) == 201
    assert metrics.budget_size(10, 0.01) == 1


def test_classification_summary_and_bootstrap():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    scores = y + rng.normal(0, 0.7, 1000)
    s = metrics.classification_summary(y, scores)
    assert 0.5 < s["roc_auc"] <= 1 and s["average_precision"] > y.mean()
    b = metrics.bootstrap_ef(y, scores, 0.05, n_boot=200)
    assert b["ci_low"] <= s["enrichment@5%"]["enrichment_factor"] <= b["ci_high"]


def test_regression_summary():
    r = metrics.regression_summary([1, 2, 3], [1, 2, 4])
    assert r["mae"] == pytest.approx(1 / 3)
    assert r["rmse"] == pytest.approx(math.sqrt(1 / 3))


# --- cleaning / splitting / screening --------------------------------------------------------

def test_clean_binary_dataset_handles_invalid_duplicates_and_conflicts():
    df = pd.DataFrame({
        "smiles": ["CCO", "OCC", "c1ccccc1", "C1=CC=CC=C1", "CC(=O)O", "not_a_smiles", None, "CCN"],
        "y": [0, 0, 1, 0, 1, 1, 0, 2],
    })
    clean, audit = data.clean_binary_dataset(df, "smiles", "y")
    assert audit["invalid_smiles"] == 1
    assert audit["missing_smiles_or_label"] == 1
    assert audit["nonbinary_labels"] == 1
    assert audit["conflicting_label_structures_removed"] == 1   # benzene 1 vs 0
    assert audit["redundant_duplicate_rows_removed"] == 1       # ethanol twice, same label
    assert set(clean["smiles"]) == {"CCO", "CC(=O)O"}


def test_clean_regression_dataset_averages_and_drops_inconsistent():
    df = pd.DataFrame({"s": ["CCO", "OCC", "CCC", "C(C)C", "CCN"], "t": [1.0, 1.2, 0.0, 3.0, 2.0]})
    clean, audit = data.clean_regression_dataset(df, "s", "t", max_duplicate_range=0.5)
    assert audit["inconsistent_duplicate_structures_removed"] == 1
    assert clean.set_index("smiles").loc["CCO", "target"] == pytest.approx(1.1)
    assert "CCC" not in set(clean["smiles"])


def test_scaffold_splits_never_share_scaffolds():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1C", "C1CCCCC1CC", "c1ccncc1C",
              "c1ccncc1CC", "C1CCOC1", "c1ccc2ccccc2c1", "c1ccsc1", "C1CCNCC1", "c1cocc1"]
    tr, va, te = data.randomized_scaffold_split(smiles, 0.6, 0.2, seed=3)
    assert len(tr) + len(va) + len(te) == len(smiles)
    sc = [chemistry.murcko_scaffold(s) for s in smiles]
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert not ({sc[i] for i in a} & {sc[i] for i in b})


def test_deepchem_scaffold_split_partitions_all_molecules():
    pytest.importorskip("deepchem")
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1C", "c1ccncc1C", "c1ccsc1",
              "C1CCNCC1", "c1cocc1", "C1CCOC1", "c1ccc2ccccc2c1"]
    tr, va, te = data.deepchem_scaffold_split(smiles, np.zeros(len(smiles)))
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(len(smiles)))


def test_prepare_library_filters_invalid_duplicates_and_benchmark():
    lib = pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f"],
                        "smiles": ["CCO", "OCC", "bad!", "c1ccccc1", "CC(=O)O", "*CC*"]})
    out, counts = prepare_library(lib, benchmark={"c1ccccc1"})
    assert counts == {"input_rows": 6, "invalid_removed": 1, "dummy_atom_structures_removed": 1,
                      "duplicates_removed": 1, "benchmark_overlap": 1, "ranked": 2}
    assert set(out["smiles"]) == {"CCO", "CC(=O)O"}
    kept, _ = prepare_library(lib, benchmark={"c1ccccc1"}, keep_overlap=True)
    assert "c1ccccc1" in set(kept["smiles"])


def test_prepare_library_catches_stereo_variants_of_benchmark_molecules():
    azt_with_stereo = "Cc1cn([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O"
    azt_no_stereo = chemistry.canonical_smiles("Cc1cn(C2CC(N=[N+]=[N-])C(CO)O2)c(=O)[nH]c1=O")
    lib = pd.DataFrame({"name": ["azt", "ethanol"], "smiles": [azt_with_stereo, "CCO"]})
    out, counts = prepare_library(lib, benchmark={azt_no_stereo})
    assert counts["benchmark_overlap"] == 1 and list(out["name"]) == ["ethanol"]
    exact, exact_counts = prepare_library(lib, benchmark={azt_no_stereo}, stereo_insensitive=False)
    assert exact_counts["benchmark_overlap"] == 0 and len(exact) == 2
