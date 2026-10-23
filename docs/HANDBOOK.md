# AI for Drug Discovery — End-to-End Project Handbook

**Stack:** Python 3.10 · RDKit · DeepChem 2.8.0 · scikit-learn · pandas · NumPy · SciPy · Matplotlib  
**Demonstration dataset:** MoleculeNet HIV screening labels (>40,000 compounds in the source dataset)  
**Goal:** Predict *experimental antiviral assay activity* from molecular structure and prioritize candidate compounds by ML ranking.  
**Status:** Educational retrospective screening pipeline, **not** clinical or laboratory confirmation of drug efficacy.  
**Corresponding résumé bullet:** "Applied machine learning with DeepChem and RDKit for molecular property prediction, virtual screening, and model evaluation, improving hit identification efficiency by ~2× over random screening." **The ~2× value must be independently reproduced before it is claimed.**

> **Choice of project:** The résumé description does not identify the original drug target, assay, dataset, model, or protocol. This handbook therefore selects the public MoleculeNet HIV assay as a *fully specified example*. It is not a reconstruction of unknown original experiments. The biological outcome is an assay-defined label, not evidence that a compound is safe, clinically antiviral, or an inhibitor of a particular HIV protein.

## Contents

1. Research problem and hypotheses
2. Skills and conceptual background
3. Installation and folder structure
4. Obtain and audit the data
5. Clean structures and prevent leakage
6. Represent molecules numerically
7. Scaffold split and train baseline models
8. Train, validate and select a supervised model
9. Evaluate on a locked test set
10. Measure twofold enrichment accurately
11. Screen a new unlabeled library
12. Extend to GNNs, descriptors and ADMET
13. Troubleshooting and scientific limitations
14. Deliverables, timeline and presentation
15. Sources and follow-up questions

---

## 1. Research problem and hypotheses

**Scientific question:** Can structural information in small molecules help predict experimentally measured antiviral assay activity, yielding more true actives among the highest-ranked molecules than a random ranking?

**Machine-learning problem:** Binary classification. For each molecule we have:

- `smiles`: an ASCII line notation that encodes molecular structure, e.g. ethanol `CCO`.
- `HIV_active`: 1 (assay-labeled active) or 0 (assay-labeled inactive).

**Prediction:** An ML score for activity (a ranking score; not assumed to be a well-calibrated probability).

**Primary business/scientific metric:** Enrichment Factor at top 5% (EF@5%); also report hits at top 1%, top 10%, average precision, and ROC-AUC.

**Null baseline:** Uniform random selection without replacement from the **same** held-out compound pool, using **the same** screening budget, e.g. screen 5% of molecules in both cases.

**Research hypothesis (not guaranteed):** The model achieves EF@5% > 1; optionally pursue EF@5% >= 2. Do not optimize against or inspect test labels repeatedly in pursuit of a predetermined number.

**Not in scope:** Wet-lab testing, molecular docking, 3D binding affinity prediction, cellular toxicity studies, human efficacy, synthesis plans or clinical safety evaluation.

## 2. Concepts to understand before coding

| Term | What it means | Relevance |
|---|---|---|
| SMILES | A way to write a chemical graph as text | Raw model input |
| RDKit | A cheminformatics toolkit | Parse molecules; compute fingerprints and physical descriptors |
| Morgan fingerprint | Binary vector encoding circular atom environments | Features for model |
| Molecular descriptor | Calculated molecular property such as molecular weight or logP | Interpret/filter candidates |
| DeepChem | Molecular ML utilities for datasets, splits, models | Scaffold splitting and model wrapper |
| Scaffold | Core ring and linker framework | Group compounds to reduce close-analog leakage |
| Random Forest | Ensemble of decision trees | Reproducible initial classification model |
| Virtual screening | Ranking a chemical collection using a predictive model | Prioritization without assay measurements |
| Hit | Experimentally positive compound **for the stated assay** | Retrospective ground truth |
| EF@5% | Model's hit rate in its top 5% divided by full-pool hit rate | Main enrichment claim |

**Important nuance:** A computed descriptor (logP, MW) is **not** a measured property prediction. This core implementation predicts assay-based bioactivity and computes physicochemical descriptors. For an additional *continuous molecular property prediction* module, see §12.

## 3. Installation

### 3.1 Get Python 3.10

Use Miniconda or conda, particularly if installing DeepChem and RDKit on Windows. DeepChem 2.8.0's documented compatibility is most reliable around Python 3.10; newer Python 3.13 may create dependency difficulties.

```bash
conda create -n ai-drug-discovery python=3.10 -y
conda activate ai-drug-discovery
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If not using conda, create and activate an equivalent Python 3.10 `venv` before `pip install -r requirements.txt`.

Quick verification:

```bash
python -c "import deepchem, rdkit, sklearn; print('DeepChem:', deepchem.__version__); print('RDKit:', rdkit.__version__)"
python -m pytest -q
```

The graph neural network extension requires additional Torch/DGL/DGL-LifeSci dependencies. **Do not install them for the baseline project** unless you pursue the optional extension. See [DeepChem's official installation guide](https://deepchem.readthedocs.io/en/2.8.0/get_started/installation.html).

### 3.2 Repository structure

```text
ai-drug-discovery/
├── README.md                      # Project overview and headline results
├── REPORT.md                      # Research-style report
├── EXPERIMENT_LOG.md              # Completed log with real run results and decisions
├── docs/HANDBOOK.md               # This complete handbook
├── requirements.txt               # Pinned CPU dependencies (Python 3.10)
├── data/
│   ├── README.md                  # Sources, SHA256 hashes, download commands
│   ├── sample_library.csv         # Demo molecules for screening
│   └── HIV.csv, delaney-processed.csv, clintox.csv.gz   # Downloaded (not committed)
├── src/
│   ├── chemistry.py               # RDKit parsing/descriptors/fingerprints/scaffolds
│   ├── data.py                    # Download, audit/cleaning, scaffold splits, provenance
│   ├── metrics.py                 # AP, ROC-AUC, EF, hypergeometric p, bootstrap
│   ├── plots.py                   # Figures from evaluated results
│   ├── train.py                   # Experiment A: cleaning, split, selection, locked test
│   ├── solubility.py              # Experiment B: ESOL regression
│   ├── screen.py                  # Rank an external unlabeled compound library
│   ├── robustness.py              # Split-sensitivity analysis (no re-tuning)
│   └── analysis.py                # Post-hoc error analysis of locked test predictions
├── tests/
│   └── test_chemistry_metrics.py
└── outputs/                       # Results committed; model .joblib files are not
```

### 3.3 How to run, in order

From the repository **root**:

```bash
python -m pytest -q
python -m src.train
python -m src.solubility
python -m src.screen --library data/sample_library.csv
python -m src.robustness      # optional: split-sensitivity analysis
python -m src.analysis        # optional: post-hoc error analysis
```

The HIV training command downloads the public HIV CSV only if `data/HIV.csv` is missing. The ESOL regression command independently downloads the Delaney CSV (same DeepChem S3 source that `dc.molnet.load_delaney` uses) if `data/delaney-processed.csv` is missing. An internet connection is needed for the initial download; datasets and predictions are processed locally thereafter.

---

## 4. Obtain and audit the data

**Dataset:** MoleculeNet HIV, based on the U.S. NCI Drug Therapeutics Program AIDS antiviral assay. The source distinguishes confirmed active, moderately active and confirmed inactive; the commonly used MoleculeNet binary outcome treats confirmed + moderately active as active. [DeepChem HIV loader/source](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/hiv_datasets.py).

**Official CSV URL** (used by DeepChem itself):

```text
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv
```

Expected columns: `smiles`, `activity`, `HIV_active`. This implementation selects `smiles` and `HIV_active`. Counts change after removing invalid, duplicated and conflicting records.

**Data audit questions** (record answers in your report):

1. How many rows? How many SMILES strings parse successfully?
2. What fraction of the compounds are active? Is the problem imbalanced?
3. Are molecular structures duplicated or represented by different equivalent SMILES?
4. Do duplicate structures have conflicting labels?
5. How many unique structural scaffolds exist?
6. Does the dataset actually reflect a single consistent assay condition?
7. What limitations follow from merging the source's moderate and confirmed actives?

You can substitute a CSV from a different **public** bioactivity dataset (with binary activity labels) using:

```bash
python -m src.train --csv data/my_assay.csv \
  --smiles-column smiles --label-column active
```

Do not substitute data with continuous pIC50 without changing the task to regression and adapting metrics.

## 5. Chemical cleaning and leakage controls

`src/chemistry.py` parses each SMILES using `Chem.MolFromSmiles`, then canonicalizes it via `Chem.MolToSmiles(..., canonical=True, isomericSmiles=True)`.

`src/train.py` then:

1. Removes missing, invalid, nonbinary and unparseable rows.
2. Deduplicates by canonical SMILES.
3. **Excludes all structures with conflicting binary labels** rather than choosing a favorable label.
4. Creates train/validation/test groups by **Bemis–Murcko molecular scaffold** with DeepChem's `ScaffoldSplitter` (80/10/10 nominal split fractions).
5. Writes `outputs/split_assignments.csv` so you can audit the exact assignments.

Why scaffold instead of random? A random split may put near-analog chemical structures in both training and evaluation, yielding deceptively optimistic generalization. Scaffold splitting is still imperfect (e.g., different scaffolds can remain chemically related; assay artifacts and dataset shift persist).

**Chemical standardization caveat:** This starter implementation deliberately does **not** collapse salts, merge tautomers or normalize ionization states. Those are scientifically meaningful choices. For a publishable project, define and record a standardization policy, use RDKit MolStandardize when appropriate, and evaluate how many structures and labels the policy changes. Never use test labels to guide standardization decisions.

**No leakage checklist:**

- Do not featurize with label-derived variables.
- Never tune hyperparameters, cutoff thresholds or descriptor selection on the held-out test set.
- Keep duplicates out of multiple splits.
- For prospective screening, remove any compounds also present in your benchmark, even if they belonged to its held-out test subset.

## 6. Featurize molecules

For every canonical SMILES, `src/chemistry.py` computes a **2048-bit Morgan circular fingerprint** with radius 2 (ECFP-like representation) and chirality enabled, using RDKit's current `rdFingerprintGenerator.GetMorganGenerator` API.

Example:

```python
from src.chemistry import fingerprint, descriptors

fp = fingerprint("CCO", radius=2, fp_size=2048)
print(fp.shape)                  # (2048,)
print(fp.sum())                  # count of set bits
print(descriptors("CCO"))
```

Each bit indicates whether a particular atom environment is present. The feature matrix has dimensions `(number_of_molecules, 2048)`.

RDKit descriptors used for screening interpretation:

- molecular weight (MW)
- calculated octanol/water logP
- topological polar surface area (TPSA)
- H-bond donors / acceptors
- rotatable bonds
- illustrative Lipinski Rule-of-Five flag

The Rule-of-Five flag is **not a drug efficacy, safety, developability or ADMET prediction**. Compounds may be viable drug candidates while violating these guidelines; meeting them is not proof of clinical suitability.

### Optional descriptor model

As an **additional supervised model**, you could concatenate standardized descriptors with fingerprints, then re-run training/validation/test from the same split. Fit any scaler on the **training partition only** (then refit on train+validation following model selection). Compare results rather than assuming more features perform better.

## 7. The scaffold split and baselines

The code creates `dc.data.NumpyDataset` with molecule SMILES stored in `ids`, then calls:

```python
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42
)
```

**Use three baselines/controls:**

- **Random ranking:** No learning; the expected hit rate in the selected set equals the overall active fraction. This is the main comparison for EF.
- **Simple physicochemical heuristic:** Rank by a predeclared descriptor rule (only if scientifically justified). This may perform poorly or have no meaningful biological relationship to activity.
- **Random Forest fingerprints:** A competent ML baseline. An optional graph neural network should be compared against this model on **identical molecule splits**.

### Why not use plain accuracy?

In an imbalanced assay, a model that predicts every molecule is inactive can show high accuracy while finding **zero** useful active molecules. The central metrics must reflect ranking and recovery of positive compounds.

## 8. Train, tune and select a model

Main implementation: scikit-learn `RandomForestClassifier`, wrapped as `dc.models.SklearnModel`, trained on `dc.data.NumpyDataset` fingerprint arrays. This is a legitimate DeepChem model integration rather than merely importing the library.

`src/train.py` evaluates three prespecified RF configurations that vary number of trees, tree depth and leaf size. During training:

1. `train` is used to fit each candidate.
2. `valid` is used to compute **average precision** and EF@5%.
3. The configuration with the highest validation average precision is selected.
4. The chosen configuration is refit on **train + validation**.
5. Exactly one final evaluation occurs on the held-out `test` split.

Why choose average precision? It evaluates ranking quality for the positive class over different score thresholds and is especially useful in imbalanced classification. The script also records validation EF@5%, but **does not pick the winner based on the test set**.

**Scores** from `predict_proba(... )[:, 1]` are used to rank molecules. In an uncalibrated, class-weighted Random Forest, these scores are not necessarily well-calibrated probabilities. If probability estimates are needed, consider post-hoc calibration using a separate validation or calibration subset and a properly designed evaluation protocol.

**Reproducibility:** Set random state (`42`), keep exact split assignments, record package versions, preserve raw data/source URLs, export code and configuration. For results intended for publication, use multiple scaffold splits or external assay holdouts rather than relying on one deterministic split.

## 9. Evaluate final performance

After training, open `outputs/metrics.json`. It includes:

| Metric | Interpretation | Better |
|---|---|---|
| ROC-AUC | Probability a randomly picked active scores above a random inactive | Higher |
| Average precision | Precision-recall ranking summary | Higher |
| Hits@5% | Number of labeled actives among top 5% of scored test compounds | Higher |
| Precision@5% | Hits@5% divided by number selected | Higher |
| EF@5% | Precision@5% / overall test prevalence | Above 1 indicates enrichment |
| Random-hit p-value | Probability random selection obtains at least as many hits | Lower provides evidence against random ranking |
| Bootstrap EF@5% CI | Approximate uncertainty from resampling held-out compounds | Report interval, not just point estimate |

Outputs:

```text
outputs/
├── metrics.json                    # audit, splits, validation, locked test, comparators, provenance
├── validation_trials.csv
├── split_assignments.csv
├── test_ranked.csv                 # test molecules ranked, with similarity to training pool
├── test_top_false_positives.csv
├── test_lowest_scored_actives.csv
├── plots/
│   ├── test_curves.png             # ROC, PR, hit recovery, EF@k vs random
│   ├── validation_trials.png
│   └── test_by_similarity.png
└── artifacts/
    ├── model.joblib                # not committed (reproducible)
    └── all_benchmark_smiles.txt
```

**Inspect these files**, do not just present the best number: compare class prevalences across splits, see how many hits were found in the top 1%, 5% and 10%, and evaluate how performance changes by scaffold families and chemical similarity when possible.

### Interpretation of ROC-AUC vs. hit enrichment

A respectable ROC-AUC does not guarantee useful top-k early enrichment. Drug-screening budgets generally limit how many compounds can be evaluated, making EF@k and hit counts directly relevant.

### Optional further tests

- Multiple random/scaffold partitions, with mean and standard deviation of EF.
- Diversity among top predictions (cluster by Tanimoto similarity).
- Confidence/uncertainty (bootstrap, ensemble disagreement or conformal methods adapted to data).
- External holdout (different publication/assay series), particularly if transferring beyond MoleculeNet HIV.
- Precision-recall curves by scaffold subgroup.

## 10. Quantify the 2× claim precisely

Let:

- `N` = number of molecules in the **held-out test pool**.
- `A` = number of true actives in that pool.
- `k = ceil(0.05 × N)` = molecules you can screen under a top-5% budget.
- `H` = actual number of actives among the model's top `k` predicted compounds.

Then:

```text
Test prevalence             = A / N
Model hit rate at 5%        = H / k
Expected random hits        = k × A / N
Enrichment Factor EF@5%     = (H / k) / (A / N)
                           = H / expected_random_hits
```

### Worked arithmetic: a **hypothetical** example

If test `N=4,000` compounds, `A=100` actives (2.5% prevalence), and you can screen `k=200` compounds:

- Random selection **expects** `200 × 0.025 = 5` true actives.
- Suppose the ML model recovers **10** true actives in its top 200.
- Precision@5% = `10/200 = 5%`.
- EF@5% = `5% / 2.5% = 2.0`.

**Permissible conclusion from these hypothetical numbers:** "The model achieved 2.0× enrichment in observed assay positives relative to expected random selection at a fixed 5% screening budget on a held-out set."

**Not justified from that alone:** "We reduced laboratory costs by 50%, doubled the drug discovery success rate, discovered 10 drugs, validated antiviral activity prospectively, or doubled model accuracy."

The script reports exact random-selection uncertainty using a **hypergeometric model**: random choice of `k` compounds without replacement from `N`, with `A` positives. It also calculates a simple bootstrap uncertainty interval for EF@5%. These quantify retrospective ranking uncertainty, not causal improvements in real experimental throughput.

When EF < 2, say so and analyze why. An honest result below the target is still a valid scientific project.

## 11. Virtual screen a new chemical library

Create a CSV with at least `smiles` and optionally `name`:

```csv
name,smiles
molecule_A,CCO
molecule_B,c1ccccc1
molecule_C,CC(=O)O
```

After training, run:

```bash
python -m src.screen --library data/sample_library.csv
```

The code will:

1. Read the library and canonicalize structures.
2. Remove invalid structures and duplicate canonical SMILES.
3. Exclude molecules appearing **anywhere in the original benchmark** (training, validation, or test), by default.
4. Compute identical Morgan fingerprint features used in training.
5. Score with the fitted model and rank descending.
6. Add RDKit physicochemical descriptors and a simple Rule-of-Five indicator.
7. Export `outputs/screened_library.csv`.

**Retrospective vs prospective:** Held-out test molecules have known experimental labels and can support hit-enrichment metrics. New library molecules do not have known experimental labels: scoring them is *virtual prioritization*, **not** evidence of an actual experimental hit. Any newly predicted hits need orthogonal experimental confirmation, reproducibility checks, toxicity screens and downstream medicinal chemistry assessment.

**Do not select exclusively by high ML score:** Consider chemical diversity, domain of applicability, known assay interference, availability, and physicochemical constraints. Lipinski indicators are preliminary, not definitive filters.

### Suggested candidate table for reporting

| Rank | Compound ID | Canonical SMILES | ML ranking score | MW | logP | HBD/HBA | Novel vs benchmark |
|---|---|---|---|---|---|---|---|
| 1 | C001 | *Exported by code* | *Output* | *Output* | *Output* | *Output* | Yes |
| 2 | C002 | *Exported by code* | *Output* | *Output* | *Output* | *Output* | Yes |

Only the actual execution should populate molecule names, scores and rankings; the table above is a display template, **not predicted results**.

## 12. Extensions for a stronger project

### A. DeepChem graph neural network (optional advanced track)

A Graph Convolutional Network learns representations of atoms and bonds rather than starting from fixed Morgan bits. DeepChem has a `MolGraphConvFeaturizer` and a `GCNModel`, but the latter requires compatible **PyTorch, DGL and DGL-LifeSci**, whose installation varies by OS/version. Consult the [GCNModel source and example](https://github.com/deepchem/deepchem/blob/master/deepchem/models/torch_models/gcn.py) and [featurizer docs](https://deepchem.readthedocs.io/en/stable/api_reference/featurizers.html).

Conceptual skeleton (run only after configuring the optional graph dependencies):

```python
import deepchem as dc

smiles = ["CCO", "CCN", "c1ccccc1"]
labels = [0, 1, 0]
featurizer = dc.feat.MolGraphConvFeaturizer()
graphs = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(X=graphs, y=labels)
model = dc.models.GCNModel(n_tasks=1, mode="classification", batch_size=16)
model.fit(dataset, nb_epoch=5)
print(model.predict(dataset))
```

For a fair study, run the GCN on **the exact same saved train/valid/test molecule sets** and report its own hyperparameters and test enrichment. This snippet is a tiny API illustration, not a scientifically meaningful benchmark.

### B. Separate measured molecular property prediction

**Implemented as `src/solubility.py`**. It uses the separate public **ESOL/Delaney solubility** dataset (the CSV behind `dc.molnet.load_delaney`, loaded directly so the same RDKit cleaning and chirality-aware fingerprints as Experiment A apply). The target is *measured log10 aqueous solubility (mol/L)*, not the HIV activity label. RDKit calculates the same 2,048-bit Morgan fingerprints; DeepChem provides a scaffold-based train/validation/test split and a scikit-learn model wrapper.

Run:

```bash
python -m src.solubility
```

The code compares a **mean-only baseline**, **ridge regression**, and **Random Forest regression**, selects by **validation MAE**, then evaluates just once on test data. It exports `outputs/solubility_metrics.json`, `outputs/solubility_test_predictions.csv`, `outputs/plots/solubility_scatter.png`, and `outputs/artifacts/solubility_model.joblib`. **Important detail:** `transformers=[]` disables DeepChem's default target normalization, so test MAE and RMSE are reported in the measured log10(mol/L) units, not normalized units. The outcome is a measured physicochemical property relevant to drug discovery, not proof of biological activity. If you want better domain controls, also compare a model trained on RDKit molecular descriptors and a random-split sensitivity check; those would need additional code and explicit reporting. [DeepChem getting-started tutorial](https://deepchem.io/tutorials/).

### C. Better molecular chemistry

- Use a documented salt, stereochemistry, tautomer and charge protocol.
- Inspect activity cliffs and near-duplicate pairs.
- Calculate Tanimoto similarity between predictions and training compounds.
- Track scaffold diversity and avoid selecting 20 nearly identical top-ranked structures.
- Consider applicability-domain labeling or uncertainty intervals for novel chemotypes.

### D. Better comparison and robustness

- Logistic regression and gradient-boosted trees as non-deep-learning baselines.
- Hyperparameter tuning with nested group-aware validation.
- Multiple scaffold splits, repeated benchmarks, and external assay datasets.
- Experiment tracking, immutable data manifests and pinned environments.
- Model calibration only on properly separated data.

### E. What lab validation would require

If you were actually working with a laboratory, top-ranked molecules would need procurement or synthesis, assay controls, independent verification and assays designed to identify artifacts. **Do not substitute computational scores for experimental evidence**. These activities are not included in this project.

## 13. Troubleshooting

| Problem | Likely cause | First action |
|---|---|---|
| `ModuleNotFoundError: deepchem` | Wrong Python environment / incomplete install | Activate Python 3.10 conda environment, reinstall requirements |
| `ModuleNotFoundError: src` | Running from the wrong directory or running `pytest` directly | `cd` into project root and use `python -m pytest` / `python -m src.train` |
| Invalid SMILES warnings | Dataset contains malformed records | Review cleaning log; record removed counts |
| One split contains only one class | Very small or strongly imbalanced custom dataset | Use a larger dataset; inspect scaffold distribution; do not use ROC-AUC on one-class split |
| Download fails | Initial internet / source access needed | Download the URL manually into `data/HIV.csv` then rerun |
| Memory or slow training | Dense 2048-bit feature matrix and many RF trees | Use a machine with sufficient RAM; for exploration use `--fp-size 1024` (record changed protocol) |
| Unusual model scores | Imbalance and lack of calibration | Evaluate ranking metrics and do not interpret score as probability |
| Unexpectedly excellent validation scores | Possible duplicates, leakage, or scaffold similarity | Audit canonical SMILES, partition boundaries and label columns |
| Screening outputs no molecules | All structures invalid or known to benchmark | Provide genuinely unseen valid molecules; check exclusion counts |
| GCN missing DGL | Advanced dependency not installed | Keep CPU Random Forest workflow until graph dependencies are configured |

**Safety:** `joblib.load` unpickles arbitrary code. Only load your own trusted model artifact, never a model downloaded from an untrusted site.

**Reproducibility note:** The full pipeline has been run in the documented environment; actual results, versions and data hashes are in `EXPERIMENT_LOG.md`, `REPORT.md` and `outputs/`. This handbook itself describes the method, not the results.

## 14. Deliverables and project timeline

### Minimum credible finished deliverables

- [ ] Version-controlled repository with licensing and source links.
- [ ] Data audit table and documented structure-cleaning decisions.
- [ ] Partition CSV with scaffold-aware 80/10/10 split.
- [ ] Trained model artifact and reproducible hyperparameters.
- [ ] Validation comparison of at least 3 model configurations.
- [ ] Frozen-test ROC-AUC, AP, EF@1%, EF@5% and EF@10%.
- [ ] Random-screen expected hit count and inferential uncertainty.
- [ ] Saved ROC and precision-recall figures.
- [ ] Unlabeled library ranking CSV with RDKit descriptors.
- [ ] Separate ESOL property-regression holdout evaluation (MAE, RMSE, R²).
- [ ] Completed `EXPERIMENT_LOG.md` containing real run results and environment versions.
- [ ] Report presenting limitations, conclusions and next experiments.

### A realistic project progression for Spring 2025–Spring 2026

This is a **suggested historical timeline to organize documentation**, not a statement of what happened in those months.

| Phase | Approximate time | Work | Deliverable |
|---|---|---|---|
| 1 — Research framing | Spring 2025 | Literature review, ML objectives, benchmark selection | One-page experiment proposal |
| 2 — Data collection | Early summer 2025 | Assay source, label audit, class imbalance | Dataset card |
| 3 — RDKit pipeline | Summer 2025 | Structure validation, fingerprints/descriptors | Clean feature pipeline |
| 4 — Baseline modeling | Fall 2025 | Random Forest and split controls | Initial validation report |
| 5 — Optimization | Late fall 2025 | Hyperparameter selection and error analysis | Frozen model settings |
| 6 — Evaluation | Winter 2025–2026 | Locked holdout, enrichment vs random, uncertainty | Quantified comparison |
| 7 — Screening | Early spring 2026 | Rank new compounds, chemical filtering | Candidate shortlist |
| 8 — Final report | Spring 2026 | Figures, methodology, code, limitations | Public portfolio case study |

### Suggested final report structure (8–12 pages)

1. **Abstract:** problem, data, high-level method, validated metrics (200–250 words).
2. **Background:** why early screening is expensive; why chemical ML helps; limits of assay labels.
3. **Dataset and ethics:** experimental source, label definitions, data quality and licensing.
4. **Methodology:** cleaning, fingerprints, scaffold splits, models and validation protocol.
5. **Results:** class prevalence; AP and ROC-AUC; EF@k; random baseline and uncertainty.
6. **Case study:** top-ranked distinct molecules from unlabeled screen; descriptors.
7. **Discussion:** potential screening efficiency and major reasons predictions may fail.
8. **Conclusion/future work:** prospective assay testing, calibration, GNNs, external validation.
9. **Appendix:** software versions, seed, parameters, dataset hash, linked repository.

### Defensible résumé phrasing

**Only if the test results actually support ~2× EF@k:**

> Developed a reproducible RDKit/DeepChem molecular bioactivity screening pipeline using 2,048-bit Morgan fingerprints, scaffold-aware validation and Random Forest classification; achieved ~2× early hit enrichment over expected random selection at a fixed top-k screening budget on a held-out benchmark.

**If the reported enrichment turns out to differ:**

> Developed a reproducible RDKit/DeepChem molecular screening pipeline for assay-activity classification, scaffold-aware evaluation, and retrospective hit-enrichment benchmarking against random selection.

### Questions you should be able to answer in an interview

1. What was the **biological activity label**, and who measured it?
2. Why did you choose Morgan fingerprints rather than raw SMILES or a GNN?
3. Which steps in RDKit converted a molecule to input features?
4. How were duplicate structures and conflicting labels handled?
5. What does a scaffold split guard against, and what does it **not** guard against?
6. How did you deal with class imbalance? Why not use accuracy?
7. Explain your **exact denominator** in “2× improvement.”
8. How many true hits were in the top selected molecules, and what would random selection have yielded?
9. Why might average precision and EF@5% tell different stories?
10. Were any high-scoring candidates actually assayed in a laboratory?
11. What were the main false positives, and what chemistry or assay effects could cause them?
12. What would you change to make the results more generalizable or publication-ready?

## 15. Source documentation

1. [DeepChem MoleculeNet HIV loader, label definitions and data URL](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/hiv_datasets.py).
2. [DeepChem 2.8.0 installation](https://deepchem.readthedocs.io/en/2.8.0/get_started/installation.html).
3. [DeepChem ScaffoldSplitter API](https://deepchem.readthedocs.io/en/2.8.0/api_reference/splitters.html).
4. [DeepChem SklearnModel API](https://deepchem.readthedocs.io/en/2.8.0/api_reference/models.html).
5. [RDKit Morgan fingerprint generator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).
6. [DeepChem GCN example and dependencies](https://github.com/deepchem/deepchem/blob/master/deepchem/models/torch_models/gcn.py).
7. [DeepChem molecular ML tutorials](https://deepchem.io/tutorials/).

---

**Final caution:** An in-silico screen is a useful *prioritization system*. It does not establish antiviral potency, target-specific binding, experimental hit validity, toxicity or clinical drug efficacy. All reported performance numbers must come from reproducible runs and fixed evaluation protocols.
