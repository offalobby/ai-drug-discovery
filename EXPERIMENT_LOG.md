# Experiment log — completed with real runs

All numbers below are copied from files in `outputs/` produced by the commands in the README.
Nothing was tuned on the test split. Where a decision was made after seeing a result, it is
marked **[post-hoc]**.

## Protocol registration
- Research question: Can structure-based ML recover more assay-active molecules than random selection at the same 5% test-set screening budget?
- Secondary question: Can the same RDKit fingerprint workflow predict measured aqueous solubility better than a mean-only baseline?
- Data: MoleculeNet HIV assay and Delaney ESOL, downloaded from DeepChem's public S3 bucket (URLs and hashes below).
- Preprocessing: RDKit canonical isomeric SMILES, exclude invalid structures, deduplicate exact canonical structures, remove conflicting activity labels. No salt stripping, tautomer or charge normalization (deliberate; see limitations).
- Partition: DeepChem `ScaffoldSplitter` (Bemis–Murcko) train/validation/test 80/10/10; final test hidden from model selection.
- Model features: RDKit Morgan radius 2, 2,048 bits, chirality encoded (`rdFingerprintGenerator`).
- Models (HIV): three Random Forest configurations (all `class_weight="balanced_subsample"`, `max_features="sqrt"`, wrapped in `dc.models.SklearnModel`):
  `rf_200trees_fulldepth_leaf1`, `rf_500trees_fulldepth_leaf3`, `rf_500trees_depth30_leaf1`.
  Reported but not eligible for selection: logistic regression (C=0.1, balanced) and a molecular-weight-descending heuristic.
- Models (ESOL): mean-only baseline; ridge (alpha 1, 10, 100) on fingerprints; RF on fingerprints (two settings); RF on 10 RDKit descriptors; RF on fingerprints + descriptors.
- Seed: 42. Model selection: validation average precision for HIV; validation MAE for ESOL.
- Primary end metrics: HIV test EF@5%, AP, ROC-AUC, number of hits; ESOL test MAE, RMSE and R².

## Environment
- Run date: 2026-10-23 (UTC 06:15 first run; identical rerun 06:28)
- Python version: 3.10.22 (uv-managed virtual environment)
- RDKit version: 2024.03.5
- DeepChem version: 2.8.0 (CPU, without PyTorch/TensorFlow; only splitters and `SklearnModel` are used)
- scikit-learn version: 1.5.2 (NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1)
- Operating system: Windows 11 (10.0.26200)
- CPU/GPU: Intel64 Family 6 Model 170, 22 logical cores; no GPU used
- Data URL/date and SHA256 hash (downloaded 2026-10-23):
  - `HIV.csv` — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv — `9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22`
  - `delaney-processed.csv` — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv — `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
  - `clintox.csv.gz` (screening library) — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz — `7324a84a180ad9068d3847677c85d58e8432e82b3ae6ac95cf0e98b3a58d9697`
- Reproducibility check: a second full run with the same seed reproduced every validation and test metric exactly.

## Data audit
- Raw molecule rows: 41,127 (source classes: 39,684 CI, 1,039 CM, 404 CA)
- Parsed valid molecules: 41,127 (0 invalid with RDKit 2024.03.5)
- Removed exact duplicates: 0 (all 41,127 canonical isomeric SMILES unique)
- Removed conflicting labels: 0
- Stereo-variant pairs (same structure ignoring stereo): 0
- Percent assay-active: 3.51% (1,443 actives = CM + CA)
- Unique Bemis–Murcko scaffolds: 19,089 (1,582 acyclic molecules have an empty scaffold)
- Train/validation/test counts and active prevalence:

| Split | Molecules | Actives | Prevalence | Unique scaffolds |
|---|---:|---:|---:|---:|
| Train | 32,901 | 1,196 | 3.64% | 10,863 |
| Validation | 4,113 | 133 | 3.23% | 4,113 |
| Test | 4,113 | 114 | 2.77% | 4,113 |

- Non-empty scaffolds shared between train and test: 0. DeepChem fills train with the largest scaffold
  families first, so validation and test consist **entirely of singleton scaffolds**: a deliberately hard,
  "new chemotype" evaluation.
- ESOL: 1,128 rows, 0 invalid; 11 structures had duplicate records — 10 consistent duplicates averaged,
  1 structure with records differing by > 0.5 log units excluded → 1,116 molecules
  (train 892 / valid 112 / test 112). Target range −11.60 to 1.58 log10(mol/L).

## Validation decisions (complete before viewing test results)
- Candidate models and hyperparameters: see Protocol registration.
- Validation AP, EF@5%:

| Model | Val AP | Val ROC-AUC | Val EF@1% | Val EF@5% | Val EF@10% |
|---|---:|---:|---:|---:|---:|
| rf_200trees_fulldepth_leaf1 | 0.296 | 0.810 | 17.67 | 9.43 | 5.38 |
| **rf_500trees_fulldepth_leaf3** | **0.313** | 0.805 | 16.94 | **9.46** | 5.55 |
| rf_500trees_depth30_leaf1 | 0.265 | 0.696 | 17.67 | 8.41 | 4.50 |
| logistic regression (comparator) | 0.254 | 0.738 | 19.14 | 7.06 | 4.35 |
| MW heuristic (comparator) | 0.087 | 0.669 | 5.89 | 4.80 | 3.38 |

- Selected model and explanation: `rf_500trees_fulldepth_leaf3` — highest validation AP (the registered criterion); it also had the highest validation EF@5%. Refit on train + validation (37,014 molecules) before the single test evaluation.
- ESOL: `rf_fp_plus_descriptors` selected (validation MAE 0.721; next best `rf_descriptors` 0.742; best fingerprint-only model `ridge_fp_alpha10` 1.251; mean baseline 1.824).
- Changes made after seeing validation results: none.
- **[post-hoc, disclosed]** After the locked test run, the original starter package was found to declare a different
  set of three RF candidates (150/depth 18/leaf 1, 200/depth 24/leaf 2, 250/full depth/leaf 3). That set was
  run as a separate replication (`--config-set starter_package`, results in `outputs/starter_package_configs/`).
  It does **not** replace the primary result.

## Locked held-out test evaluation
- HIV ROC-AUC: **0.730**
- HIV average precision: **0.298** (random-ranking AP ≈ prevalence = 0.028)
- HIV EF@1% / EF@5% / EF@10%: **22.33 / 7.71 / 4.73**
- Molecules selected in top 5%: 206 (of 4,113)
- Hits recovered in top 5%: **44** (precision 21.4%, recall 38.6% of the 114 test actives)
- Expected random hits at same budget: 5.71 (central 95% range for random selection: 2–11 hits)
- Random-selection p-value (hypergeometric, P[X ≥ 44]): 2.6 × 10⁻²⁹
- Bootstrap EF@5% confidence interval (2,000 resamples of test molecules): **6.08 – 9.43**
  (EF@1%: 16.7 – 28.2; EF@10%: 3.87 – 5.63)
- Comparators on the same test set (refit on train + validation):
  logistic regression AP 0.196, ROC-AUC 0.714, EF@5% 6.31; MW heuristic AP 0.078, ROC-AUC 0.664, EF@5% 4.03.
- Starter-package replication (disclosed, secondary): selected `rf_250trees_fulldepth_leaf3`; test AP 0.299, ROC-AUC 0.726, EF@5% 7.88.
- ESOL selected model: `rf_fp_plus_descriptors` (RF, 500 trees, `max_features=0.3`, on 2,048 Morgan bits + 10 RDKit descriptors)
- ESOL MAE / RMSE / R² (log10 mol/L): **0.598 / 0.749 / 0.810**
  (bootstrap 95% CI: MAE 0.517–0.683, RMSE 0.654–0.850, R² 0.746–0.854).
  Mean-only baseline: 1.496 / 1.933 / −0.265. MAE improvement vs baseline 0.90 (95% CI 0.68–1.12).
  Best fingerprint-only model on test: `rf_fp_leaf1_allfeatures` MAE 1.145, R² 0.190 — fingerprints alone predict solubility poorly.

## Split-sensitivity analysis [post-hoc, no re-tuning]
The selected configuration (`rf_500trees_fulldepth_leaf3`, no re-tuning) was re-evaluated on
alternative partitions of the same cleaned dataset, five seeds each (`outputs/robustness.csv`,
`outputs/plots/robustness_ef5.png`):

| Partition | Test ROC-AUC (mean ± sd) | Test AP | Test EF@5% (mean, range) |
|---|---:|---:|---:|
| DeepChem scaffold split (locked result) | 0.730 | 0.298 | **7.71** |
| Randomized scaffold split, 5 seeds | 0.812 ± 0.011 | 0.377 ± 0.057 | 10.22 (9.85–11.05) |
| Random split, no scaffold grouping, 5 seeds | 0.855 ± 0.019 | 0.460 ± 0.058 | 11.67 (10.75–12.63) |

The locked result is the **most conservative** of the three, as intended: the DeepChem splitter sends
every large scaffold family to training, so the test set is made only of singleton scaffolds. Random
splits, which scatter close analogues across the partitions, inflate EF@5% by roughly 50% and ROC-AUC
by 0.13 — a direct measurement of the leakage optimism that scaffold splitting is meant to prevent.
Enrichment is well above 1 under every partition tested, so the conclusion does not depend on the split.

## Error analysis
- Chemistry/scaffold similarities or activity cliffs:
  - Performance depends strongly on nearest-neighbour Tanimoto similarity to the training pool:
    test AP 0.12 / 0.10 / 0.13 / 0.39 / 0.74 for similarity bins [0–0.3), [0.3–0.4), [0.4–0.5), [0.5–0.7), [0.7–1.0];
    EF@5% within those bins 5.2 / 3.2 / 3.5 / 9.7 / 14.3. Median similarity of top-5% hits 0.68 vs 0.43 for missed actives.
  - Even with a scaffold split, 377 test molecules (9%) have a training neighbour with Tanimoto ≥ 0.7 — scaffold splitting does not remove close analogues with different ring systems.
  - Top-5% hits are mostly confirmed actives: recall 68% for CA (17/25) vs 30% for CM (27/89).
- False positives and false negatives:
  - 162 of the 206 top-5% picks are labelled inactive. The highest-scoring inactives include polysulfonated azo dyes and
    dipyranocoumarins (calanolide/inophyllum-like) — chemical classes that contain many assay actives in training,
    so these look like activity cliffs or assay-noise cases rather than random errors (`outputs/test_top_false_positives.csv`).
  - Top-ranked molecules are large (median MW 562 vs 362 for the rest; top-5% hits median MW 720), echoing the
    MW-heuristic result: part of the signal is "big, polyanionic molecules", a known feature of this assay.
  - Missed actives (`outputs/test_lowest_scored_actives.csv`) are mostly dissimilar to training chemistry (median similarity 0.43).
- Assay and label limitations: single cell-based NCI screen; CM and CA merged into one positive class; no dose–response
  or mechanism; cytotoxic or promiscuous compounds may score as protective or be mislabeled; one record per structure.
- Uncertainty on previously unseen chemistry: the [0–0.5) similarity bins (2,666 test molecules) have AP 0.10–0.13 — still
  above prevalence (≈0.02–0.03) but far weaker than the headline. Scores for dissimilar molecules should be treated as low-confidence.

## Unlabeled screening run
- Source of new molecule library: MoleculeNet ClinTox (FDA-approved drugs and drugs that failed trials for toxicity); its labels were not used.
  Also an 18-row hand-written demo library (`data/sample_library.csv`).
- Count before/after invalid, duplicate, benchmark-overlap filtering: ClinTox 1,484 rows → 4 invalid, 1 dummy-atom (polymer) structure,
  19 duplicates, 100 benchmark overlaps removed → **1,360 ranked**.
  **[post-hoc fix]** The first screen used exact isomeric-SMILES overlap and ranked zidovudine (AZT) #1 although AZT is in the
  HIV training set without stereo labels. Overlap is now also checked on stereo-free SMILES (default), which removed 40 more molecules.
- Number of novel candidates ranked: 1,360
- Top-10 ranked molecule IDs (not 'validated hits'): LIB01028, LIB00632, LIB01184, LIB00934, LIB00932, LIB01002, LIB01005,
  LIB00933, LIB00943, LIB00114 — see `outputs/screened_clintox.csv` (IDs are 1-based row numbers in the ClinTox file).
  By structure, the top ranks include a thymidine stereoisomer, a phenothiazine (mesoridazine-like), kinase inhibitors
  (imatinib, nilotinib, sorafenib, ponatinib-like structures), leflunomide, floxuridine and **emtricitabine** (rank 11),
  an approved HIV nucleoside reverse-transcriptase inhibitor that was not in the benchmark.
- Chemical diversity and physical-property assessment: top-10 contains 8 unique scaffolds; the diversity-filtered shortlist
  (Tanimoto < 0.6 between picks) is flagged in `diverse_shortlist`. Median similarity of the top-10 to training is 0.41
  (library median 0.35), so most candidates are outside the well-predicted high-similarity region. 2 of the top 10 violate
  more than one Lipinski rule (illustrative only).

## Interpret and communicate the results
- Did model screening beat expected random yield at a fixed budget? Yes. 44 actives in the top 5% versus 5.7 expected
  (p ≈ 3 × 10⁻²⁹) on a scaffold-held-out test set.
- Did your experiment reach a roughly 2× EF@5%? It exceeded it: EF@5% = 7.7× (95% bootstrap CI 6.1–9.4×). However,
  a trivial molecular-weight ranking already gives 4.0× and logistic regression 6.3×, so the fair statement is
  "7.7× over random, about 1.9× over a size heuristic". The 2× placeholder was never used as a target.
- What does the experiment NOT establish? Prospective hit rates, antiviral potency, mechanism or target, selectivity,
  cytotoxicity, ADMET, or real laboratory cost savings. EF is a retrospective statistic on one public assay and one
  deterministic split, and it is inflated for test molecules that resemble training chemistry.
- Proposed next step for biological validation (requires a partner laboratory): procure a diversity-filtered shortlist
  of top-ranked, benchmark-novel compounds plus matched low-scoring controls, test in an orthogonal HIV replication assay
  with cytotoxicity counter-screen and dose–response, and compare the observed hit rate with that of the controls.
