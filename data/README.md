# Data

Raw datasets are public and are **not committed**; scripts download them on first run
(or download manually into this folder). Hashes below are from the files used for the
reported results (downloaded 2026-10-23). If a hash differs, the upstream file changed and
results may not reproduce exactly.

| File | Source URL | Used by | SHA256 |
|---|---|---|---|
| `HIV.csv` | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv | `src.train` (Experiment A) | `9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22` |
| `delaney-processed.csv` | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv | `src.solubility` (Experiment B) | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |
| `clintox.csv.gz` | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz | `src.screen` external library (labels ignored) | `7324a84a180ad9068d3847677c85d58e8432e82b3ae6ac95cf0e98b3a58d9697` |
| `sample_library.csv` | Hand-written demo (committed) | `src.screen` demo | — |

* **HIV** — MoleculeNet HIV, from the NCI/DTP AIDS Antiviral Screen. Columns `smiles`,
  `activity` (CI = confirmed inactive, CM = confirmed moderately active, CA = confirmed active),
  `HIV_active` (1 for CM or CA). See the DeepChem loader:
  https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/hiv_datasets.py
* **ESOL** — Delaney (2004) J. Chem. Inf. Comput. Sci. 44, 1000-1005. Target column
  `measured log solubility in mols per litre` (log10 mol/L).
* **ClinTox** — FDA-approved drugs and drugs that failed clinical trials for toxicity
  (MoleculeNet). Used here **only as a source of unlabeled drug-like structures** to screen; its
  labels are not used.
* `sample_library.csv` — a handful of well-known molecules (including approved antiretrovirals
  and one deliberately invalid SMILES and one duplicate notation) to demonstrate the screening
  filters. Most of these are removed as benchmark overlaps, which is the intended behaviour.

Download commands:

```bash
curl -L -o data/HIV.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv
curl -L -o data/delaney-processed.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
curl -L -o data/clintox.csv.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz
```
