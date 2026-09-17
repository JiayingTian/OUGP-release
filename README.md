# OUGP + TTA IID

This README covers the five-case IID experiment only. Its configuration is
`configs/iid/exp_iid_main_table.yaml`. The matrix contains
Cora, PubMed, Cora Full, DBLP, and ACM with four seeds each. Pre-existing OOD
files in this working directory are not used by this command. Raw datasets and
model checkpoints are not included.

## Environment

From the release root:

```bash
conda env create -f environment.yml
conda activate ougp
export PYTHON_BIN="$(command -v python)"
```

The launcher uses `PYTHON_BIN` when it is set; otherwise it resolves `python`
from the active environment.

## Datasets

Use paths relative to the release root. No dataset files are shipped.

- Cora and PubMed: the loader downloads Planetoid files to
  `data/raw/cora/raw/` and `data/raw/pubmed/raw/` on first use. It parses the
  raw files and row-normalizes features.
- Cora Full and DBLP: PyTorch Geometric `CitationFull` downloads and processes
  them under `data/raw/pyg/citationfull_cora/` and
  `data/raw/pyg/citationfull_dblp/` on first use.
- ACM: PyTorch Geometric `HGBDataset` downloads and processes it under
  `data/raw/hgb/` on first use. The OUGP loader selects the labeled node type
  and its within-type citation edges.

The experiment entrypoint applies the configured `stratified_2_1_1` split.
First use requires network access; for offline runs, populate those same cache
directories in advance using the corresponding public dataset sources and the
installed PyTorch Geometric version. Keep data under `data/raw/`, not under a
personal home-directory path.

## Run

```bash
bash scripts/launch_exp.sh configs/iid/exp_iid_main_table.yaml
```

The output root is `experiments/iid_main_table/`. The summary command
also reads the included baseline table at
`experiments/exp206_main_table_baselines_forward_flops_clean_211/main_table_acc_flops_ratio.csv`.

The IID launcher does not wait for OOD-only data. The summary script includes
baseline columns for all five IID datasets.
