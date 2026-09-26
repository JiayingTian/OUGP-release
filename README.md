# OUGP Release: IID and OOD Para-TTA

This release keeps the OUGP source-side training and deployment path unchanged.
The default TTA path is forward-only ESPM (Energy-Shift Parameter Mask): it
measures target feature channel energy, computes a channel-wise shift, and
re-ranks a fixed-budget parameter mask. It does not use target labels,
loss, backward, optimizer steps, controller training, output MLP, or prediction
propagation. The pruning budget is preserved. IID disables the energy gate; OOD disables it for source deployment and enables it only for target ESPM mask re-ranking.

The release includes these entry configurations:

- `configs/iid/exp_iid_main_table.yaml`: Cora, PubMed, Cora Full, DBLP, and ACM.
- `configs/ood/exp_citation_dblp_to_acm.yaml`: DBLPv8 to ACMv9.
- `configs/ood/exp_citation_acm_to_dblp.yaml`: ACMv9 to DBLPv8.

Raw datasets and model checkpoints are not included. All paths are relative to
the release root; no personal home-directory paths are required.

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
- Citation OOD: place the UDAGCN DBLPv8 and ACMv9 raw files under
  `data/raw/udagcn/` using the directory names expected by the citation
  loader. These files are not downloaded by the release launcher.
- Twitch OOD: place the Twitch domain files under the configured relative data
  root before launching the matrix. The release launcher does not copy data
  from another checkout.

The experiment entrypoint applies the configured `stratified_2_1_1` split.
First use requires network access; for offline runs, populate those same cache
directories in advance using the corresponding public dataset sources and the
installed PyTorch Geometric version. Keep data under `data/raw/`, not under a
personal home-directory path.

## Run

```bash
bash scripts/launch_exp.sh configs/iid/exp_iid_main_table.yaml
bash scripts/launch_exp.sh configs/ood/exp_citation_dblp_to_acm.yaml
bash scripts/launch_exp.sh configs/ood/exp_citation_acm_to_dblp.yaml
bash scripts/launch_exp.sh configs/ood/exp_twitch_espm.yaml
```

Use `--dry-run` after a configuration to inspect the generated commands without
starting jobs. Outputs are written below the experiment root declared by each
configuration.

For reproducible launches, set `PYTHON_BIN` explicitly to the intended
environment and keep each configuration's seed list unchanged.
