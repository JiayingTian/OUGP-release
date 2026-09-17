"""Graph dataset loading for OUGP case studies."""

from __future__ import annotations

import pickle
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


PLANETOID_BASE_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")
AMAZON_PHOTO_URL = (
    "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
    "amazon_electronics_photo.npz"
)
BENCHMARK_NPZ_URLS = {
    "amazon_computers": (
        "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
        "amazon_electronics_computers.npz"
    ),
    "coauthor_cs": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz",
}
ACTOR_BASE_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film"
PYG_DATASET_SPECS = {
    "coauthor_physics": ("coauthor", "Physics"),
    "wiki_cs": ("wikics", ""),
    "webkb_cornell": ("webkb", "Cornell"),
    "webkb_texas": ("webkb", "Texas"),
    "webkb_wisconsin": ("webkb", "Wisconsin"),
    "chameleon": ("wikipedia", "chameleon"),
    "squirrel": ("wikipedia", "squirrel"),
    "roman_empire": ("heterophilous", "Roman-empire"),
    "amazon_ratings": ("heterophilous", "Amazon-ratings"),
    "minesweeper": ("heterophilous", "Minesweeper"),
    "tolokers": ("heterophilous", "Tolokers"),
    "citationfull_cora_ml": ("citationfull", "Cora_ML"),
    "citationfull_cora": ("citationfull", "Cora"),
    "citationfull_dblp": ("citationfull", "DBLP"),
    "citationfull_citeseer": ("citationfull", "CiteSeer"),
    "citationfull_pubmed": ("citationfull", "PubMed"),
}


@dataclass(frozen=True)
class CitationGraph:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    task_type: str = "multiclass"
    metric_name: str = "accuracy"

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        if self.y.ndim == 2:
            return int(self.y.size(1))
        return int(self.y.max().item() + 1)


def stratified_split_2_1_1(graph: CitationGraph, split_seed: int) -> CitationGraph:
    """Return a deterministic label-stratified 50%/25%/25% node split."""

    if graph.y.ndim != 1:
        raise ValueError("stratified 2:1:1 split requires single-label node targets")
    labels = graph.y.detach().cpu()
    labeled = labels >= 0
    if int(labeled.sum().item()) == 0:
        raise ValueError("stratified 2:1:1 split requires at least one labeled node")
    generator = torch.Generator(device="cpu").manual_seed(int(split_seed))
    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    for label in torch.unique(labels[labeled], sorted=True):
        indices = torch.where(labels == label)[0]
        if indices.numel() < 3:
            raise ValueError(f"class {int(label)} has fewer than three nodes")
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_end = min(max(1, int(round(0.50 * indices.numel()))), indices.numel() - 2)
        val_end = train_end + int(round(0.25 * indices.numel()))
        val_end = min(max(train_end + 1, val_end), indices.numel() - 1)
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
    return CitationGraph(
        x=graph.x,
        y=graph.y,
        edge_index=graph.edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        task_type=graph.task_type,
        metric_name=graph.metric_name,
    )


def apply_split_mode(graph: CitationGraph, split_mode: str, split_seed: int) -> CitationGraph:
    if split_mode == "original":
        return graph
    if split_mode == "stratified_2_1_1":
        return stratified_split_2_1_1(graph, split_seed)
    raise ValueError(f"unknown split mode: {split_mode!r}")


def _download(url: str, path: Path) -> None:
    """Fetch a public dataset file through the host's reliable IPv4 route."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.partial")
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--ipv4",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "3600",
                "--output",
                str(partial_path),
                url,
            ],
            check=True,
        )
        partial_path.replace(path)
    except FileNotFoundError:
        partial_path.unlink(missing_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    except subprocess.CalledProcessError as exc:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download dataset file from {url}") from exc


def _pyg_download_url(url: str, folder: str, log: bool = True, filename: str | None = None) -> str:
    """PyG-compatible downloader that avoids the unavailable IPv6 route."""

    if filename is None:
        filename = url.rpartition("/")[2].split("?", maxsplit=1)[0]
    path = Path(folder) / filename
    if not path.exists():
        if log:
            print(f"Downloading {url}", file=sys.stderr)
        _download(url, path)
    return str(path)


def ensure_planetoid_raw(root: Path, name: str) -> Path:
    raw_dir = root / name.lower() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix in PLANETOID_FILES:
        filename = f"ind.{name.lower()}.{suffix}"
        path = raw_dir / filename
        if not path.exists():
            _download(f"{PLANETOID_BASE_URL}/{filename}", path)
    return raw_dir


def _parse_index_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _sample_mask(indices: list[int] | np.ndarray, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _row_normalize(mx: sp.spmatrix) -> sp.csr_matrix:
    rowsum = np.asarray(mx.sum(1)).flatten()
    inv = np.power(rowsum, -1.0, where=rowsum != 0)
    inv[rowsum == 0] = 0.0
    return sp.diags(inv).dot(mx).tocsr()


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_planetoid(root: str | Path, name: str = "cora") -> CitationGraph:
    """Load Cora/CiteSeer/PubMed using the standard Planetoid split."""

    root = Path(root)
    raw_dir = ensure_planetoid_raw(root, name)
    prefix = raw_dir / f"ind.{name.lower()}"

    x, tx, allx, y, ty, ally, graph = [_load_pickle(Path(f"{prefix}.{s}")) for s in PLANETOID_FILES[:-1]]
    test_idx_reorder = np.array(_parse_index_file(Path(f"{prefix}.test.index")), dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    if name.lower() == "citeseer":
        full_range = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(full_range), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(full_range), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _row_normalize(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1)

    edges = []
    for src, dsts in graph.items():
        for dst in dsts:
            edges.append((src, dst))
            edges.append((dst, src))
    edge_index_np = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
    edge_index = torch.as_tensor(edge_index_np.T, dtype=torch.long)

    n_nodes = labels.shape[0]
    train_mask = _sample_mask(range(len(y)), n_nodes)
    val_mask = _sample_mask(range(len(y), len(y) + 500), n_nodes)
    test_mask = _sample_mask(test_idx_range, n_nodes)

    return CitationGraph(
        x=torch.as_tensor(features.toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _split_masks(num_nodes: int, seed: int, train_ratio: float = 0.10, val_ratio: float = 0.10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)
    train_mask = _sample_mask(perm[:train_end], num_nodes)
    val_mask = _sample_mask(perm[train_end:val_end], num_nodes)
    test_mask = _sample_mask(perm[val_end:], num_nodes)
    return train_mask, val_mask, test_mask


def load_udagcn_domain(root: str | Path, name: str, split_seed: int = 0) -> CitationGraph:
    """Load the DBLPv8/ACMv9 raw format used by UDAGCN.

    The paper-referenced files are stored as ``<root>/<name>/raw`` and use
    comma-separated dense features, comma-separated edge pairs, and one label
    per line.  The source/target split is deterministic and label-free at
    deployment time; labels are retained only for final evaluation.
    """

    name = name.lower()
    if name not in {"dblp", "acm"}:
        raise ValueError("UDAGCN domain must be 'dblp' or 'acm'.")
    root = Path(root)
    if not (root / name).exists() and (root / "udagcn" / name).exists():
        root = root / "udagcn"
    raw_dir = root / name / "raw"
    docs_path = raw_dir / f"{name}_docs.txt"
    edge_path = raw_dir / f"{name}_edgelist.txt"
    label_path = raw_dir / f"{name}_labels.txt"
    missing = [str(path) for path in (docs_path, edge_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing UDAGCN citation-domain files: " + ", ".join(missing)
        )

    features = np.loadtxt(docs_path, delimiter=",", dtype=np.float32)
    features = np.atleast_2d(features)
    edges = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
    edges = np.atleast_2d(edges)
    if edges.shape[1] != 2:
        raise ValueError(f"{edge_path} must contain two integer columns.")
    labels = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name} feature/label count mismatch: {features.shape[0]} vs {labels.shape[0]}"
        )
    if edges.size and (edges.min() < 0 or edges.max() >= features.shape[0]):
        raise ValueError(f"{edge_path} contains a node index outside [0, {features.shape[0]}).")

    train_mask, val_mask, test_mask = _split_masks(
        features.shape[0], split_seed, train_ratio=0.70, val_ratio=0.10
    )
    return CitationGraph(
        x=torch.from_numpy(features),
        y=torch.from_numpy(labels),
        edge_index=torch.from_numpy(edges.T.copy()),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _index_mask(index: torch.Tensor | np.ndarray, size: int) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        index = index.cpu().numpy()
    return _sample_mask(np.asarray(index, dtype=np.int64).reshape(-1), size)


def _dataset_root(root: Path, family: str, name: str) -> Path:
    if root.name == "planetoid":
        root = root.parent
    return root / family / name / "raw"


def ensure_amazon_photo_raw(root: Path) -> Path:
    raw_dir = _dataset_root(root, "amazon", "photo")
    path = raw_dir / "amazon_electronics_photo.npz"
    if not path.exists():
        _download(AMAZON_PHOTO_URL, path)
    return path


def ensure_benchmark_npz_raw(root: Path, name: str) -> Path:
    """Fetch a gnn-benchmark NPZ dataset used by the cross-graph suite."""

    if name not in BENCHMARK_NPZ_URLS:
        raise ValueError(f"Unknown benchmark NPZ dataset: {name!r}.")
    family, dataset = name.split("_", maxsplit=1)
    raw_dir = _dataset_root(root, family, dataset)
    filename = BENCHMARK_NPZ_URLS[name].rsplit("/", maxsplit=1)[-1]
    path = raw_dir / filename
    if not path.exists():
        _download(BENCHMARK_NPZ_URLS[name], path)
    return path


def ensure_actor_raw(root: Path) -> tuple[Path, Path]:
    raw_dir = _dataset_root(root, "actor", "actor")
    feature_path = raw_dir / "out1_node_feature_label.txt"
    edge_path = raw_dir / "out1_graph_edges.txt"
    for path in (feature_path, edge_path):
        if not path.exists():
            _download(f"{ACTOR_BASE_URL}/{path.name}", path)
    return feature_path, edge_path


def _adj_from_npz(loader: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    if {"adj_data", "adj_indices", "adj_indptr", "adj_shape"}.issubset(loader.files):
        return sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )
    if "adj_matrix" in loader.files:
        adj = loader["adj_matrix"].item()
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        return adj.tocsr()
    raise KeyError(f"Unsupported Amazon Photo adjacency keys: {loader.files}")


def _features_from_npz(loader: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    if {"attr_data", "attr_indices", "attr_indptr", "attr_shape"}.issubset(loader.files):
        return sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )
    if "attr_matrix" in loader.files:
        attr = loader["attr_matrix"].item()
        if not sp.issparse(attr):
            attr = sp.csr_matrix(attr)
        return attr.tocsr()
    if "features" in loader.files:
        return sp.csr_matrix(loader["features"])
    raise KeyError(f"Unsupported Amazon Photo feature keys: {loader.files}")


def load_amazon_photo(root: str | Path, split_seed: int = 0) -> CitationGraph:
    """Load Amazon Photo from the public gnn-benchmark NPZ file.

    The dataset has no canonical Planetoid split here, so this loader uses a
    fixed 10%/10%/80% train/validation/test split.
    """

    path = ensure_amazon_photo_raw(Path(root))
    with np.load(path, allow_pickle=True) as loader:
        adj = _adj_from_npz(loader)
        features = _features_from_npz(loader)
        labels = loader["labels"].astype(np.int64)

    features = _row_normalize(features)
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()

    coo = adj.tocoo()
    edge_index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    train_mask, val_mask, test_mask = _split_masks(labels.shape[0], split_seed)

    return CitationGraph(
        x=torch.as_tensor(features.toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_benchmark_npz(root: str | Path, name: str, split_seed: int = 0) -> CitationGraph:
    """Load a transductive graph from gnn-benchmark NPZ data.

    Coauthor CS and Amazon Computers have no canonical split in this project,
    so they follow the existing Amazon Photo 10%/10%/80% seed-controlled split.
    """

    path = ensure_benchmark_npz_raw(Path(root), name)
    with np.load(path, allow_pickle=True) as loader:
        adj = _adj_from_npz(loader)
        features = _features_from_npz(loader)
        labels = loader["labels"].astype(np.int64)

    features = _row_normalize(features)
    adj = adj.tocsr().maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    coo = adj.tocoo()
    edge_index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    train_mask, val_mask, test_mask = _split_masks(labels.shape[0], split_seed)
    return CitationGraph(
        x=torch.as_tensor(features.toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_actor(root: str | Path, split_seed: int = 0) -> CitationGraph:
    """Load Actor (the ``film`` graph in Geom-GCN) without PyG."""

    feature_path, edge_path = ensure_actor_raw(Path(root))
    node_features: list[tuple[int, list[int], int]] = []
    max_node = -1
    max_feature = -1
    with feature_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            node_text, features_text, label_text = line.rstrip("\n").split("\t")
            node_id = int(node_text)
            feature_ids = [int(item) for item in features_text.split(",") if item]
            node_features.append((node_id, feature_ids, int(label_text)))
            max_node = max(max_node, node_id)
            if feature_ids:
                max_feature = max(max_feature, max(feature_ids))

    features = sp.lil_matrix((max_node + 1, max_feature + 1), dtype=np.float32)
    labels = np.zeros(max_node + 1, dtype=np.int64)
    for node_id, feature_ids, label in node_features:
        features[node_id, feature_ids] = 1.0
        labels[node_id] = label

    edges: list[tuple[int, int]] = []
    with edge_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            source, target = (int(item) for item in line.rstrip("\n").split("\t"))
            edges.extend(((source, target), (target, source)))
    edge_index_np = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
    train_mask, val_mask, test_mask = _split_masks(labels.shape[0], split_seed)
    return CitationGraph(
        x=torch.as_tensor(_row_normalize(features.tocsr()).toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=torch.as_tensor(edge_index_np.T, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _pyg_split_masks(data, num_nodes: int, split_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use a provided split column when available, otherwise make a fixed split."""

    masks = [getattr(data, name, None) for name in ("train_mask", "val_mask", "test_mask")]
    if any(mask is None for mask in masks):
        return _split_masks(num_nodes, split_seed)

    selected: list[torch.Tensor] = []
    for mask in masks:
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask[:, split_seed % mask.size(1)]
        if mask.ndim != 1 or mask.numel() != num_nodes:
            raise ValueError("PyG split mask must have shape [num_nodes] or [num_nodes, num_splits].")
        selected.append(mask)
    return tuple(selected)  # type: ignore[return-value]


def load_pyg_dataset(root: str | Path, name: str, split_seed: int = 0) -> CitationGraph:
    """Load an additional single-label benchmark through the optional PyG adapter."""

    if name not in PYG_DATASET_SPECS:
        raise ValueError(f"Unknown PyG dataset: {name!r}.")
    try:
        from torch_geometric.datasets import (
            CitationFull,
            Coauthor,
            HeterophilousGraphDataset,
            WebKB,
            WikiCS,
            WikipediaNetwork,
        )
    except ImportError as exc:
        raise ImportError("Install torch-geometric to load this benchmark dataset.") from exc

    family, dataset_name = PYG_DATASET_SPECS[name]
    # PyG imports ``download_url`` into each dataset module.  Replace that
    # function before dataset construction so all adapter families use the
    # same IPv4 downloader as the native OUGP loaders.
    for dataset_class in (Coauthor, WebKB, WikiCS, WikipediaNetwork, HeterophilousGraphDataset, CitationFull):
        module = sys.modules[dataset_class.__module__]
        setattr(module, "download_url", _pyg_download_url)

    dataset_root = Path(root)
    if dataset_root.name == "planetoid":
        dataset_root = dataset_root.parent
    cache_root = dataset_root / "pyg" / name
    if family == "coauthor":
        dataset = Coauthor(str(cache_root), name=dataset_name)
    elif family == "wikics":
        dataset = WikiCS(str(cache_root))
    elif family == "webkb":
        dataset = WebKB(str(cache_root), name=dataset_name)
    elif family == "wikipedia":
        dataset = WikipediaNetwork(str(cache_root), name=dataset_name, geom_gcn_preprocess=True)
    elif family == "heterophilous":
        dataset = HeterophilousGraphDataset(str(cache_root), name=dataset_name)
    elif family == "citationfull":
        dataset = CitationFull(str(cache_root), name=dataset_name)
    else:
        raise RuntimeError(f"Unhandled PyG dataset family: {family!r}.")

    data = dataset[0]
    if data.x is None or data.y is None or data.edge_index is None:
        raise ValueError(f"PyG dataset {name!r} does not expose x, y, and edge_index.")
    x = data.x.to(dtype=torch.float32)
    y = data.y.reshape(-1).to(dtype=torch.long)
    if y.numel() != x.size(0) or bool((y < 0).any().item()):
        raise ValueError(f"PyG dataset {name!r} is not a fully labeled single-label node task.")
    train_mask, val_mask, test_mask = _pyg_split_masks(data, x.size(0), split_seed)
    return CitationGraph(
        x=x,
        y=y,
        edge_index=data.edge_index.to(dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_hgb_dataset(root: str | Path, name: str, split_seed: int = 0) -> CitationGraph:
    """Load an HGB citation graph as a homogeneous target-node task.

    HGB stores ACM/DBLP as heterogeneous graphs. The target paper node type
    already exposes features and node-classification labels; this adapter uses
    target-to-target citation relations as the homogeneous graph consumed by
    OUGP and splits the provided training labels into train/validation masks.
    """

    try:
        from torch_geometric.datasets import HGBDataset
    except ImportError as exc:
        raise ImportError("Install torch-geometric to load HGB citation datasets.") from exc

    dataset_root = Path(root)
    if dataset_root.name == "planetoid":
        dataset_root = dataset_root.parent
    dataset = HGBDataset(str(dataset_root / "hgb"), name=name.upper())
    data = dataset[0]

    candidates: list[tuple[str, int]] = []
    for node_type in data.node_types:
        store = data[node_type]
        labels = getattr(store, "y", None)
        train_mask = getattr(store, "train_mask", None)
        test_mask = getattr(store, "test_mask", None)
        if labels is not None and train_mask is not None and test_mask is not None:
            labels = torch.as_tensor(labels).reshape(-1)
            candidates.append((node_type, int((labels >= 0).sum().item())))
    if not candidates:
        raise ValueError(f"HGB dataset {name!r} has no labeled target node type.")
    target_type = max(candidates, key=lambda item: item[1])[0]
    target_store = data[target_type]
    y = torch.as_tensor(target_store.y, dtype=torch.long).reshape(-1)
    num_nodes = int(y.numel())

    target_edges: list[torch.Tensor] = []
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if src_type == target_type and dst_type == target_type:
            target_edges.append(data[edge_type].edge_index.to(dtype=torch.long))
    if not target_edges:
        raise ValueError(f"HGB dataset {name!r} has no {target_type}->{target_type} citation relation.")
    edge_index = torch.cat(target_edges, dim=1)

    features = getattr(target_store, "x", None)
    if features is None:
        degree = torch.bincount(edge_index[0], minlength=num_nodes).float().unsqueeze(1)
        features = degree / degree.mean().clamp_min(1.0)
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim != 2 or x.size(0) != num_nodes:
        raise ValueError(f"HGB dataset {name!r} target features must have shape [N, F].")

    provided_train = torch.as_tensor(target_store.train_mask, dtype=torch.bool).reshape(-1)
    provided_test = torch.as_tensor(target_store.test_mask, dtype=torch.bool).reshape(-1)
    labeled = y >= 0
    provided_train &= labeled
    provided_test &= labeled
    if int(provided_train.sum().item()) < 2 or int(provided_test.sum().item()) == 0:
        train_mask, val_mask, test_mask = _split_masks(num_nodes, split_seed)
        train_mask &= labeled
        val_mask &= labeled
        test_mask &= labeled
    else:
        train_indices = torch.where(provided_train)[0]
        generator = torch.Generator(device="cpu").manual_seed(int(split_seed))
        permutation = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        val_count = max(1, int(round(0.2 * permutation.numel())))
        val_count = min(val_count, permutation.numel() - 1)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[permutation[:-val_count]] = True
        val_mask[permutation[-val_count:]] = True
        test_mask = provided_test

    return CitationGraph(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_graph_dataset(root: str | Path, name: str, split_seed: int = 0) -> CitationGraph:
    lowered = name.lower()
    if lowered in {"udagcn_dblp", "udagcn-dblp"}:
        return load_udagcn_domain(root, "dblp", split_seed)
    if lowered in {"udagcn_acm", "udagcn-acm"}:
        return load_udagcn_domain(root, "acm", split_seed)
    if lowered in {"cora", "citeseer", "pubmed"}:
        return load_planetoid(root, lowered)
    if lowered in {"photo", "amazon_photo", "amazon-photo"}:
        return load_amazon_photo(root, split_seed)
    if lowered in {"computers", "amazon_computers", "amazon-computers"}:
        return load_benchmark_npz(root, "amazon_computers", split_seed)
    if lowered in {"coauthor_cs", "coauthor-cs", "coauthorcs"}:
        return load_benchmark_npz(root, "coauthor_cs", split_seed)
    if lowered in {"actor", "film"}:
        return load_actor(root, split_seed)
    if lowered in {"acm", "hgb_acm", "hgb-acm"}:
        return load_hgb_dataset(root, "acm", split_seed)
    if lowered in PYG_DATASET_SPECS:
        return load_pyg_dataset(root, lowered, split_seed)
    if lowered in {"ogbn-arxiv", "arxiv"}:
        return load_ogbn_node_property(root, "ogbn-arxiv")
    if lowered in {"ogbn-products", "products"}:
        return load_ogbn_node_property(root, "ogbn-products")
    if lowered in {"ogbn-proteins", "proteins"}:
        return load_ogbn_node_property(root, "ogbn-proteins")
    raise ValueError(f"Unknown dataset {name!r}.")


def _aggregate_edge_features(num_nodes: int, edge_index: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
    features = np.zeros((num_nodes, edge_feat.shape[1]), dtype=np.float32)
    degree = np.zeros((num_nodes, 1), dtype=np.float32)
    src, dst = edge_index
    np.add.at(features, dst, edge_feat.astype(np.float32))
    np.add.at(degree, dst, 1.0)
    return features / np.maximum(degree, 1.0)


def load_ogbn_node_property(root: str | Path, name: str) -> CitationGraph:
    """Load OGB node property prediction datasets through ogb.nodeproppred."""

    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError("Install `ogb` in the active environment to load OGB datasets.") from exc

    root = Path(root)
    if root.name == "planetoid":
        root = root.parent / "ogb"
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat
    try:
        dataset = NodePropPredDataset(name=name, root=str(root))
    finally:
        torch.load = original_torch_load
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    num_nodes = int(graph["num_nodes"])
    edge_index_np = np.asarray(graph["edge_index"], dtype=np.int64)

    node_feat = graph.get("node_feat")
    task_type = "multiclass"
    metric_name = "accuracy"
    if node_feat is None:
        edge_feat = graph.get("edge_feat")
        if edge_feat is None:
            raise ValueError(f"{name} has neither node_feat nor edge_feat.")
        node_feat = _aggregate_edge_features(num_nodes, edge_index_np, np.asarray(edge_feat))

    y_np = np.asarray(labels)
    if name == "ogbn-proteins":
        task_type = "multilabel"
        metric_name = "rocauc"
        y = torch.as_tensor(y_np, dtype=torch.float32)
    else:
        y = torch.as_tensor(y_np.reshape(-1), dtype=torch.long)

    x = torch.as_tensor(np.asarray(node_feat), dtype=torch.float32)
    train_mask = _index_mask(split_idx["train"], num_nodes)
    val_mask = _index_mask(split_idx["valid"], num_nodes)
    test_mask = _index_mask(split_idx["test"], num_nodes)

    return CitationGraph(
        x=x,
        y=y,
        edge_index=torch.as_tensor(edge_index_np, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        task_type=task_type,
        metric_name=metric_name,
    )
