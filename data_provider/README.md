# Data Provider Guide

## File Overview

| File | Purpose |
|------|---------|
| `__init__.py` | Defines experiment dataset groups (pretrain/downstream per exp) |
| `data_generator.py` | Dataset load functions with label names and text transforms |
| `data_loader.py` | Main loader pipeline: sampling, compression, batching |
| `pygform_creator.py` | Converts raw datasets into PyG format |
| `fewshot_generator.py` | Generates/loads few-shot splits from `datasets_split/` |
| `_downloading.py` | Downloads all datasets |
| `maxnode_loader.py` | Batches graphs by max total node count (used in all experiments) |
| `maxedge_loader.py` | Batches graphs by max total edge count |
| `maxsize_loader.py` | Batches graphs by max total size (nodes + edges) |
| `multigraph_loader.py` | Loader for multi-graph datasets |
| `subbatch_loader.py` | Sub-batch loader for single large graphs |

---

## `__init__.py` — Experiment Dataset Configuration

This file defines which datasets are used in each experiment group:

- **Pretrain groups:** `pretrain` (exp1/2), `pretrain_exp3_cite`, `pretrain_exp3_social`, `pretrain_exp3_molecule`, `pretrain_exp4`, `pretrain_exp0` (testing)
- **Downstream groups:** `NC_exp{0-4}`, `EC_exp{0-4}`, `GC_exp{0-4}` (node/edge/graph classification)

### No Label Name Datasets

Some datasets have no `label_names` / `label_descs` (marked with `### No label name ###` in the code). **G2P2 and GraphCLIP cannot test on these datasets** because they require label text descriptions. When running G2P2/GraphCLIP, comment out these datasets in the corresponding exp group.

Affected datasets: Chameleon, ogbn-proteins, ogbn-mag, Products, DGraph (edge), WIKI, PCBA, Photo.

---

## `data_generator.py` — Dataset Load Interfaces

Each function (e.g., `cora()`, `elliptic()`) returns a PyG dataset with optional text transforms that attach `label_names`, `label_descs`, `raw_texts`, `relation_texts`, and `edge_type`.

### Multi-Label Datasets

Some datasets support multiple classification granularities via different label dimensions:

| Dataset | Label Dimensions | Meaning |
|---------|-----------------|---------|
| DGraph | 2-class, 3-class, 4-class | Normal/fraudster (2), +background (3), +subtype (4) |
| Elliptic | 2-class, 3-class | Licit/illicit (2), +others (3) |

In `data_generator.py`, `-1` in labels means "no need to test" (background/unlabeled nodes are masked with `y = -1`). The label splits are pre-generated in `datasets_split/` — **do not modify the mask logic in `data_generator.py` unless regenerating splits**.

---

## `datasets_split/` — Pre-Generated Splits

Splits are stored per dataset. The program only recognizes `node-0` / `split_node_0.pt` as the active split.

### Elliptic (2-class / 3-class)

```
datasets_split/Elliptic/
  few-shot/
    node-0/            # currently active
    node2way-0/        # 2-class: licit / illicit
    node3way-0/        # 3-class: licit / illicit / others
  split/
    split_node_0.pt
    split_node2way_0.pt
    split_node3way_0.pt
```

### DGraph (2-class / 3-class / 4-class)

```
datasets_split/DGraph/
  few-shot/
    node-0/            # currently active
    node2way-0/        # 2-class
    node3way-0/        # 3-class
    node4way-0/        # 4-class
  split/
    split_node_0.pt
    split_node2way_0.pt
    split_node3way_0.pt
    split_node4way_0.pt
```

### Switching Classification Variant

Delete the current `node-0/` and `split_node_0.pt`, then copy the desired variant and rename it:

```bash
# e.g., switch Elliptic to 3-class
cd datasets_split/Elliptic/
rm -r few-shot/node-0/
cp -r few-shot/node3way-0/ few-shot/node-0/
rm split/split_node_0.pt
cp split/split_node3way_0.pt split/split_node_0.pt
```

---

## `data_loader.py` — Main Loader Pipeline

### Pretrain Pipeline

`pretrain_loader()` → `pretrain_sampler()` → `general_loader()`

1. `pretrain_sampler()`: Iterates over dataset generators, converts to PyG Data, samples subgraphs via `data_sampler()` (for single graphs) or `multidata_sampler()` (for multi-graph datasets)
2. `general_loader()`: Groups sampled subgraphs into batches using `MultiGraphLoader`, applies compression, aligns edge attributes, and yields `Batch` objects

### Loader Types

The loader used in `general_loader()` is imported at the top of `data_loader.py`:

```python
from data_provider.maxnode_loader import MultiGraphLoader  # current
```

| Loader | Batching Strategy | Status |
|--------|------------------|--------|
| `maxnode_loader` | Batch until total nodes reach `max_nodes` | **Used in all experiments** |
| `maxedge_loader` | Batch until total edges reach limit | Alternative, better control for dense graphs |
| `maxsize_loader` | Batch until total nodes + edges reach limit | Alternative, balanced control |

To switch loader, change the import in `data_loader.py` line 6.

### Key Functions

- `create_x()`: Creates node features (one-hot) if dataset has no `x`
- `complete_data()`: Standardizes data attributes (`x`, `edge_index`, `edge_type`, `raw_texts`, etc.)
- `data_sampler()`: NeighborLoader-based subgraph sampling for single large graphs
- `multidata_sampler()`: BatchGraphLoader for multi-graph datasets (e.g., molecule datasets)

---

## Other Files

- **`pygform_creator.py`**: Custom dataset classes that convert raw data into PyG `Data`/`InMemoryDataset` format (Cora, FB15K-237, TFinance, WN18RR, etc.)
- **`multigraph_loader.py`**: Loader for multi-graph datasets, groups small graphs into batches
- **`subbatch_loader.py`**: Splits a single large graph's subgraphs into batches by graph-level indices
