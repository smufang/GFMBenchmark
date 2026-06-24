# Scripts Guide

## Overview

All scripts should be run from the **project root directory** (not from `scripts/`):

```bash
bash scripts/<script_name>.sh
```

Logs are saved to `logs/<model>/`, PIDs to `pids/`.

---

## 1. Data Preparation

| Script | Description |
|--------|-------------|
| `data_download.sh` | Download all datasets (~1.5 hours) |
| `fewshot_generate.sh` | Generate few-shot splits for downstream evaluation |

Run these **before** any training.

---

## 2. Cross-Domain Models (via `run.py`)

These models use `run.py` as the unified entry point with `--pattern cross`.

### Models: MDGPT, SAMGPT, MDGFM

**Pretrain:**

```bash
# Edit the script to set MODEL_ID and backbone, then run:
bash scripts/mdgpt_pretrain.sh
bash scripts/samgpt_pretrain.sh
bash scripts/mdgfm_pretrain.sh
```

**Downstream:**

```bash
bash scripts/mdgpt_downstream.sh
bash scripts/samgpt_downstream.sh
bash scripts/mdgfm_downstream.sh
```

#### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--model` | Model name | `mdgpt`, `samgpt`, `mdgfm` |
| `--model_id` | Pretrain experiment setting | `exp1`, `exp3cite`, `exp3social`, `exp3molecule`, `exp4`, `exp0` (testing only), `none` |
| `--exp_id` | Downstream experiment setting | `exp1`, `exp2`, `exp3`, `exp4`, `exp0` (testing only) |
| `--backbone` | Graph encoder | `gcn`, `gat`, `fagcn` |
| `--mode` | Pretrain task | `lp` (link prediction, for MDGFM), `gcl` (graph contrastive learning, for MDGPT/SAMGPT) |
| `--task_name` | Downstream task | `node`, `edge`, `graph` |

> For experiment-to-parameter mapping, see [Section 6: Experiment Compatibility Matrix](#6-experiment-compatibility-matrix).

---

### Models: DDGCL, TGN, HeCo

These are **single-domain** models (`--pattern single`). Their `--model_id` is a dataset name (e.g., `exp2`, `ACM`).

```bash
bash scripts/ddgcl_pretrain.sh   # backbone: tgat
bash scripts/tgn_pretrain.sh     # backbone: tgn
bash scripts/heco_pretrain.sh    # backbone: gcn
```

---

### Baseline Models: GCN, GAT, SimpleHGN

No pretraining needed. Use `--pattern none` for downstream only:

```bash
bash scripts/gcn_downstream.sh
bash scripts/gat_downstream.sh
bash scripts/simplehgn_downstream.sh
```

---

## 3. Self-Contained Models (own entry point)

These models have their own pretrain/downstream scripts and do **not** use `run.py`.

### GFT

Uses `GFT/GFT/pretrain.py` and `GFT/GFT/finetune.py` with `--exp` (integer).

```bash
# Pretrain: edit EXP=1 in the script
bash scripts/gft_pretrain.sh

# Downstream: edit EXP, TASK_NAME, datasets
bash scripts/gft_downstream.sh
```

| `--exp` | Pretrain Data | Backbone (default) |
|---------|--------------|-------------------|
| 1 | exp1 (all 20 datasets) | sage |
| 2 | exp1 (same as exp1) | sage |
| 3 | exp3 (citation subset) | sage |
| 4 | exp4 (6 datasets) | sage |

**Exp5 = FAGCN variant:** GFT's `--backbone` arg defaults to `sage`. To run with FAGCN, add `--backbone fagcn` in the script. GFT does not have a built-in exp5 mapping; "exp5" means rerunning exp4 with `--backbone fagcn`.

### UniGraph2

Uses `UniGraph2/pretrain.py` and `UniGraph2/downstream.py` with `--exp` (integer).

```bash
# Pretrain
bash scripts/unigraph2_pretrain.sh

# Downstream
bash scripts/unigraph2_downstream.sh
```

| `--exp` | Pretrain Data | GNN Backbone |
|---------|--------------|-------------|
| 1 | exp1 (all 20 datasets) | GATConv |
| 2 | exp1 (same as exp1) | GATConv |
| 3 | exp3 (citation subset) | GATConv |
| 4 | exp4 (6 datasets) | GATConv |
| 5 | exp4 (same as exp4) | **FAGCN** (FAConv) |

**Exp5 = FAGCN variant:** Set `--exp 5` to use `unigraph2_fagcn.py` (FAConv) instead of the default GATConv. Uses the same pretrain data as exp4.

### GraphPrompt / DGI

Both share the same pretrain script. Pretrain is per-dataset (single-domain).

```bash
# Pretrain (set DATASET in script)
bash scripts/graphprompt_pretrain.sh   # or dgi_pretrain.sh (same file)

# Downstream
bash scripts/graphprompt_downstream.sh
bash scripts/dgi_downstream.sh         # DGI uses --usemlp yes
```

### G2P2 / GraphCLIP

Use their own `run_g2p2_train.py` / `run_graphclip_train.py` entry points.

```bash
bash scripts/g2p2_pretrain.sh
bash scripts/g2p2_downstream.sh
bash scripts/graphclip_pretrain.sh
bash scripts/graphclip_downstream.sh
```

These use `--pattern simple` and set `--model_id` / `--backbone` in the script.

### GCOPE

Uses `GCOPE/exec.py` with its own config system (fastargs, not argparse).

```bash
# Pretrain
bash scripts/gcope_pretrain.sh

# Downstream
bash scripts/gcope_downstream.sh
```

Key args: `--general.model_id`, `--model.backbone.model_type` (`gcn`/`fagcn`/`gat`), `--data.name` (`node`/`edge`/`graph` for downstream, `pretrain` for pretrain).

### FAGCN / DSSL

Standalone execution scripts:

```bash
bash scripts/fagcn_execute.sh
bash scripts/dssl_execute.sh
```

---

## 4. Utility Scripts

| Script | Description |
|--------|-------------|
| `jsd.sh` | Run Jensen-Shannon Divergence analysis (`utils.jsd`) |
| `tfidf.sh` | Run TF-IDF feature extraction (`utils.tfidf`) |

---

## 5. Avoiding OOM (Out of Memory)

When encountering CUDA OOM, adjust the following parameters **per dataset**:

### General Rule

| Parameter | Default | Reduce to avoid OOM |
|-----------|---------|-------------------|
| `--batch_size` | 32768 (downstream) / 1024 (pretrain) | Halve iteratively until it fits |
| `--max_nodes` | 60000-120000 | Reduce for pretrain |
| `--num_neighbors` | `10 10 10 10` | Use `5 5 5 5` or `5 5 5` |
| `--num_workers` | 4 | Set to 0-1 to reduce memory |

### Dataset-Specific OOM Settings

**Large-scale datasets** (these always need reduced batch size):

| Dataset | Context | Recommended `--batch_size` |
|---------|---------|---------------------------|
| ogbn-mag | downstream (all models) | 2048 or lower |
| ogbn-mag | SimpleHGN | 128, with `--num_neighbors 5 5 5` |
| Products | downstream (most models) | 4096-8192 |
| Products | GCN/GAT downstream | 512 (GCN) / 1024 (GAT) |
| Products | GAT 5-shot | 128 |
| ogbn-proteins | GCN/GAT | Must use `--compress_function pca --input_dim 50` |

**Model-specific OOM settings:**

| Model | Context | Setting |
|-------|---------|---------|
| SAMGPT | Products 5-shot | `--batch_size 4096` |
| MDGFM | graph tasks | `--batch_size 8192` |
| MDGFM | Products downstream | `--num_neighbors 5 5 5 5 --batch_size 4096` |
| MDGFM | ogbn-mag downstream | `--batch_size 2048` |
| GraphCLIP | default downstream | `--batch_size 16384` |
| GraphCLIP | HIV | `--batch_size 4096` |
| GraphCLIP | PROTEINS | `--batch_size 512` (GPSConv crashes on large graph-size variance in batch) |
| MDGPT/MDGFM/SAMGPT | exp4-fagcn, ogbn-mag | `--num_neighbors 5 5 5 --batch_size 2048` |
| MDGPT/MDGFM/SAMGPT | exp4-fagcn, Products | `--batch_size 2048` |

### Class Prototype OOM (SAMGPT, MDGPT, MDGFM)

These models use **class prototypes** for downstream fine-tuning, which requires loading all shots of all classes into the model at once to compute prototypes — this step **cannot be split into batches**. When the number of classes is large (e.g., ogbn-mag), this easily causes OOM. The only effective mitigation is reducing `num_neighbors` to shrink the per-node subgraph size. However, the downstream `--num_neighbors` CLI arg is **not** passed to this step — `num_neighbors` is hardcoded inside `exp/exp_downstream_batch.py` → `_get_loader()`:

- **Node tasks (line ~176):** `num_neighbors = [10, 10, 10, 10]` (default) or `[5, 5, 5, 5]` (hetero)
  - Already has special cases for `simple_hgn`/`fagcn` on ogbn-mag → `[5, 5, 5]`, and `mdgfm` on Products → `[5, 5, 5, 5]`
- **Edge tasks (line ~200):** `num_neighbors = [5, 5, 5, 5]`

To fix OOM, manually change the `num_neighbors` list in the corresponding branch (e.g., `[10, 10, 10, 10]` → `[5, 5, 5]` or `[5, 5]`).

### Tips

- If OOM occurs during **pretrain**, reduce `--max_nodes` and `--batch_size` first.
- If OOM occurs during **downstream**, reduce `--batch_size` first, then `--num_neighbors`.
- GraphCLIP uses GPSConv with multi-head attention internally. A batch with extreme variation in graph sizes (e.g., node counts from 4 to 620) can cause CUDA illegal memory access. Reduce `--batch_size` to limit per-batch variance.
- Setting `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (uncomment in scripts) can help with memory fragmentation.

---

## 6. Experiment Compatibility Matrix

### Cross-Domain Models (MDGPT, SAMGPT, MDGFM)

| Experiment | `--model_id` | `--exp_id` | `--backbone` | Status |
|------------|-------------|-----------|-------------|--------|
| exp1 | `exp1` | `exp1` | `gcn` | OK |
| exp2 | `exp1` | `exp2` | `gcn` | OK (exp2 reuses exp1 pretrain) |
| exp3-citation | `exp3cite` | `exp3` | `gcn` | OK |
| exp3-social | `exp3social` | `exp3` | `gcn` | OK |
| exp3-molecule | `exp3molecule` | `exp3` | `gcn` | OK |
| exp4 | `exp4` | `exp4` | `gcn` | OK |
| exp4-fagcn | `exp4` | `exp4` | `fagcn` | OK |

### GFT

| Experiment | `--exp` | `--backbone` | Status |
|------------|---------|-------------|--------|
| exp1 | `1` | `sage` (default) | OK |
| exp2 | `2` | `sage` | OK |
| exp3 | `3` | `sage` | OK (uses citation subset) |
| exp4 | `4` | `sage` | OK |
| exp5 (fagcn) | `4` | `fagcn` (add `--backbone fagcn`) | OK — reuses exp4 data, changes encoder |

> Note: GFT uses integer `--exp` (1-4), **not** string model_id. There is no `exp3cite/social/molecule` split — GFT's exp3 only has the citation subset. Checkpoint path is auto-generated from exp number.

### UniGraph2

| Experiment | `--exp` | GNN Backbone | Status |
|------------|---------|-------------|--------|
| exp1 | `1` | GATConv | OK |
| exp2 | `2` | GATConv | OK |
| exp3 | `3` | GATConv | OK (citation subset) |
| exp4 | `4` | GATConv | OK |
| exp5 (fagcn) | `5` | FAConv (FAGCN) | OK — uses exp4 data with FAGCN backbone |

### GCOPE

| Experiment | `--general.model_id` | `--model.backbone.model_type` | Status |
|------------|---------------------|------------------------------|--------|
| exp1 | `exp1` | `fagcn` | OK |
| exp3cite | `exp3cite` | `fagcn` | OK |
| exp3social | `exp3social` | `fagcn` | OK |
| exp3molecule | `exp3molecule` | `fagcn` | OK |
| exp4 | `exp4` | `fagcn` | OK |
| exp4 (gcn) | `exp4` | `gcn` | OK — for comparison with other models |

> GCOPE exp1-3 default to `fagcn` backbone. Exp4 adds a `gcn` experiment for cross-model comparison. Use `--data.name pretrain` for pretraining and `--data.name node/edge/graph` for downstream. The `--general.model_id` controls which pretrain checkpoint path to use; adjust `--adapt.pretrained_file` accordingly.

### GraphPrompt / DGI

Single-domain only (per-dataset pretrain). No exp1-4 concept — pretrain and evaluate on the same dataset.

### G2P2 / GraphCLIP

Use `--pattern simple` with `--model_id` matching the pretrain setting. Support `--backbone fagcn` for exp4 variant.
