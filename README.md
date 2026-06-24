# GFMBenchmark

A comprehensive benchmark for Graph Foundation Models (GFMs), covering pre-training, fine-tuning, and evaluation across diverse graph domains and tasks.

[[Paper]](https://arxiv.org/abs/2603.10033) | [[Hugging Face]](TBD) | [[Website]](TBD)

---

## Models

| Model | Type | Pattern | Entry Point |
|-------|------|---------|-------------|
| MDGPT | Cross-domain | `cross` | `run.py` |
| SAMGPT | Cross-domain | `cross` | `run.py` |
| MDGFM | Cross-domain | `cross` | `run.py` |
| GFT | Cross-domain | — | `GFT/GFT/pretrain.py`, `GFT/GFT/finetune.py` |
| UniGraph2 | Cross-domain | — | `UniGraph2/pretrain.py`, `UniGraph2/downstream.py` |
| GCOPE | Cross-domain | — | `GCOPE/exec.py` |
| G2P2 | Cross-domain | `simple` | `run_g2p2_train.py` |
| GraphCLIP | Cross-domain | `simple` | `run_graphclip_train.py` |
| GraphPrompt | Single-domain | `single` | `GraphPrompt/pretrain.py` |
| DGI | Single-domain | `single` | `GraphPrompt/pretrain.py` |
| DDGCL | Single-domain | `single` | `run.py` |
| TGN | Single-domain | `single` | `run.py` |
| HeCo | Single-domain | `single` | `run.py` |
| GCN | Baseline | `none` | `run.py` |
| GAT | Baseline | `none` | `run.py` |
| SimpleHGN | Baseline | `none` | `run.py` |
| FAGCN | Baseline | — | `scripts/fagcn_execute.sh` |
| DSSL | Baseline | — | `scripts/dssl_execute.sh` |

---

## Installation

```bash
conda create -n gfm python=3.10
conda activate gfm
pip install -r requirements.txt
```

Ensure PyTorch is compatible with your CUDA version.

---

## Data Preparation

### Cloud Resources

**Download Hub: [https://smu.sg/GFMBenchmark](https://smu.sg/GFMBenchmark)**

| Source | Description | Recommendation |
|--------|-------------|----------------|
| **Full Datasets** | All raw datasets | For Reproduction |
| **Lite Datasets** | Essential subset for quick testing | For Quick Start |
| **GFM-checkpoints** | Pre-trained model weights | For Evaluation |
| **datasets_split.zip** | Pre-generated few-shot splits + GCOPE `induced_graph` & GraphCLIP `subgraph summary` | Highly Recommended |

> Generating GCOPE `induced_graph` and GraphCLIP summaries locally is time/memory-consuming. We strongly suggest downloading the **Pre-processed Files** directly.

### Option A: Automatic Download

```bash
bash scripts/data_download.sh
```

### Option B: Generate Few-Shot Splits

If not using pre-processed splits from the cloud:

```bash
bash scripts/fewshot_generate.sh
```

**DGraph & Elliptic** support multiple classification granularities. To switch variant, delete the current `node-0` / `split_node_0.pt` and copy the desired variant (e.g., `node2way-0` → `node-0`). See `data_provider/README.md` for details.

---

## Training & Evaluation

All scripts are run from the **project root directory**:

```bash
# Pre-training
bash scripts/samgpt_pretrain.sh

# Downstream evaluation
bash scripts/samgpt_downstream.sh
```

Modify the scripts to adjust parameters such as `--model_id`, `--exp_id`, `--backbone`, `--task_name`, etc. See [`scripts/README.md`](scripts/README.md) for per-model usage, experiment settings, OOM tuning, and compatibility matrix.

### Dataset Configuration

Adjust datasets in `data_provider/__init__.py` per experiment group. Datasets marked with `### No label name ###` are **incompatible** with G2P2 and GraphCLIP — comment them out when running these models. See [`data_provider/README.md`](data_provider/README.md) for details on data loading, dataset splits, and loader types.

---

## Citation

```bibtex
@article{yu2026evaluating,
  title={Evaluating progress in graph foundation models: A comprehensive benchmark and new insights},
  author={Yu, Xingtong and Ye, Shenghua and Liang, Ruijuan and Zhou, Chang and Cheng, Hong and Zhang, Xinming and Fang, Yuan},
  journal={arXiv preprint arXiv:2603.10033},
  year={2026}
}
```
