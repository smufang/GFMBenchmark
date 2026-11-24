# GFMBenchmark

A comprehensive benchmark for Graph Foundation Models (GFMs), covering pre-training, fine-tuning, and evaluation across diverse graph domains and tasks.

[[arXiv]](https://arxiv.org/abs/2603.10033) | [[HF Datasets]](https://huggingface.co/datasets/smufang/GFM-datasets) | [[HF Checkpoints]](https://huggingface.co/smufang/GFM-checkpoints) | [[Website]](https://xiexuanting.github.io/GFMBenchmark_Home/)

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

### Hugging Face

| Repository | Contents | Size |
|------------|----------|------|
| [**GFM-datasets**](https://huggingface.co/datasets/smufang/GFM-datasets) | `datasets/`: raw files, PyG-processed files, and model-specific caches (such as GCOPE `induced_graphs.pt` and GraphCLIP subgraph summaries) for all datasets except *Products*, *DGraph* and *Elliptic*; `datasets_split/`: pre-generated few-shot splits for every dataset | ~75 GB |
| [**GFM-checkpoints**](https://huggingface.co/smufang/GFM-checkpoints) | Pre-trained weights of the evaluated models in each setting, laid out in the folders the code reads from | ~6.1 GB |

### Option A: Download from Hugging Face (Recommended)

Requires `pip install -U "huggingface_hub>=0.34"`. Files are written into the project root (`datasets/`, `datasets_split/`, `checkpoints/`, `GFT/ckpts/`, ...), and re-running a command resumes an interrupted download.

```bash
bash scripts/hf_download.sh datasets                   # all hosted datasets + few-shot splits (~75 GB)
bash scripts/hf_download.sh datasets Cora ACM Reddit   # selected datasets + all few-shot splits
bash scripts/hf_download.sh splits                     # few-shot splits only (~0.3 GB)
bash scripts/hf_download.sh checkpoints                # pre-trained checkpoints (~6.1 GB)
```

> Generating GCOPE `induced_graph` and GraphCLIP summaries locally is time/memory-consuming. We strongly suggest downloading them from Hugging Face instead.

**Products, DGraph and Elliptic** are not hosted on Hugging Face, because their original licenses restrict redistribution; their few-shot splits are. To use them, get the raw files from the original providers:

| Dataset | How to get it |
|---------|---------------|
| Products | Download `processed_data.pt` from the [LLaGA release](https://utexas.app.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb/file/1441818321356) and put it in `datasets/Products/raw/` |
| DGraph | Register at [dgraph.xinye.com](https://dgraph.xinye.com), download `DGraphFin.zip` and put it in `datasets/DGraph/raw/` |
| Elliptic | Downloaded automatically by PyG the first time it is loaded (e.g., `bash scripts/data_download.sh`) |

The loaders process these raw files on first use, and the model-specific caches are rebuilt automatically when missing. GraphCLIP pre-training on Elliptic also needs subgraph summaries, which are generated with `GraphCLIP/generate_summary_api.py` (OpenAI API).

Zip archives are also available from our earlier download hub: [https://smu.sg/GFMBenchmark](https://smu.sg/GFMBenchmark).

### Option B: Download from Original Sources

```bash
bash scripts/data_download.sh
```

A few datasets (*Products*, *DGraph* and *T-Finance*) cannot be fetched automatically; the loader prints the download link and the folder to put the file in.

### Option C: Generate Few-Shot Splits

If not using the pre-generated splits:

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

To evaluate our released checkpoints without pre-training, run `bash scripts/hf_download.sh checkpoints` and then the `*_downstream.sh` scripts.

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
