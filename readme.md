# GFMBenchmark

## 📖 Introduction
**GFMBenchmark** is a comprehensive codebase for Graph Foundation Models (GFMs). It simplifies pre-training, evaluation, and benchmarking across diverse graph domains.

## 🛠️ Installation

### 1. Environment Setup
Create a generic Python 3.10 environment using Conda:

```bash
conda create -n gfm python=3.10
conda activate gfm
```

### 2. Install Dependencies
Install the required packages. Ensure PyTorch is compatible with your CUDA version.

```bash
pip install -r requirements.txt
```

---

## 💾 Data Preparation

You can prepare the data using **Cloud Downloads** (Recommended) or **Scripts**.

### 🔗 Cloud Resources (Download Links)
**👉 Download Hub: [https://smu.sg/GFMBenchmark](https://smu.sg/GFMBenchmark)**
| Resource | Description | Recommendation |
| :--- | :--- | :--- |
| **Lite Datasets** | Essential subset for quick testing | ✅ For Quick Start |
| **Pre-processed Files** | Includes **GCOPE** `induced_graph` & **GraphCLIP** `subgraph summary` | ⭐ **Highly Recommended** |
| **Full Bundle** | All datasets + Pre-processed files | For Reproduction |
| **Checkpoints** | Pre-trained model weights | For Evaluation |

> **Note**: Generating GCOPE `induced_graph` and GraphCLIP summaries locally is time-consuming. We strongly suggest downloading the **Pre-processed Files** directly.

### Option A: Automatic Script (Raw Data Only)
To download the standard raw datasets automatically:
```bash
bash scripts/data_download.sh
```

### Option B: Data Splits Generation
After downloading datasets (via Cloud or Script), generate the few-shot splits required for downstream tasks:
```bash
bash scripts/fewshot_generate.sh
```

---

## 🚀 Training & Evaluation

Once the environment is set up and data is prepared, you can run experiments using the scripts provided in the `scripts/` folder.

### Run Experiments
```bash
# Example: Pre-training
bash scripts/samgpt_pretrain.sh

# Example: Downstream Task Evaluation
bash scripts/samgpt_downstream.sh
```

Modify the scripts to adjust parameters such as `TASK_NAMES`, `model_id`, or `exp_id` as needed.

### ⚡ Batch Size Configuration
For optimal performance and to avoid OOM errors, refer to `scripts/_parameters`. 
