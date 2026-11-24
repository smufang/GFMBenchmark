#!/bin/bash
# Download GFMBenchmark datasets, few-shot splits and pre-trained checkpoints from the Hugging Face Hub.
# Files are written into the project root, at the paths the code reads from:
#   datasets/  datasets_split/  checkpoints/  FAGCN/modelset/  GCOPE/storage/
#   GFT/ckpts/  GraphPrompt/checkpoints/  UniGraph2/checkpoints/
#
# Usage (from the project root):
#   bash scripts/hf_download.sh all                         # datasets + splits + checkpoints (~82 GB)
#   bash scripts/hf_download.sh datasets                    # all hosted datasets + few-shot splits (~75 GB)
#   bash scripts/hf_download.sh datasets Cora ACM Reddit    # selected datasets + all few-shot splits
#   bash scripts/hf_download.sh splits                      # pre-generated few-shot splits only (~0.3 GB)
#   bash scripts/hf_download.sh checkpoints                 # pre-trained checkpoints only (~6.1 GB)
#
# Requires the `hf` command: pip install -U "huggingface_hub>=0.34"
# An interrupted download resumes when you re-run the same command.
set -euo pipefail

HF_NAMESPACE="${HF_NAMESPACE:-smufang}"
DATA_REPO="${HF_NAMESPACE}/GFM-datasets"
CKPT_REPO="${HF_NAMESPACE}/GFM-checkpoints"

# Not hosted on the Hub because their original terms restrict redistribution; only their
# few-shot splits are. Keep this list in sync with WITHHELD_DATASETS in scripts/hf_upload.sh.
WITHHELD_DATASETS=(Products DGraph Elliptic)

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."  # project root

if ! command -v hf >/dev/null 2>&1; then
    echo "'hf' command not found. Install it with: pip install -U \"huggingface_hub>=0.34\"" >&2
    exit 1
fi

is_withheld() {
    local name
    for name in "${WITHHELD_DATASETS[@]}"; do [[ "$1" == "$name" ]] && return 0; done
    return 1
}

print_withheld_help() {
    cat >&2 <<'EOF'

Note: Products, DGraph and Elliptic are not hosted on Hugging Face (their licenses restrict
redistribution); their few-shot splits are. To use them, get the raw files from the source:
  Products  download processed_data.pt from the LLaGA release
            https://utexas.app.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb/file/1441818321356
            and put it in datasets/Products/raw/
  DGraph    register at https://dgraph.xinye.com, download DGraphFin.zip
            and put it in datasets/DGraph/raw/
  Elliptic  downloaded automatically by PyG the first time it is loaded
            (e.g. bash scripts/data_download.sh)
The loaders process the raw files on first use. GraphCLIP pre-training on Elliptic also needs
its subgraph summaries, which are generated with GraphCLIP/generate_summary_api.py.
EOF
}

download_datasets() {
    # Splits are small (~0.3 GB) and their folder names do not always match the dataset
    # folder names (e.g. datasets/Wiki vs datasets_split/WIKI), so they are always fetched in full.
    local patterns=("datasets_split/*") requested=() withheld=() name
    for name in "$@"; do
        if is_withheld "$name"; then withheld+=("$name"); else requested+=("$name"); fi
    done
    if [[ $# -eq 0 ]]; then
        patterns+=("datasets/*")
    else
        for name in ${requested[@]+"${requested[@]}"}; do patterns+=("datasets/${name}/*"); done
    fi
    hf download "$DATA_REPO" --repo-type dataset --local-dir . --include "${patterns[@]}"

    for name in ${requested[@]+"${requested[@]}"}; do
        [[ -d "datasets/${name}" ]] || echo "Warning: no dataset folder '${name}' in ${DATA_REPO} (names are case-sensitive, e.g. 'ogbn-arxiv', 'FB15K-237')." >&2
    done
    if [[ $# -eq 0 || ${#withheld[@]} -gt 0 ]]; then print_withheld_help; fi
}

download_splits() {
    hf download "$DATA_REPO" --repo-type dataset --local-dir . --include "datasets_split/*"
}

download_checkpoints() {
    hf download "$CKPT_REPO" --repo-type model --local-dir . \
        --include "checkpoints/*" "FAGCN/*" "GCOPE/*" "GFT/*" "GraphPrompt/*" "UniGraph2/*"
}

target="${1:-}"
[[ $# -gt 0 ]] && shift
case "$target" in
    all)         download_datasets; download_checkpoints ;;
    datasets)    download_datasets "$@" ;;
    splits)      download_splits ;;
    checkpoints) download_checkpoints ;;
    *)
        sed -n '2,14p' "$SCRIPT" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
