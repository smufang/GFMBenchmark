#!/bin/bash

##### Dgi and GraphPrompt share the same pre-training file. ####

GPU=0
DATASET=("ACM") # "Cora" "ACM" "Reddit" "Wisconsin" "Elliptic" "Photo" "HIV" "COX2" "PROTEINS" "FB15K-237"

mkdir -p logs/dgi
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/dgi/dgi_pretrain_${timestamp}.log

echo "=== dgi Pretraining Started ==="

for DATASET in "${DATASET[@]}"
do
    echo "Dataset: ${DATASET}"
    nohup python -u GraphPrompt/pretrain.py --gpu ${GPU} --dataset ${DATASET} \
    > $LOGFILE 2>&1 & PID=$!
done


echo "PID: ${PID}"
echo "Log file: ${LOGFILE}"
echo "Check by: tail -f ${LOGFILE}"
echo "To stop dgi_pretrain.sh: kill $$"
echo "To stop GraphPrompt/pretrain.py: kill ${PID}"