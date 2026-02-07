#!/bin/bash
#nohup bash ./scripts/graphprompt_pretrain.sh > graphprompt_pretrain_run.log 2>&1 &

GPU=0
DATASET=("Reddit") # "Cora" "ACM" "Reddit" "Wisconsin" "Elliptic" "Photo" "HIV" "COX2" "PROTEINS" "FB15K-237"

mkdir -p logs/graphprompt
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/graphprompt/graphprompt_pretrain_${timestamp}.log

echo "=== graphprompt Pretraining Started ==="

nohup python -u GraphPrompt/pretrain.py --gpu ${GPU} --data ${DATASET} > $LOGFILE 2>&1 &
PID=$!

echo "PID: ${PID}"
echo "Log file: ${LOGFILE}"
echo "Check by: tail -f ${LOGFILE}"
echo "To stop graphprompt_pretrain.sh: kill $$"
echo "To stop GraphPrompt/pretrain.py: kill ${PID}"