#!/bin/bash
#nohup bash ./scripts/unigraph2_pretrain.sh > unigraph2_pretrain_run.log 2>&1 &

EXP=1
GPU=0
mkdir -p logs/unigraph2
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/unigraph2/unigraph2_pretrain_${EXP}_${timestamp}.log

echo "=== UniGraph2 Pretraining Started ==="

nohup python -u UniGraph2/pretrain.py --exp ${EXP} --gpu ${GPU} > $LOGFILE 2>&1 &
PID=$!

echo "PID: ${PID}"
echo "Log file: ${LOGFILE}"
echo "Check by: tail -f ${LOGFILE}"
echo "To stop unigraph2_pretrain.sh: kill $$"
echo "To stop UniGraph2/pretrain.py: kill ${PID}"