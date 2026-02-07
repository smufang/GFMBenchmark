#!/bin/bash
#nohup bash ./scripts/gft_pretrain.sh > gft_pretrain_run.log 2>&1 &

EXP=1
GPU=3
mkdir -p logs/gft
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/gft/gft_pretrain_${EXP}_${timestamp}.log

echo "=== GFT Pretraining Started ==="

nohup python -u GFT/GFT/pretrain.py --exp ${EXP} --gpu ${GPU} > $LOGFILE 2>&1 &
PID=$!

echo "PID: ${PID}"
echo "Log file: ${LOGFILE}"
echo "Check by: tail -f ${LOGFILE}"
echo "To stop gft_pretrain.sh: kill $$"
echo "To stop GFT/GFT/pretrain.py: kill ${PID}"