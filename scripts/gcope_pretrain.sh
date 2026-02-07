#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

backbone="fagcn"
MODEL_ID="exp3"
# target='Citeseer'  

mkdir -p GCOPE/storage/${backbone}/reconstruct_${EXP}
mkdir -p logs/gcope
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/gcope/gcope_pretrain_${timestamp}.log

echo "=== GCOPE Pretraining Started ===" > $LOGFILE
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> $LOGFILE
echo "Timestamp: ${timestamp}" >> $LOGFILE
echo "====================================" >> $LOGFILE


nohup python GCOPE/exec.py \
    --config-file  GCOPE/pretrain.json \
    --general.model_id "${MODEL_ID}" \
    --general.func pretrain \
    --general.save_dir "GCOPE/storage/${backbone}/reconstruct_${MODEL_ID}/" \
    --general.reconstruct 0.2 \
    --data.name "pretrain" \
    >> $LOGFILE 2>&1 &

echo $! > gcope_pretrain_${timestamp}.pid
echo "You can view real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat gcope_pretrain_${timestamp}.pid) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat gcope_pretrain_${timestamp}.pid)" | tee -a $LOGFILE