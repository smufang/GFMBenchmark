#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1 # stuck from input batch when using DataParallel 
export PYTHONUNBUFFERED=1

MODEL="graphclip"
TASK="pretrain"
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --model_id exp3 \
    --task_name ${TASK} \
    --pattern simple \
    --compress_function none \
    --batch_size 512 \
    --lr 1e-5 \
    --epochs 30 \
    --patience 30 \
    --input_dim 384 \
    --hidden_dim 1024 \
    --is_logging True "

mkdir -p logs/${MODEL}
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${MODEL}/${MODEL}_${TASK}_${timestamp}.log
PID_FILE=${MODEL}_${TASK}_${timestamp}.pid
: > $PID_FILE

echo "=== ${MODEL^^} Pretraining Started ===" > $LOGFILE
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> $LOGFILE

for SEED in "${SEEDS[@]}"; do
  echo "====================================" >> "$LOGFILE"
  echo "Seed: ${SEED}" >> "$LOGFILE"
  echo "====================================" >> "$LOGFILE"

  nohup python run_${MODEL}_train.py ${COMMON_ARGS} --seed ${SEED} \
    >> $LOGFILE 2>&1 &

  echo $! >> $PID_FILE
  sleep 3
done

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE