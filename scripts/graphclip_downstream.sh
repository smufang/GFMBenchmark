#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="graphclip"
TASK_NAMES=("graph")  # ("node" "edge" "graph")
NUM_SHOTS=(1)         # 5
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
  --model_id exp3 \
  --exp_id exp3 \
  --pattern simple \
  --batch_size 16384 \
  --num_workers 0 \
  --compress_function none \
  --input_dim 384 \
  --hidden_dim 1024 \
  --output_dim 384 \
  --sampler rw \
  --walk_step 256 \
  --lr 1e-4 \
  --epochs 100 \
  --patience 100 \
  --is_logging True \
  --num_tasks 50"

mkdir -p logs/${MODEL}
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${MODEL}/${MODEL}_downstream_${timestamp}.log
PID_FILE=${MODEL}_downstream_${timestamp}.pid
echo "=== ${MODEL^^} Downstream Tasks Started ===" > $LOGFILE
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> $LOGFILE
: > $PID_FILE

for TASK in "${TASK_NAMES[@]}"
do
    for SHOT in "${NUM_SHOTS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            echo "====================================" >> $LOGFILE
            echo "Task: ${TASK}" >> $LOGFILE
            echo "Num_shot: ${SHOT}" >> $LOGFILE
            echo "Seed: ${SEED}" >> $LOGFILE
            echo "====================================" >> $LOGFILE

            nohup python run_${MODEL}_test.py ${COMMON_ARGS} \
                --task_name ${TASK} --num_shots ${SHOT} --seed ${SEED} \
                >> $LOGFILE 2>&1 &

            echo $! >> $PID_FILE
            sleep 5
        done
    done
done

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE
