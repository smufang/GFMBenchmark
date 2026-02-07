#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="heco"
TASK_NAMES=("node")  # ("node" "edge" "graph")
NUM_SHOTS=(1 5)         # 5
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --model_id exp2 \
    --exp_id exp2 \
    --pattern single \
    --preprocess simple \
    --mode none \
    --backbone gcn \
    --use_gpu True \
    --devices 0 \
    --gpu_type cuda \
    --num_workers 0 \
    --learning_rate 0.05 \
    --epochs 50 \
    --patience 50 \
    --compress_function none \
    --input_dim -1 \
    --hidden_dim 64 \
    --num_layers 0 \
    --num_heads 0 \
    --temperature 0.8 \
    --aug_type edge \
    --weight_decay 0.0 \
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

            nohup python run.py ${COMMON_ARGS} \
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
