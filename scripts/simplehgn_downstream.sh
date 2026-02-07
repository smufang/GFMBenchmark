#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="simple_hgn"
TASK_NAMES=("node")  # ("node" "edge" "graph")
NUM_SHOTS=(1 5)         # 5
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --model_id none \
    --exp_id exp2 \
    --pattern none \
    --mode none \
    --backbone gat \
    --use_gpu True \
    --devices 0 \
    --gpu_type cuda \
    --batch_size 32768 \
    --num_workers 4 \
    --learning_rate 5e-4 \
    --epochs 300 \
    --patience 300 \
    --dropout 0.5 \
    --activation elu \
    --compress_function none \
    --cache_compress True \
    --criterion nll \
    --input_dim -1 \
    --hidden_dim 64 \
    --edge_dim 64 \
    --num_layers 2 \
    --num_heads 8 \
    --alpha 0.05 \
    --beta 0.05 \
    --weight_decay 1e-4 \
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
