#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

MODEL="tgn"
TASK="pretrain"
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --model_id exp2 \
    --task_name ${TASK} \
    --pattern single \
    --preprocess simple \
    --mode lp \
    --backbone tgn \
    --use_gpu True \
    --devices 0 \
    --gpu_type cuda \
    --batch_size 200 \
    --num_workers 4 \
    --learning_rate 0.0001 \
    --epochs 200 \
    --patience 5 \
    --dropout 0.1 \
    --compress_function none \
    --input_dim -1 \
    --hidden_dim 100 \
    --num_heads 2 \
    --num_neg_samples 1 \
    --is_logging True"

NUM_GPUS=1
PORT=$(shuf -i 29500-65535 -n 1)

mkdir -p logs/${MODEL}
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE=logs/${MODEL}/${MODEL}_${TASK}_${timestamp}.log
PID_FILE=${MODEL}_${TASK}_${timestamp}.pid
: > $PID_FILE

echo "=== ${MODEL^^} Pretraining Started ===" > $LOGFILE
echo "GPU: ${CUDA_VISIBLE_DEVICES}" >> $LOGFILE
for SEED in "${SEEDS[@]}"
do
    echo "====================================" >> $LOGFILE
    echo "Seed: ${SEED}" >> $LOGFILE
    echo "Port: ${PORT}" >> $LOGFILE
    echo "====================================" >> $LOGFILE

    nohup torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --rdzv_id=exp1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:${PORT} \
        run.py ${COMMON_ARGS} --seed ${SEED} \
        >> $LOGFILE 2>&1 &

    echo $! >> $PID_FILE
    PORT=$((PORT + 1))
    sleep 5
done

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE
