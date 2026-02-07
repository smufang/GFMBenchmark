#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

MODEL="mdgfm"
TASK="pretrain"
SEEDS=(0)
COMMON_ARGS="--model ${MODEL} \
    --model_id exp1 \
    --task_name ${TASK} \
    --pattern cross \
    --mode lp \
    --backbone gcn \
    --use_gpu True \
    --devices 0 \
    --gpu_type cuda \
    --num_neighbors 5 5 5 5\
    --max_nodes 40000 \
    --batch_size 256 \
    --num_workers 4 \
    --learning_rate 0.001 \
    --epochs 200 \
    --patience 20 \
    --dropout 0.1 \
    --activation prelu \
    --compress_function pca \
    --cache_compress True \
    --combinetype mul \
    --input_dim 50 \
    --hidden_dim 256 \
    --num_layers 3 \
    --num_heads 0 \
    --temperature 0.2 \
    --drop_percent 0.5 \
    --k 15 \
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
