#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

MODEL="gcope"
backbone="fagcn"
source_dataset_str="pretrain"
MODEL_ID="exp3"
EXP_ID="exp3"

TASK_NAMES=("graph")  # ("node" "edge" "graph")
NUM_SHOTS=(1 5)         # 5
SEEDS=(0)
COMMON_ARGS="
    --general.model_id ${MODEL_ID} \
    --general.exp_id ${EXP_ID} \
    --general.func adapt \
    --general.save_dir GCOPE/storage/${backbone}/balanced_few_shot_fine_tune_backbone_with_rec \
    --general.reconstruct 0.0 \
    --data.node_feature_dim 100 \
    --adapt.method finetune \
    --model.backbone.model_type ${backbone} \
    --model.saliency.model_type none \
    --adapt.method finetune \
    --adapt.pretrained_file GCOPE/storage/${backbone}/reconstruct_${MODEL_ID}/${source_dataset_str}_pretrained_model.pt \
    --adapt.finetune.learning_rate 1e-4 \
    --adapt.repeat_times 50 \
    --adapt.batch_size 32768 \
    --adapt.finetune.backbone_tuning 1 \
"

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
            echo "====================================" >> $LOGFILE

            nohup python GCOPE/exec.py ${COMMON_ARGS} \
                --data.name ${TASK} \
                --general.few_shot ${SHOT} \
                --general.seed ${SEED} \
                >> $LOGFILE 2>&1 &

            echo $! >> $PID_FILE
            sleep 5
        done
    done
done

echo "View real-time logs with: tail -f $LOGFILE" | tee -a $LOGFILE
echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE
