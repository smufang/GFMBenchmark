#!/bin/bash
# run one by one to avoid GPU OOM
mkdir -p logs/dgi
timestamp=$(date +"%Y%m%d_%H%M%S")

GPU=0
BATCH_SIZE=256
TASK_NAME=node # node edge graph
NUM_SHOTS=(1 5)

EXP_node_DATASETS=(Cora ACM Reddit Wisconsin Photo Elliptic) # Cora ACM Reddit Wisconsin Photo Elliptic
EXP_edge_DATASETS=(FB15K237)
EXP_graph_DATASETS=(HIV COX2 PROTEINS)

DATASETS_VAR=EXP_${TASK_NAME}_DATASETS
DATASETS=("${!DATASETS_VAR}")

echo "======================================"
echo "Start downstream"
echo "TASK: ${TASK_NAME}"
echo "Datasets: ${DATASETS[*]}"
echo "To stop dgi_downstream.sh: kill $$"
echo "======================================"

PID_FILE=logs/dgi/dgi_down_${TASK_NAME}_${SHOT}_${timestamp}.pid

for SHOT in "${NUM_SHOTS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        LOGFILE=logs/dgi/dgi_down_${TASK_NAME}_${dataset}_${SHOT}_${timestamp}.log
        echo "[START] dataset=${dataset}, shot=${SHOT}"
        echo "Log file: ${LOGFILE}"

        nohup python -u GraphPrompt/downstream.py \
                --usemlp yes \
                --task_name ${TASK_NAME} \
                --n_shot ${SHOT} \
                --gpu ${GPU} \
                --batch_size ${BATCH_SIZE} \
                --data ${dataset} \
                > ${LOGFILE} 2>&1 &

        echo $! >> $PID_FILE
        echo "[STARTED BACKGROUND] dataset=${dataset}, shot=${SHOT}, PID=$!"
        echo "[DONE] dataset=${dataset}, shot=${SHOT}"
        echo "--------------------------------------"
    done
done

echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE