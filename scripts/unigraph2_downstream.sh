#!/bin/bash
#nohup bash ./scripts/unigraph2_downstream.sh > unigraph2_downstream_run.log 2>&1 &

mkdir -p logs/unigraph2
timestamp=$(date +"%Y%m%d_%H%M%S")

EXP=2
GPU=0
BATCH_SIZE=256
TASK_NAME=node # node edge graph
NUM_SHOTS=(1 5)

EXP1_node_DATASETS=(Pubmed Wikipedia Actor Chameleon Products T-Finance DGraph ogbn-proteins ogbn-mag)
EXP1_edge_DATASETS=(WIKI WN18RR DGraph)
EXP1_graph_DATASETS=(BZR PCBA)

EXP2_node_DATASETS=(Cora)  # ACM Reddit Wisconsin Photo Elliptic
EXP2_edge_DATASETS=(FB15K237)
EXP2_graph_DATASETS=(HIV COX2 PROTEINS)

EXP3_node_DATASETS=(Wikipedia Actor Products T-Finance DGraph ogbn-proteins)
EXP3_edge_DATASETS=(WIKI WN18RR DGraph)
EXP3_graph_DATASETS=(BZR PCBA)

EXP4_node_DATASETS=(Pubmed Products Actor Wikipedia Chameleon ogbn-mag)
EXP4_edge_DATASETS=(WIKI WN18RR)
EXP4_graph_DATASETS=(PCBA)

DATASETS_VAR=EXP${EXP}_${TASK_NAME}_DATASETS
DATASETS=("${!DATASETS_VAR}")

echo "======================================"
echo "Start downstream"
echo "EXP: ${EXP}"
echo "TASK: ${TASK_NAME}"
echo "Datasets: ${DATASETS[*]}"
echo "To stop unigraph2_downstream.sh: kill $$"
echo "======================================"

PID_FILE=logs/unigraph2/unigraph2_down_${EXP}_${TASK_NAME}_${SHOT}_${timestamp}.pid

for SHOT in "${NUM_SHOTS[@]}"
do

    for dataset in "${DATASETS[@]}"
    do
        LOGFILE=logs/unigraph2/unigraph2_down_${EXP}_${TASK_NAME}_${dataset}_${SHOT}_${timestamp}.log
        echo "[START] dataset=${dataset}, shot=${SHOT}"
        echo "Log file: ${LOGFILE}"

        nohup python -u UniGraph2/downstream.py \
                --exp ${EXP} \
                --task_name ${TASK_NAME} \
                --n_shot ${SHOT} \
                --gpu ${GPU} \
                --batch_size ${BATCH_SIZE} \
                --dataset ${dataset} \
                2>&1 | tee ${LOGFILE}

        echo $! >> $PID_FILE
        echo "[DONE] dataset=${dataset}, shot=${SHOT}"
        echo "--------------------------------------"
    done
done

echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE