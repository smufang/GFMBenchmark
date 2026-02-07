#!/bin/bash
#nohup bash ./scripts/fagcn_execute.sh > fagcn_execute_run.log 2>&1 &

mkdir -p logs/fagcn
timestamp=$(date +"%Y%m%d_%H%M%S")

NUM_SHOTS=(1 5)

echo "======================================"
echo "Start downstream"
echo "To stop fagcn_downstream.sh: kill $$"
echo "======================================"

PID_FILE=logs/fagcn/fagcn_down_${SHOT}_${timestamp}.pid

for SHOT in "${NUM_SHOTS[@]}"
do
    LOGFILE=logs/fagcn/fagcn_wisconsin_${SHOT}_${timestamp}.log
    echo "[START] wisconsin, shot=${SHOT}"
    echo "Log file: ${LOGFILE}"

    nohup python -u FAGCN/fagcn_execute.py \
            --n_shot ${SHOT} \
            2>&1 | tee ${LOGFILE}

    echo $! >> $PID_FILE
    echo "[DONE] wisconsin, shot=${SHOT}"
    echo "--------------------------------------"
done

echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE