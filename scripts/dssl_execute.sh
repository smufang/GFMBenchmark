#!/bin/bash
#nohup bash ./scripts/dssl_execute.sh > dssl_execute_run.log 2>&1 &

mkdir -p logs/dssl
timestamp=$(date +"%Y%m%d_%H%M%S")

NUM_SHOTS=(1)

echo "======================================"
echo "Start downstream"
echo "To stop dssl_downstream.sh: kill $$"
echo "======================================"

PID_FILE=logs/dssl/dssl_down_${SHOT}_${timestamp}.pid

for SHOT in "${NUM_SHOTS[@]}"
do
    LOGFILE=logs/dssl/dssl_wisconsin_${SHOT}_${timestamp}.log
    echo "[START] wisconsin, shot=${SHOT}"
    echo "Log file: ${LOGFILE}"

    nohup python -u FAGCN/dssl_execute.py \
            --n_shot ${SHOT} \
            2>&1 | tee ${LOGFILE}

    echo $! >> $PID_FILE
    echo "[DONE] wisconsin, shot=${SHOT}"
    echo "--------------------------------------"
done

echo "To check if the process is running, use: ps -p \$(cat $PID_FILE) | xargs" | tee -a $LOGFILE
echo "To stop the process, use: kill \$(cat $PID_FILE)" | tee -a $LOGFILE