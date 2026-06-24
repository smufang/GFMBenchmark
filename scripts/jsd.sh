#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p logs/jsd
mkdir -p pids
mkdir -p z_temp

# Setup timestamps and file paths
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/jsd/jsd_${timestamp}.log"
PID_FILE="pids/jsd_${timestamp}.pid"

# Log the start of the pipeline
echo "=== JSD Pipeline Started ===" | tee -a "$LOGFILE"
echo "Workdir: $PWD" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# Run the python module in the background
nohup python -m utils.jsd >> "$LOGFILE" 2>&1 &

# Save the background process ID
echo $! > "$PID_FILE"

# Log the process info and commands to stop/tail
echo "PID file: $PID_FILE" | tee -a "$LOGFILE"

# Added tee command to write these instructions into the log file as well
echo "Tail logs: tail -f $LOGFILE" | tee -a "$LOGFILE"
echo "Stop job: kill $(cat "$PID_FILE")" | tee -a "$LOGFILE"