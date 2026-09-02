#!/bin/bash
PID_FILE="$HOME/port-forward.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ -z "$PID" ]; then
        echo "PID file is empty."
        rm -f "$PID_FILE"
        exit 1
    fi
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "Process $PID stopped."
    else
        echo "No process found with PID $PID."
    fi
    rm -f "$PID_FILE"
else
    echo "PID file not found."
fi