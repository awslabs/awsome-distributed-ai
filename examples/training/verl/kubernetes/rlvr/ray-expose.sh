#!/bin/bash

# Use RAY_DASHBOARD_PORT from env_vars, default to 8265 if not set
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

EXPOSED=$(lsof -i :${RAY_DASHBOARD_PORT})
if [ "$?" == 0 ]; then
	echo "Ray is exposed on port ${RAY_DASHBOARD_PORT}"
else
	PID_FILE="$HOME/port-forward.pid"
	export SERVICEHEAD=$(kubectl get service | grep head-svc | awk '{print $1}' | head -n 1)

	kubectl port-forward --address 0.0.0.0 service/${SERVICEHEAD} ${RAY_DASHBOARD_PORT}:8265 > /dev/null 2>&1 &
	echo $! > "$PID_FILE"
	echo "Port-forward started, PID $! saved in $PID_FILE"
	sleep 1
	echo "Port forwarded, visit http://localhost:${RAY_DASHBOARD_PORT}"
fi