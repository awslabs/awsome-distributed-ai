# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# TensorBoard Deployment + ClusterIP Service (reach it with kubectl port-forward)
# Generated from config.yaml -- do not edit directly
# ============================================================================
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isaaclab-tensorboard
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: isaaclab-tensorboard
  template:
    metadata:
      labels:
        app: isaaclab-tensorboard
    spec:
      containers:
      - name: tensorboard
        image: ${IMAGE}
        command: ["/bin/bash", "-c"]
        args:
        - |
          echo "Waiting for training logs to appear on FSx..."
          LOG_DIRS="${JOB_FSX_LOG_DIR}"
          for i in $$(seq 1 120); do
            for dir in $$LOG_DIRS; do
              if find $$dir -name "events.out.tfevents*" 2>/dev/null | head -1 | grep -q .; then
                echo "Found TensorBoard event files in $$dir!"
                break 2
              fi
            done
            echo "Attempt $$i: No event files yet, waiting 10s..."
            sleep 10
          done
          echo "Starting TensorBoard on port 6006..."
          /opt/isaaclab-venv/bin/python -m tensorboard.main \
            --logdir /fsx \
            --host 0.0.0.0 \
            --port 6006 \
            --reload_interval 15
        env:
        - name: ACCEPT_EULA
          value: "Y"
        - name: PRIVACY_CONSENT
          value: "Y"
        # The Isaac Lab image sets NVIDIA_VISIBLE_DEVICES=all; TensorBoard
        # needs no GPU, so don't expose any (and allow CPU-only nodes).
        - name: NVIDIA_VISIBLE_DEVICES
          value: "void"
        ports:
        - containerPort: 6006
          name: tensorboard
        volumeMounts:
        - name: fsx
          mountPath: /fsx
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: fsx
        persistentVolumeClaim:
          claimName: ${FSX_PVC_NAME}
      nodeSelector:
        ${NODE_HEALTH_LABEL}: ${NODE_HEALTH_VALUE}
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
---
apiVersion: v1
kind: Service
metadata:
  name: isaaclab-tensorboard
  namespace: default
spec:
  selector:
    app: isaaclab-tensorboard
  ports:
  - port: 6006
    targetPort: 6006
  type: ClusterIP
