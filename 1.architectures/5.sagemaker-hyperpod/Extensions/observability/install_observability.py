#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Orchestrator script that installs the observability stack on a HyperPod node.
# Called by setup_observability.sh after node type detection and config parsing.

import os
import json
import shutil
import argparse
import subprocess
import socket


def get_region_from_resource_config():
    """Extract the AWS region from the HyperPod resource_config.json."""
    filepath = "/opt/ml/config/resource_config.json"
    with open(filepath, "r") as f:
        d = json.load(f)
    cluster_arn = d["ClusterConfig"]["ClusterArn"]
    region = cluster_arn.split(":")[3]
    return region


def create_file_from_template(template_path, output_path, replacements):
    """Render an OTel config template with the given replacements."""
    with open(template_path, "r") as template_file:
        template = template_file.read()
    content = template.format(**replacements)
    with open(output_path, "w") as output_file:
        output_file.write(content)


def install_observability(node_type, prometheus_remote_write_url, advanced=False, nccl_metrics=False):
    region = get_region_from_resource_config()
    hostname = socket.gethostname()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    env_vars = os.environ.copy()
    env_vars["REGION"] = region
    env_vars["ADVANCED"] = "1" if advanced else "0"

    if nccl_metrics and node_type == "compute":
        path = "/var/lib/node_exporter/nccl_inspector"
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)

    otel_config_dir = os.path.join(script_dir, "otel_config")
    replacements = {
        "REGION": region,
        "AMPREMOTEWRITEURL": prometheus_remote_write_url,
        "HOSTNAME": hostname,
    }

    if node_type == "controller":
        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            os.path.join(otel_config_dir, "config-head-template.yaml"),
            "/etc/otel/config.yaml",
            replacements,
        )
        subprocess.run(["bash", os.path.join(script_dir, "install_node_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_slurm_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_otel_collector.sh")], env=env_vars, check=True)

    elif node_type == "compute":
        os.makedirs("/etc/dcgm-exporter", exist_ok=True)
        dcgm_config_dir = os.path.join(script_dir, "dcgm_metrics_config")
        if advanced:
            shutil.copy(os.path.join(dcgm_config_dir, "dcgm-metrics-advanced.csv"), "/etc/dcgm-exporter/dcgm-metrics.csv")
        else:
            shutil.copy(os.path.join(dcgm_config_dir, "dcgm-metrics-basic.csv"), "/etc/dcgm-exporter/dcgm-metrics.csv")

        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            os.path.join(otel_config_dir, "config-compute-template.yaml"),
            "/etc/otel/config.yaml",
            replacements,
        )
        subprocess.run(["bash", os.path.join(script_dir, "install_node_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_dcgm_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_efa_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_otel_collector.sh")], env=env_vars, check=True)

    elif node_type == "login":
        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            os.path.join(otel_config_dir, "config-login-template.yaml"),
            "/etc/otel/config.yaml",
            replacements,
        )
        subprocess.run(["bash", os.path.join(script_dir, "install_node_exporter.sh")], env=env_vars, check=True)
        subprocess.run(["bash", os.path.join(script_dir, "install_otel_collector.sh")], env=env_vars, check=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Install HyperPod observability stack")
    argparser.add_argument("--node-type", required=True, help="Node type (controller, login, compute)")
    argparser.add_argument("--prometheus-remote-write-url", required=True, help="AMP remote write URL")
    argparser.add_argument("--advanced", action="store_true", default=False, help="Enable advanced metrics")
    argparser.add_argument("--nccl-metrics", action="store_true", default=False, help="Enable NCCL metrics collection")
    args = argparser.parse_args()

    assert args.node_type in ["controller", "login", "compute"]

    print("Starting observability installation")
    install_observability(args.node_type, args.prometheus_remote_write_url, args.advanced, args.nccl_metrics)
    print("---")
    print("Finished observability installation")
