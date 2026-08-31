import os
import json
import shutil
import argparse
import subprocess
import socket


def get_region_from_resource_config():

    filepath = "/opt/ml/config/resource_config.json"
    with open(filepath, 'r') as f:
        d = json.loads(f.read())
    cluster_arn = d["ClusterConfig"]["ClusterArn"]
    region = cluster_arn.split(":")[3]
    return region


def create_file_from_template(template_path, output_path, replacements):
    with open(template_path, 'r') as template_file:
        
        template = template_file.read()

    # use Python's string substitution
    content = template.format(**replacements)

    with open(output_path, 'w') as output_file:
        output_file.write(content)


def install_observability(node_type, prometheus_remote_write_url, advanced=False, nccl_metrics=False):

    region = get_region_from_resource_config()
    hostname = socket.gethostname()

    env_vars = os.environ.copy()
    env_vars["REGION"] = region
    env_vars["ADVANCED"] = "1" if advanced else "0"

    if nccl_metrics and node_type == "compute":
        path = "/var/lib/node_exporter/nccl_inspector"
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)

    if node_type=="controller":

        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            "./otel_config/config-head-template.yaml",
            "/etc/otel/config.yaml",
            {
                "REGION": region,
                "AMPREMOTEWRITEURL": prometheus_remote_write_url,
                "HOSTNAME": hostname
            }
        )

        subprocess.run( ["bash", "./install_node_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_slurm_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_otel_collector.sh"], env=env_vars )

    elif node_type=="compute":

        os.makedirs("/etc/dcgm-exporter", exist_ok=True)
        if advanced:
            shutil.copy("./dcgm_metrics_config/dcgm-metrics-advanced.csv", "/etc/dcgm-exporter/dcgm-metrics.csv")
        else:
            shutil.copy("./dcgm_metrics_config/dcgm-metrics-basic.csv", "/etc/dcgm-exporter/dcgm-metrics.csv")

        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            "./otel_config/config-compute-template.yaml",
            "/etc/otel/config.yaml",
            {
                "REGION": region,
                "AMPREMOTEWRITEURL": prometheus_remote_write_url,
                "HOSTNAME": hostname
            }
        )

        subprocess.run( ["bash", "./install_node_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_dcgm_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_efa_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_otel_collector.sh"], env=env_vars )

    elif node_type=="login":

        os.makedirs("/etc/otel", exist_ok=True)
        create_file_from_template(
            "./otel_config/config-login-template.yaml",
            "/etc/otel/config.yaml",
            {
                "REGION": region,
                "AMPREMOTEWRITEURL": prometheus_remote_write_url,
                "HOSTNAME": hostname
            }
        )

        subprocess.run( ["bash", "./install_node_exporter.sh"], env=env_vars )
        subprocess.run( ["bash", "./install_otel_collector.sh"], env=env_vars )


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Script to install HyperPod observability")
    argparser.add_argument('--node-type', action="store", required=True, help='Node type (controller, login, compute)')
    argparser.add_argument('--prometheus-remote-write-url', action="store", required=True, help='Prometheus remote write URL (e.g. https://aps-workspaces.us-west-2.amazonaws.com/workspaces/ws-e37c0ee4-7f7f-4f65-b72b-5455852d0c23/api/v1/remote_write)')
    argparser.add_argument('--advanced', action="store_true", default=False, help='Advanced setup (default: False)')
    argparser.add_argument('--nccl-metrics', action="store_true", default=False, help='Enable NCCL metrics collection (default: False)')
    args = argparser.parse_args()

    assert args.node_type in ["controller", "login", "compute"]

    print("Starting observability installation")

    install_observability(args.node_type, args.prometheus_remote_write_url, args.advanced, args.nccl_metrics)

    print("---")
    print("Finished observability installation")
