import os
import shutil
import argparse
import subprocess
import socket


def stop_observability(node_type):

    if node_type=="controller":

        subprocess.run( ["systemctl", "stop", "slurm_exporter"] )
        subprocess.run( ["docker", "stop", "node-exporter"] )
        subprocess.run( ["docker", "stop", "otel-collector"] )

    elif node_type=="compute":

        subprocess.run( ["docker", "stop", "node-exporter"] )
        subprocess.run( ["docker", "stop", "dcgm-exporter"] )
        subprocess.run( ["docker", "stop", "efa-exporter"] )
        subprocess.run( ["docker", "stop", "otel-collector"] )

    elif node_type=="login":

        subprocess.run( ["docker", "stop", "node-exporter"] )
        subprocess.run( ["docker", "stop", "otel-collector"] )


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Script to stop HyperPod observability")
    argparser.add_argument('--node-type', action="store", required=True, help='Node type (controller, login, compute)')
    args = argparser.parse_args()

    assert args.node_type in ["controller", "login", "compute"]

    print("Stopping observability")

    stop_observability(args.node_type)

    print("---")
    print("Stopping observability")
