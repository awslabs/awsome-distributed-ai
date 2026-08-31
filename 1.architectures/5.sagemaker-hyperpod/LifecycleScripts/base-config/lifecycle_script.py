#!/usr/bin/env python

import argparse
from enum import Enum
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from config import Config


SLURM_CONF = os.getenv("SLURM_CONF", "/opt/slurm/etc/slurm.conf")

class SlurmNodeType(str, Enum):
    HEAD_NODE = "controller"
    LOGIN_NODE = "login"
    COMPUTE_NODE = "compute"


class ExecuteBashScript:
    def __init__(self, script_name: str):
        self.script_name = script_name

    def run(self, *args):
        print(f"Execute script: {self.script_name} {' '.join([str(x) for x in args])}")
        result = subprocess.run(["sudo", "bash", self.script_name, *args])
        result.check_returncode()
        print(f"Script {self.script_name} executed successully")


class ResourceConfig:
    INSTANCE_GROUP_NAME = "Name"
    INSTANCE_NAME = "InstanceName"
    CUSTOMER_IP_ADDRESS = "CustomerIpAddress"

    def __init__(self, path: str):
        with open(path, "r") as f:
            self._config = json.load(f)

    def find_instance_by_address(self, address) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        for group in self._config["InstanceGroups"]:
            for instance in group.get("Instances") or []:
                if instance.get(ResourceConfig.CUSTOMER_IP_ADDRESS) == address:
                    return group, instance
        return None, None

    def get_list_of_addresses(self, group_name) -> List[str]:
        for group in self._config["InstanceGroups"]:
            if group.get(ResourceConfig.INSTANCE_GROUP_NAME) != group_name:
                continue
            return [i.get(ResourceConfig.CUSTOMER_IP_ADDRESS) for i in group.get("Instances") or []]
        return []


class ProvisioningParameters:
    WORKLOAD_MANAGER_KEY: str = "workload_manager"
    FSX_DNS_NAME: str = "fsx_dns_name"
    FSX_MOUNT_NAME: str = "fsx_mountname"
    FSX_OPENZFS_DNS_NAME: str = "fsx_openzfs_dns_name"
    SLURM_CONFIGURATIONS: str = "slurm_configurations"

    def __init__(self, path: str):
        with open(path, "r") as f:
            self._params = json.load(f)

    @property
    def workload_manager(self) -> Optional[str]:
        return self._params.get(ProvisioningParameters.WORKLOAD_MANAGER_KEY)

    @property
    def fsx_settings(self) -> Tuple[str, str]:
        return self._params.get(ProvisioningParameters.FSX_DNS_NAME), self._params.get(ProvisioningParameters.FSX_MOUNT_NAME)

    @property
    def fsx_openzfs_settings(self) -> Optional[str]:
        return self._params.get(ProvisioningParameters.FSX_OPENZFS_DNS_NAME)

    @property
    def controller_group(self) -> Optional[str]:
        return self._params.get("controller_group")

    @property
    def login_group(self) -> Optional[str]:
        return self._params.get("login_group")

    @property
    def slurm_configurations(self) -> Dict[str, Any]:
        slurm_configurations = self._params.get(ProvisioningParameters.SLURM_CONFIGURATIONS)
        if not slurm_configurations:
            return {}

        return slurm_configurations

def get_ip_address():
    max_retries = 7
    retry_delay_seconds = 5
    IP = '127.0.0.1'

    retry_count = 0
    while retry_count < max_retries:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't even have to be reachable
            s.connect(('10.254.254.254', 1))
            IP = s.getsockname()[0]
            break
        except Exception as e:
            print(f"Failed to get IP address of the current host. Reason: {repr(e)}.")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
                retry_delay_seconds = retry_delay_seconds * 2  # Exponential backoff
            else:    
                print(f"Exceeded maximum retries ({max_retries}) to get IP address. Returning default IP address {IP}.")
        finally:
            s.close()
    return IP


def wait_for_slurm_conf(controllers: List[str]) -> bool:
    """
    SageMaker agents do a slurm configuration. Wait for a signal that slurm is ready to start.
    This means that we can start controller, or do additional setup
    Returns:
        bool: True if valid slurm configuration found
    """
    sleep = 5 # sec
    timeout = 60  # sec
    for i in range(timeout // sleep):
        if not os.path.exists(SLURM_CONF):
            print("slurm.conf is not present. It is fine for login/compute nodes")
            return True
        with open(SLURM_CONF, "rt") as f:
            data = f.read()
            # check if controller information is present
            for ip in controllers:
                if ip in data:
                    print("slurm.conf found. It contains at least one controller address")
                    return True
        time.sleep(sleep)
    return False

def wait_for_scontrol():
    """
    Checks if 'scontrol show nodes' command returns output within specified time.
    This means we can proceed with install scripts which require nodes to be registered with slurm.
    Returns:
        bool: True if the command returns output from scontrol within the specified time, False otherwise.
    """
    timeout = 120
    sleep = 5
    for i in range (timeout // sleep):
        try:
            output = subprocess.check_output(['scontrol', 'show', 'nodes'])
            if output.strip():
                print("Nodes registered with Slurm, Proceeding with install scripts.", output)
                return True
        except subprocess.CalledProcessError:
            pass

        print(f"Waiting for output. Retrying in {sleep} seconds...")
        time.sleep(sleep)

    print(f"Exceeded maximum wait time of {timeout} seconds. No output from scontrol.")
    return False


def main(args):
    params = ProvisioningParameters(args.provisioning_parameters)
    resource_config = ResourceConfig(args.resource_config)

    ExecuteBashScript("./utils/install_ansible.sh").run()

    fsx_dns_name, fsx_mountname = params.fsx_settings
    if fsx_dns_name and fsx_mountname:
        print(f"Mount fsx: {fsx_dns_name}. Mount point: {fsx_mountname}")
        ExecuteBashScript("./mount_fsx.sh").run(fsx_dns_name, fsx_mountname, "/fsx")

    # Add FSx OpenZFS mount section
    fsx_openzfs_dns_name = params.fsx_openzfs_settings
    if Config.enable_fsx_openzfs and fsx_openzfs_dns_name:
        print(f"Mount FSx OpenZFS: {fsx_openzfs_dns_name}. Mount point: /home")
        ExecuteBashScript("./mount_fsx_openzfs.sh").run(fsx_openzfs_dns_name, "/home")

    ExecuteBashScript("./add_users.sh").run()

    if params.workload_manager == "slurm":
        # Wait until slurm will be configured
        controllers = resource_config.get_list_of_addresses(params.controller_group)
        wait_for_slurm_conf(controllers)

        print("This is a slurm cluster. Do additional slurm setup")
        self_ip = get_ip_address()
        head_node_ip = resource_config.get_list_of_addresses(params.controller_group)
        login_node_ip = resource_config.get_list_of_addresses(params.login_group)
        print(f"This node ip address is {self_ip}")

        group, instance = resource_config.find_instance_by_address(self_ip)
        if instance is None:
            raise ValueError("This instance not found in resource config. Can't process")
        print(group)

        node_type = SlurmNodeType.COMPUTE_NODE
        if group.get("Name") == params.controller_group:
            node_type = SlurmNodeType.HEAD_NODE
        elif group.get("Name") == params.login_group:
            node_type = SlurmNodeType.LOGIN_NODE

        if node_type == SlurmNodeType.HEAD_NODE:
            if params.slurm_configurations:
                ExecuteBashScript("./multi_headnode_setup/headnode_setup.sh").run()
            else:
                ExecuteBashScript("./setup_mariadb_accounting.sh").run()

        ExecuteBashScript("./apply_hotfix.sh").run(node_type)
        ExecuteBashScript("./utils/motd.sh").run(node_type, ",".join(head_node_ip), ",".join(login_node_ip))

        # Only configure home directory on FSx if either FSx Lustre or FSx OpenZFS is configured and provided in the provisioning params
        if (fsx_dns_name and fsx_mountname) or (Config.enable_fsx_openzfs and fsx_openzfs_dns_name):
            if Config.enable_fsx_openzfs and fsx_openzfs_dns_name:
                ExecuteBashScript("./utils/fsx_ubuntu.sh").run("1")
            else:
                ExecuteBashScript("./utils/fsx_ubuntu.sh").run("0")
        else:
            # No FSx DNS names provided in provisioning params.
            # Auto-detect existing FSx mounts (OpenZFS/Lustre) and setup home directories if present.
            # If no mounts are found, the script logs and exits silently.
            print("[INFO] No FSx DNS names (fsx_dns_name/fsx_mountname) or OpenZFS DNS name provided in provisioning parameters.")
            print("[INFO] Running fsx_auto_detect.sh to check for existing FSx mounts on this instance...")
            try:
                ExecuteBashScript("./utils/fsx_auto_detect.sh").run()
                print("[INFO] fsx_auto_detect.sh completed successfully.")
            except Exception as e:
                print(f"[WARN] fsx_auto_detect.sh failed: {e}. Continuing without mount-based home directory setup.")

        nccl_metrics_flag = "0"
        nccl_dump_interval = "30"
        nccl_plugin_path = ""
        if Config.enable_observability:
            from config import ObservabilityConfig
            if ObservabilityConfig.nccl_metrics_enabled:
                nccl_metrics_flag = "1"
                nccl_dump_interval = str(ObservabilityConfig.nccl_metrics_dump_interval_seconds)
                nccl_plugin_path = ObservabilityConfig.nccl_profiler_plugin_path

        ExecuteBashScript("./start_slurm.sh").run(node_type, ",".join(controllers), nccl_metrics_flag, nccl_dump_interval, nccl_plugin_path)
        
        # Setup user associations for Slurm accounting (only on controller nodes)
        if node_type == SlurmNodeType.HEAD_NODE:
            ExecuteBashScript("./setup_user_associations.sh").run()
        
        ExecuteBashScript("./utils/gen-keypair-ubuntu.sh").run()
        ExecuteBashScript("./utils/ssh-to-compute.sh").run()

        # Install metric exporting software and Prometheus for observability
        if Config.enable_observability:

            from config import ObservabilityConfig

            ExecuteBashScript("./utils/install_docker.sh").run()

            cmd = [
                "python3", "-u", "install_observability.py",
                "--node-type", node_type,
                "--prometheus-remote-write-url", ObservabilityConfig.prometheus_remote_write_url, 
            ]

            if ObservabilityConfig.advanced_metrics:
                cmd += ["--advanced"]

            if ObservabilityConfig.nccl_metrics_enabled:
                cmd += ["--nccl-metrics"]
            
            subprocess.run(cmd, cwd="./observability", check=True)
        
        # Install Docker/Enroot/Pyxis
        if Config.enable_docker_enroot_pyxis:
            ExecuteBashScript("./utils/install_docker.sh").run()
            ExecuteBashScript("./utils/install_enroot_pyxis.sh").run(node_type)

        # Install and configure SSSD for ActiveDirectory/LDAP integration
        if Config.enable_sssd:
            subprocess.run(["python3", "-u", "setup_sssd.py", "--node-type", node_type], check=True)

        if Config.enable_pam_slurm_adopt:
            ExecuteBashScript("./utils/slurm_fix_plugstackconf.sh").run()
            ExecuteBashScript("./utils/pam_adopt_cgroup_wheel.sh").run()

        if Config.enable_mount_s3:
            ExecuteBashScript("./utils/mount-s3.sh").run(Config.s3_bucket)

        if Config.enable_slurm_log_rotation:
            ExecuteBashScript("./utils/enable_slurm_log_rotation.sh").run()

    print("[INFO]: Success: All provisioning scripts completed")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-rc", "--resource_config", help="Resource config JSON file containing Ip_address of head, login and compute nodes")
    parser.add_argument("-pp", "--provisioning_parameters", help="Provisioning Parameters containing the head, login and compute ID/names")
    args=parser.parse_args()

    main(args)
