# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import os
import re
import sys
import json
import socket
import pathlib
import argparse
import subprocess



def check_if_fsx_mounted():

    result = subprocess.run(
        ["df", "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in result.stdout.splitlines():
        if "/fsx" in line:
            return True

    return False


def check_if_user_directory_on_fsx() -> bool:
    """
    Checks if the user directory is using /fsx since EBS volumes are not allowed
    """
    user = os.environ.get("USER", None)
    if not user:
        raise Exception("$USER variable not set")
    return os.environ.get("HOME") == f"/fsx/{user}"


def check_if_docker_installed() -> bool:
    """Just check if docker is available on the node where you will build the"""
    try:
        # Run the command 'docker --version'
        result = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # If the command was successful, Docker is installed
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        # The command 'docker' was not found, Docker is not installed
        return False


def check_if_pyxis_installed() -> bool:
    """Check if pyxis is installed and available"""
    try:
        # Run the command and capture the output
        result = subprocess.run(
            ["srun", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if 'container-image' is in the command output
        if "pyxis" in result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def check_if_enroot_installed() -> bool:
    """enroot is installed and available"""
    try:
        # Run the command 'enroot'
        result = subprocess.run(
            ["sudo", "enroot"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # If the command was successful, enroot is installed
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        # The command 'enroot' was not found, enroot is not installed
        return False


def check_enroot_runtime_path() -> bool:
    """Check the enroot runtime path"""
    file_path = "/etc/enroot/enroot.conf"

    try:
        with open(file_path, "r") as file:
            for line in file:
                # Check if "ENROOT_RUNTIME_PATH" is in the line
                if re.search("ENROOT_RUNTIME_PATH", line):
                    return ("/opt/sagemaker/tmp/enroot/user-$(id -u)" in line.strip() or "/opt/dlami/nvme/tmp/enroot/user-$(id -u)" in line.strip())
                    print(line.strip())  # Print the matching line
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return False
    except PermissionError:
        print(f"Permission denied when trying to read {file_path}.")
        return False


def check_node_connectivity() -> bool:
    """Check if nodes are connected"""
    try:
        # Run the command 'enroot'
        result = subprocess.run(
            ["hostname"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # If the command was successful, enroot is installed
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        # The command 'enroot' was not found, enroot is not installed
        return False


def check_slurmd_service_status() -> bool:
    try:
        # Run the 'systemctl status' command for the given service
        result = subprocess.run(
            ["systemctl", "status", "slurmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the service is active
        if "active (running)" in result.stdout:
            return True
        elif "inactive" in result.stdout:
            return False
        elif "failed" in result.stdout:
            return False
        else:
            return False

    except Exception as e:
        return False


def nvidia_cli_installed() -> bool:
    """Make sure Nvidia Container CLI is installed and available

    Returns:
        _type_: _description_
    """
    try:
        # Run the command 'nvidia-container-cli --version'
        result = subprocess.run(
            ["nvidia-container-cli", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # If the command was successful, nvidia-container-cli is installed
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        # The command 'nvidia-container-cli' was not found, nvidia-container-cli is not installed
        return False


def check_docker_data_root():
    """Check data root"""
    file_path = "/etc/docker/daemon.json"

    try:
        json_data = json.loads(pathlib.Path(file_path).open().read())
        if not (json_data["data-root"] == "/opt/dlami/nvme/docker/data-root" or json_data["data-root"] == "/opt/sagemaker/docker/data-root"):
            print("Resolution: cd /tmp/sagemaker-lifecycle-* && cd src/utils/ && srun -N <no of nodes> bash install_docker.sh")
            return False
        else:
            return True
    except FileNotFoundError:
        print(f"The file {file_path} was not found. Resolution: cd /tmp/sagemaker-lifecycle-* && cd src/utils/ && srun -N <no of nodes> bash install_docker.sh")        
        return False
    except PermissionError:
        print(f"Permission denied when trying to read {file_path}. File should be readable across all users")
        return False
    except Exception as e:
        return False


def check_file_for_strings(file_path, search_strings):
    """
    Check if any of the specified strings are in the file.

    Parameters:
    - file_path: Path to the file to be checked.
    - search_strings: A list of strings to search for in the file.

    Returns:
    - True if none of the search strings are found in the file, False otherwise.
    """
    try:
        with open(file_path, "r") as file:
            for line in file:
                if any(search_string in line for search_string in search_strings):
                    print(f"\n\nFile contains {search_strings} which is an unsupported option on Hyperpod currently. Unless you've enabled gres/pmix explicitly - this won't work. Please refer to https://catalog.workshops.aws/sagemaker-hyperpod/en-US/04-advanced/08-gres\n\n")
                    return False
        return True
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return False
    except PermissionError:
        print(f"Permission denied when trying to read {file_path}.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runtime check for Hyperpod training")
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-r", "--runtime", action="store_false", help="Check runtime for provided nodes"
    )
    group.add_argument(
        "-f", "--filename", type=str, help="Name of script e.g. 1.training.sbatch"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_false", help="Enable verbose output"
    )
    parser.add_argument(
        "-w", "--who", type=str, help="Name of the user who's going to run scripts"
    )

    args = parser.parse_args()
    
    if args.filename:
        full_path = os.path.abspath(args.filename)
        # print(f"The full path of the script to check is: {full_path}")
        search_strings = ["--gpus", "pmix", "--gres=gpu"]
        if check_file_for_strings(full_path, search_strings):
            print(f"{full_path} \u2705")
        else:
            print(f"{full_path} \u274C")
        sys.exit()

    hostname = socket.gethostname()

    function_names = [
        "check_if_docker_installed",
        "check_enroot_runtime_path",
        "check_docker_data_root",
        "check_if_fsx_mounted",
        "check_if_pyxis_installed",
        "check_slurmd_service_status",
        "check_if_user_directory_on_fsx",
        "nvidia_cli_installed",
    ]

    results = {}

    for function_name in function_names:

        func = getattr(sys.modules[__name__], function_name)
        func_result = func()
        results[function_name] = func_result

        if args.verbose:
            print(f"{hostname}: test {function_name} - {func_result}")

    if all(list(results.values())):
        print(f"{hostname} ---------------- \u2705")
    else:
        print(f"{hostname} ---------------- \u274C")
