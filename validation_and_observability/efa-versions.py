#!/usr/bin/python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import subprocess
import re
from prettytable import PrettyTable
import argparse

def get_efa_installer_version(container=[]):
    try:
        version = subprocess.check_output(container + ['cat', '/opt/amazon/efa_installed_packages'])
        version = version.decode('utf-8')
        version = re.search(r'# EFA installer version: (\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_libfabric_version(container=[]):
    try:
        version = subprocess.check_output(container + ['fi_info', '--version'])
        version = version.decode('utf-8')
        version = re.search(r'libfabric: (\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_nccl_version(container=[]):
    try:
        version = subprocess.check_output(container + ['locate', 'nccl'])
        version = version.decode('utf-8')
        print(version)
        version = re.search(r'/usr/local/cuda-12.8/lib/libnccl.so.(\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_aws_ofi_nccl_version(container=[]):
    try:
        lib_path = subprocess.check_output(container + ['cat', '/etc/ld.so.conf.d/100_ofinccl.conf']).decode().strip()
        version = subprocess.check_output(container + ['strings', f'{lib_path}/libnccl-net.so'])
        version = version.decode('utf-8')
        version = re.search(r'NET/OFI Initializing aws-ofi-nccl (\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_cuda_driver_version(container=[]):
    try:
        version = subprocess.check_output(container + ['nvidia-smi'])
        version = version.decode('utf-8')
        version = re.search(r'Driver Version: (\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_cuda_version(container=[]):
    try:
        version = subprocess.check_output(container + ['nvcc', '--version'])
        version = version.decode('utf-8')
        version = re.search(r', V(\d+\.\d+\.\d+)', version).group(1)
        return version
    except Exception as e:
        print(f'Error: {e}')
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script checks the versions of NCCL, EFA, Libfabric and CUDA. ')
    parser.add_argument('-c', '--container-image', type=str, help='Container image to get versions from.')
    args = parser.parse_args()

    if args.container_image:
        table = PrettyTable(["Package", "Local", "Container"])
        print(f'Getting versions from container image {args.container_image}.')
        container=['docker', 'run', '--gpus=all', args.container_image]
        table.add_row(["EFA installer version:", get_efa_installer_version(), get_efa_installer_version(container)])
        table.add_row(['NCCL Version', get_nccl_version(), get_nccl_version(container)])
        table.add_row(['Libfabric Version', get_libfabric_version(), get_libfabric_version(container)])
        table.add_row(["AWS OFI NCCL version:", get_aws_ofi_nccl_version(), get_aws_ofi_nccl_version(container)])
        table.add_row(['Nvidia Driver', get_cuda_driver_version(), get_cuda_driver_version(container)])
        table.add_row(["CUDA Version:", get_cuda_version(), get_cuda_version(container)])
    else:
        table = PrettyTable(["Package", "Version"])
        table.add_row(["EFA installer version:", get_efa_installer_version(),])
        table.add_row(['NCCL Version', get_nccl_version()])
        table.add_row(['Libfabric Version', get_libfabric_version()])
        table.add_row(["AWS OFI NCCL version:", get_aws_ofi_nccl_version()])
        table.add_row(['Nvidia Driver', get_cuda_driver_version()])
        table.add_row(["CUDA Version:", get_cuda_version()])

    table.align = "l"  # Align the columns to the left
    table.padding_width = 2  # Set the padding width for each cell
    table.hrules = 1  # Add horizontal rules between rows

    print(table)
