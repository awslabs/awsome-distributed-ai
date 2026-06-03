#!/usr/bin/env python3

"""
Update Helm values.yaml files with ECR image references
This script updates image repositories and tags in values.yaml files based on ECR configuration
"""

import argparse
import os
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path

# try:
#     from ruamel.yaml import YAML
# except ImportError:
#     print("Error: ruamel.yaml is required. Install with: pip install ruamel.yaml")
#     sys.exit(1)

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_colored(message, color):
    """Print message with color"""
    print(f"{color}{message}{Colors.NC}")

def get_account_id():
    """Auto-detect AWS account ID"""
    try:
        result = subprocess.run(
            ['aws', 'sts', 'get-caller-identity', '--query', 'Account', '--output', 'text'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_hyperpod_health_monitoring_account_id(region):
    """Get the AWS account ID for hyperpod-health-monitoring-agent based on region"""
    # Regional account ID mapping from sagemaker-hyperpod-cli/helm_chart/readme.md
    region_account_mapping = {
        'us-east-1': '767398015722',
        'us-west-2': '905418368575', 
        'us-east-2': '851725546812',
        'us-west-1': '011528288828',
        'eu-central-1': '211125453373',
        'eu-north-1': '654654141839',
        'eu-west-1': '533267293120',
        'eu-west-2': '011528288831',
        'ap-northeast-1': '533267052152',
        'ap-south-1': '011528288864',
        'ap-southeast-1': '905418428165',
        'ap-southeast-2': '851725636348',
        'sa-east-1': '025066253954'
    }
    
    return region_account_mapping.get(region, '905418368575')  # Default to us-west-2 if region not found

def parse_ecr_config(config_file):
    """Parse ECR configuration file and extract image mappings"""
    if not os.path.exists(config_file):
        print_colored(f"Error: ECR config file not found: {config_file}", Colors.RED)
        sys.exit(1)
    
    images = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Extract repository and tag
                if ':' in value:
                    repo, tag = value.rsplit(':', 1)
                    images[key] = {'repo': repo, 'tag': tag, 'full': value}
    
    return images

def update_ecr_urls(images, region, account_id):
    """Update repository URLs to use target ECR account and region"""
    ecr_pattern = re.compile(r'\d{12}\.dkr\.ecr\.[^.]+\.amazonaws\.com')
    
    for key, image_info in images.items():
        repo = image_info['repo']
        tag = image_info['tag']
        
        # Skip images that are handled separately (use AWS regional ECR)
        if key in ['aws-efa-k8s-device-plugin', 'hyperpod-health-monitoring-agent']:
            continue
        
        # Convert all images to ECR format
        if ecr_pattern.search(repo):
            # Already an ECR URL, update to target account/region
            # Extract the image name from the ECR URL
            ecr_parts = repo.split('/')
            if len(ecr_parts) >= 2:
                image_path = '/'.join(ecr_parts[1:])  # Everything after the ECR domain
                # Special case for AWS EKS images - remove 'eks/' prefix
                if image_path.startswith('eks/'):
                    image_name = image_path[4:]  # Remove 'eks/' prefix
                else:
                    image_name = image_path
            else:
                image_name = ecr_parts[-1]
            new_repo = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}"
        else:
            # Convert non-ECR URL to ECR format
            # Extract the image name from the original repository
            if '/' in repo:
                # For repos like "mpioperator/mpi-operator" or "nvcr.io/nvidia/k8s-device-plugin"
                image_name = repo.split('/')[-1]  # Get the last part
                if repo.startswith('nvcr.io/nvidia/'):
                    # Special case for NVIDIA images
                    image_name = f"nvidia-{image_name}"
                elif repo.startswith('mpioperator/'):
                    # Special case for MPI operator
                    image_name = "mpi-operator"
                elif repo.startswith('kubeflow/'):
                    # Special case for Kubeflow images
                    image_name = f"kubeflow-{image_name}"
            else:
                image_name = repo
            
            # Create ECR repository name
            new_repo = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}"
        
        images[key]['repo'] = new_repo
        images[key]['full'] = f"{new_repo}:{tag}"
    
    return images

def backup_file(file_path):
    """Create backup of file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup.{timestamp}"
    
    if os.path.exists(file_path):
        import shutil
        shutil.copy2(file_path, backup_path)
        print_colored(f"✓ Backed up {file_path}", Colors.GREEN)
        return backup_path
    return None

def insert_image_config_in_section(lines, section_name, config_lines, comment):
    """Insert image configuration into existing YAML section with proper indentation"""
    new_lines = []
    in_section = False
    section_found = False
    config_inserted = False
    section_indent = ""
    skip_until_next_section = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the target section
        if line.strip() == f"{section_name}:":
            new_lines.append(line)
            in_section = True
            section_found = True
            skip_until_next_section = False
            # Capture the indentation level of the section
            section_indent = line[:len(line) - len(line.lstrip())]
            i += 1
            continue
        
        # If we're in the target section
        if in_section and not skip_until_next_section:
            # Check if we're leaving the section (next top-level key or end of file)
            current_indent = line[:len(line) - len(line.lstrip())]
            
            # If this line has same or less indentation than section and is not empty/comment
            if (line.strip() and 
                not line.strip().startswith('#') and 
                len(current_indent) <= len(section_indent)):
                
                # Insert config before leaving the section if not already done
                if not config_inserted:
                    new_lines.append(f"{section_indent}  # {comment}")
                    for config_line in config_lines:
                        new_lines.append(f"{section_indent}  {config_line}")
                    config_inserted = True
                in_section = False
            
            # Check if this line is our ECR comment - skip duplicates but mark as inserted
            elif line.strip() == f"# {comment}":
                if not config_inserted:
                    new_lines.append(line)
                    # Add config lines with proper indentation
                    for config_line in config_lines:
                        new_lines.append(f"{section_indent}  {config_line}")
                    config_inserted = True
                    skip_until_next_section = True
                i += 1
                continue
            
            # Check if this line contains image config we want to replace
            elif (line.strip().startswith(('image:', 'mpiOperator:', 'hmaimage:')) and 
                  len(current_indent) == len(section_indent) + 2):
                
                if not config_inserted:
                    # Add comment and config
                    new_lines.append(f"{section_indent}  # {comment}")
                    for config_line in config_lines:
                        new_lines.append(f"{section_indent}  {config_line}")
                    config_inserted = True
                    skip_until_next_section = True
                i += 1
                continue
        
        # If we're skipping until next section, check if we've reached it
        if skip_until_next_section:
            current_indent = line[:len(line) - len(line.lstrip())]
            if (line.strip() and 
                not line.strip().startswith('#') and 
                len(current_indent) <= len(section_indent)):
                skip_until_next_section = False
                in_section = False
        
        # Only add the line if we're not skipping
        if not skip_until_next_section:
            new_lines.append(line)
        
        i += 1
    
    # If we were still in the section at the end, add config
    if in_section and not config_inserted:
        new_lines.append(f"{section_indent}  # {comment}")
        for config_line in config_lines:
            new_lines.append(f"{section_indent}  {config_line}")
    
    # If section wasn't found, add it at the end
    if not section_found:
        new_lines.append("")
        new_lines.append(f"# Section added by update-values-with-ecr.py")
        new_lines.append(f"{section_name}:")
        new_lines.append(f"  # {comment}")
        for config_line in config_lines:
            new_lines.append(f"  {config_line}")
    
    return new_lines

def update_values_yaml(values_file, images, region):
    """Update the values.yaml file with ECR image overrides"""
    if not os.path.exists(values_file):
        print_colored(f"Error: Values file not found: {values_file}", Colors.RED)
        return False
    
    # Read the file as text to preserve formatting
    with open(values_file, 'r') as f:
        lines = f.readlines()
    
    # Remove trailing newlines from lines for easier processing
    lines = [line.rstrip('\n') for line in lines]
    
    updated_images = []
    
    # Update NVIDIA device plugin
    if 'nvidia-k8s-device-plugin' in images:
        image_info = images['nvidia-k8s-device-plugin']
        config_lines = [
            "image:",
            "  repository: " + image_info['repo']
        ]
        lines = insert_image_config_in_section(
            lines, 
            "nvidia-device-plugin", 
            config_lines,
            "ECR override for air-gapped environment"
        )
        updated_images.append(f"NVIDIA Device Plugin: {image_info['full']}")
        print_colored(f"✓ Added NVIDIA device plugin override: {image_info['full']}", Colors.GREEN)
    
    # Update AWS EFA device plugin - use regional AWS ECR (available in all regions)
    aws_efa_repo = f"602401143452.dkr.ecr.{region}.amazonaws.com/eks/aws-efa-k8s-device-plugin"
    config_lines = [
        "image:",
        "  repository: " + aws_efa_repo
    ]
    lines = insert_image_config_in_section(
        lines, 
        "aws-efa-k8s-device-plugin", 
        config_lines,
        "Regional AWS ECR (available in all regions)"
    )
    updated_images.append(f"AWS EFA Device Plugin: {aws_efa_repo}:v0.5.6")
    print_colored(f"✓ Added AWS EFA device plugin override: {aws_efa_repo}:v0.5.6", Colors.GREEN)
    
    # Update MPI operator
    if 'mpi-operator' in images:
        image_info = images['mpi-operator']
        config_lines = [
            "mpiOperator:",
            "  image:",
            "    repository: " + image_info['repo']
        ]
        lines = insert_image_config_in_section(
            lines, 
            "mpi-operator", 
            config_lines,
            "ECR override for air-gapped environment"
        )
        updated_images.append(f"MPI Operator: {image_info['full']}")
        print_colored(f"✓ Added MPI operator override: {image_info['full']}", Colors.GREEN)
    
    # Update health monitoring agent - use regional AWS ECR (available in all regions with regional account IDs)
    hma_account_id = get_hyperpod_health_monitoring_account_id(region)
    hma_repo = f"{hma_account_id}.dkr.ecr.{region}.amazonaws.com/hyperpod-health-monitoring-agent"
    hma_image = f"{hma_repo}:1.0.1249.0_1.0.359.0"
    config_lines = [
        f'hmaimage: "{hma_image}"'
    ]
    lines = insert_image_config_in_section(
        lines, 
        "health-monitoring-agent", 
        config_lines,
        "Regional AWS ECR (available in all regions with regional account IDs)"
    )
    updated_images.append(f"Health Monitoring Agent: {hma_image}")
    print_colored(f"✓ Added health monitoring agent override: {hma_image}", Colors.GREEN)
    
    # Update kubeflow training operator
    if 'kubeflow-training-operator' in images:
        image_info = images['kubeflow-training-operator']
        config_lines = [
            "image:",
            "  repository: " + image_info['repo']
        ]
        lines = insert_image_config_in_section(
            lines, 
            "training-operators", 
            config_lines,
            "ECR override for air-gapped environment"
        )
        updated_images.append(f"Kubeflow Training Operator: {image_info['full']}")
        print_colored(f"✓ Added kubeflow training operator override: {image_info['full']}", Colors.GREEN)
    
    # Write the updated content back to file
    with open(values_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    print_colored("✓ Updated main values.yaml file with ECR image overrides", Colors.GREEN)
    return updated_images

def show_summary(updated_images, region, account_id):
    """Show summary of updates"""
    print()
    print_colored("=== Update Summary ===", Colors.BLUE)
    print_colored("✓ Updated Helm values.yaml files with ECR image references", Colors.GREEN)
    print()
    
    if updated_images:
        print("Updated images:")
        for image in updated_images:
            print(f"  • {image}")
    else:
        print("No images were updated.")
    
    print()
    print("Target ECR configuration:")
    print(f"  • Region: {region}")
    print(f"  • Account ID: {account_id}")
    print()
    print_colored("Note: Image overrides have been added to the top-level values.yaml file.", Colors.YELLOW)
    print("      These overrides will be used by all subcharts during deployment.")
    print()
    print_colored("Next steps:", Colors.BLUE)
    print("  1. Review the updated values.yaml files")
    print("  2. Run 'helm dependency update' to update external chart dependencies")
    print("  3. Deploy with 'helm install' or 'helm upgrade'")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Update Helm values.yaml files with ECR image references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use defaults (us-west-2, auto-detect account)
  %(prog)s us-east-1                 # Use us-east-1, auto-detect account
  %(prog)s us-east-1 123456789012    # Use us-east-1 and specific account ID

Configuration:
  ECR images are defined in: tools/ecr-images.conf
  Values file updated:
    • sagemaker-hyperpod-cli/helm_chart/HyperPodHelmChart/values.yaml
        """
    )
    
    parser.add_argument(
        'region',
        nargs='?',
        default='us-east-2',
        help='AWS region for ECR repositories (default: us-west-2)'
    )
    
    parser.add_argument(
        'account_id',
        nargs='?',
        default='auto',
        help='AWS account ID for ECR repositories (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    ecr_config_file = "tools/ecr-images.conf"
    main_values_file = "sagemaker-hyperpod-cli/helm_chart/HyperPodHelmChart/values.yaml"
    
    print_colored("Updating Helm values.yaml files with ECR image references...", Colors.BLUE)
    print(f"Region: {args.region}")
    print(f"Account ID: {args.account_id}")
    print()
    
    # Get account ID if set to auto
    if args.account_id == 'auto':
        print_colored("Auto-detecting AWS account ID...", Colors.YELLOW)
        detected_account_id = get_account_id()
        if detected_account_id:
            print(f"Detected account ID: {detected_account_id}")
            args.account_id = detected_account_id
        else:
            print_colored("Error: Could not auto-detect account ID. Please provide it as second argument.", Colors.RED)
            sys.exit(1)
    
    # Parse ECR configuration
    print_colored("Parsing ECR configuration...", Colors.BLUE)
    images = parse_ecr_config(ecr_config_file)
    
    # Update ECR URLs for target account and region
    print_colored("Updating ECR URLs for target account and region...", Colors.BLUE)
    images = update_ecr_urls(images, args.region, args.account_id)
    
    # Update main values.yaml file
    print_colored("Updating main values.yaml file with ECR image overrides...", Colors.BLUE)
    backup_file(main_values_file)
    updated_images = update_values_yaml(main_values_file, images, args.region)
    
    # Show summary
    show_summary(updated_images, args.region, args.account_id)

if __name__ == "__main__":
    main()