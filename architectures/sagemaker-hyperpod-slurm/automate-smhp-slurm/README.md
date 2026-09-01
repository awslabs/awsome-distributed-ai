# SageMaker Hyperpod Cluster Automation Script

This project provides a script to automate the creation and setup of a SageMaker Hyperpod cluster with SLURM integration.

The automation script streamlines the process of setting up a distributed training environment using AWS SageMaker Hyperpod.
It handles the installation and configuration of necessary tools, clones the required repository, sets up environment variables, and configures lifecycle scripts for the SageMaker Hyperpod architecture.

## Demo

![SageMaker Hyperpod Cluster Automation Demo](/architectures/sagemaker-hyperpod-slurm/automate-smhp-slurm/media/automate-smhp-demo.gif)

This demo gif showcases the step-by-step process of creating and setting up a SageMaker Hyperpod cluster using our automation script.

- `automate-cluster-creation.sh`: The main script that automates the cluster creation process.
- `README.md`: This file, providing information about the project.

## Usage Instructions

### Prerequisites

- AWS CLI (version 2.17.1 or higher)
- Git
- Bash shell environment
- AWS account with appropriate permissions

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/awslabs/awsome-distributed-ai.git
   cd architectures/sagemaker-hyperpod-slurm/automate-smhp-slurm
   ```

2. Make the script executable:
   ```bash
   chmod +x automate-cluster-creation.sh
   ```

### Running the Script

Execute the script:

```bash
./automate-cluster-creation.sh
```

The script will guide you through the following steps:

1. Check and install/update AWS CLI if necessary.
2. Verify Git installation.
3. Clone the "awsome-distributed-ai" repository.
4. Set up environment variables.
5. Configure lifecycle scripts for SageMaker Hyperpod.

### Configuration

During execution, you'll be prompted to provide the following information:

- Name of the SageMaker VPC CloudFormation stack (default: sagemaker-hyperpod)
- Confirmation if you deployed the optional hyperpod-observability CloudFormation stack
- Instance group configuration (group name, instance type, instance count, etc)

### Troubleshooting

- If you encounter permission issues when attaching IAM policies, the script will provide options to:
  1. Run `aws configure` as an admin user within the script.
  2. Exit the script to run `aws configure` manually.
  3. Continue without configuring this step.

- If environment variable generation fails:
  1. You can choose to continue with the rest of the script (not recommended unless you know how to set the variables manually).
  2. Exit the script to troubleshoot the issue.

## Data Flow

The automation script follows this general flow:

1. Check and setup prerequisites (AWS CLI, Git)
2. Clone necessary repositories
3. Set up environment variables
4. Configure lifecycle scripts
5. Enable observability (if applicable)
6. Attach IAM policies (if applicable)
7. Cluster Configuration
8. Create cluster

```
[Prerequisites] -> [Clone Repos] -> [Setup Env Vars] -> [Configure LCS] -> [Enable Observability] -> [Attach IAM Policies] -> [Create Cluster configuration] -> [Create cluster]
```

Important technical considerations:
- Ensure you have the necessary AWS permissions before running the script.
- The script modifies the `config.py` file to enable observability if selected.
- IAM policy attachment requires admin permissions.
