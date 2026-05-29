# AWS PCS Architecture Components for Diagram

## Main Components (Required for Diagram)

### Network Layer
- **VPC** with dual CIDR blocks (10.0.0.0/16 + 10.1.0.0/16)
- **Public Subnet** (NAT Gateway placement)
- **Private Subnet** (compute nodes, FSx placement)
- **S3 VPC Endpoint** (gateway endpoint for template/data access)

### Storage Layer
- **FSx for Lustre** (scratch storage, high-speed I/O for ML training)
  - PERSISTENT_2 deployment type
  - Configurable throughput (125-1000 MB/s/TiB)
  - LZ4 compression
- **FSx for OpenZFS** (home directories, persistent user data)
  - Single-AZ HA deployment
  - NFS exports with no_root_squash
  - Automatic daily backups

### Compute Layer
- **AWS PCS Cluster** (Slurm scheduler)
  - Head node (managed service, not visible)
  - Slurm versions: 25.05, 25.11
- **Login Node Group** (public subnet)
  - SSH/SSM access point
  - Container tooling (Enroot/Pyxis)
  - Monitoring stack (Prometheus, Grafana)
- **On-Demand Compute Node Group** (private subnet)
  - CPU-based workloads
  - Auto-scaling with Slurm
  - Optional Enroot/Pyxis via UserData
- **GPU Compute Node Group - P5/P6** (private subnet)
  - 32x EFA network interfaces
  - Multi-node ML training
  - Optional Enroot/Pyxis via UserData
  - Dynamic scaling (MinCount=0)

### Optional Components
- **EC2 Image Builder** (custom AMI with Enroot/Pyxis pre-installed)
  - Scheduled builds (manual, weekly, monthly)
  - Published to SSM Parameter Store
- **Monitoring Stack** (on login node)
  - Prometheus (metrics collection)
  - Grafana (visualization)
  - DCGM Exporter (GPU metrics)
  - Slurm OpenMetrics

## Key Architectural Features

### Dual Deployment Modes
1. **Custom AMI Mode** (Production)
   - ImageBuilder pre-installs Enroot/Pyxis (~20-30 min build)
   - Fast node boot (~3 minutes)
   - Best for production clusters

2. **UserData Installation Mode** (Testing)
   - Enroot/Pyxis installed on first boot (~8-12 minutes)
   - No AMI build required
   - Best for rapid iteration and testing

### Prerequisites Stack Reuse
- Network and storage resources can be deployed once (pcs-shared-infra)
- Multiple clusters reference shared infrastructure via CloudFormation Exports/Imports
- Reduces deployment time from 20-30 minutes to 5-10 minutes for subsequent clusters

### Dual Tagging Strategy
- **ClusterName tag**: User-friendly name (CloudFormation stack name)
- **pcs-cluster-id tag**: Actual PCS cluster ID (e.g., pcs_i3bddqwdrp)
- Both applied to all instances (login nodes, compute nodes)
- Used by monitoring stack for IMDS metadata and SSM parameter paths

## Data Flow

1. **User Access**:
   - Users → SSH/SSM → Login Node (public subnet)

2. **Job Submission**:
   - Login Node → Slurm Scheduler (PCS managed)
   - Scheduler → Compute Nodes (private subnet)

3. **Storage Access**:
   - Compute Nodes → FSx for Lustre (scratch, training data)
   - Compute Nodes → FSx for OpenZFS (home directories)
   - Compute Nodes → S3 (via VPC endpoint, checkpoints/results)

4. **Container Workflow**:
   - Users → Login Node (Enroot/Pyxis commands)
   - Slurm jobs → Pull container images (Docker Hub, NVIDIA NGC)
   - Compute Nodes → Run containerized training jobs

5. **Monitoring**:
   - Compute Nodes → Prometheus (metrics push)
   - Users → Grafana dashboard (via Login Node)
   - Slurm → OpenMetrics exporter → Prometheus

## Security Features

- **Network Isolation**: Compute nodes in private subnet, NAT Gateway for outbound only
- **Security Group**: All-to-all EFA communication within cluster
- **IAM Roles**: Integrated instance profiles for S3/SSM access
- **SSM Session Manager**: SSH alternative for login node access

## Diagram Layout Recommendations

- **Top Layer**: User access (SSH/SSM) → Login Node
- **Middle Layer**: Slurm scheduler (PCS managed, abstract as service icon)
- **Compute Layer**: On-Demand + GPU node groups (separate swim lanes)
- **Storage Layer**: FSx for Lustre + OpenZFS (bottom or side panel)
- **S3 VPC Endpoint**: Connection from private subnet to S3 service
- **Optional Panel**: ImageBuilder + Monitoring stack (dashed border)

## Components to Exclude from Diagram

- Internet Gateway (generic networking)
- NAT Gateway (generic networking)
- Route Tables (generic networking)
- Elastic IP (generic networking)
- Security Group rules (detail level)
- IAM roles/policies (non-visual)
- CloudFormation stacks (deployment mechanism)
- SSM parameters (configuration storage)
