# Parameters

Every parameter of the five templates in [`../assets`](../assets). The root template
(`eks-gpu-cluster-deploy-all.yaml`) exposes the ones an operator chooses and passes the rest between
the child stacks.

## Root — `eks-gpu-cluster-deploy-all.yaml`

### Network

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrimarySubnetAZ` | AZ name | required | Zone of the public subnet, the node subnet, the NAT gateway and every GPU node. Must be the zone of the capacity reservation when one is used: EFA traffic does not cross zones |
| `SecondarySubnetAZ` | AZ name | required | Zone of the second private subnet, which exists only because EKS requires subnets in two zones. No nodes run there. Must differ from `PrimarySubnetAZ` (asserted at submit time) |
| `VpcCidr` | String | `10.0.0.0/16` | `/16` or `/17`. Split into three /20 subnets: public, node, control plane |

### Cluster

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `KubernetesVersion` | String | `1.36` | Control plane version and the AL2023 NVIDIA AMI release. Any `1.xx`; the `kubectl` the bootstrap downloads is its own parameter, because it has to stay within one minor of this and a table of the pairs would be versions to maintain here |
| `KubectlVersion` | String | `1.36.4` | `kubectl` the GPU node group's bootstrap downloads, passed through to `eks-add-gpu-nodegroup.yaml`. Within one minor of `KubernetesVersion`, so a cluster on another version needs this set to match |
| `HelmVersion`, `NvidiaDevicePluginChartVersion`, `EfaDevicePluginChartVersion` | String | `3.19.0`, `0.20.0`, `v0.5.32` | Passed through to `eks-add-gpu-nodegroup.yaml`; see its rows. Raising a chart version on a live cluster starts with Helm (README section 10) |
| `SystemInstanceType` | String | `m7i.xlarge` | Instance type of the two system nodes. They carry CoreDNS and the node-feature-discovery master, which cannot run on a tainted GPU node. The default is the newest generation offered in every Region the GPU types appear in: a type the Region does not offer fails the node group with `Unsupported - The requested configuration is currently not supported`, which names neither the type nor the Region. Check with `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=$INSTANCE_TYPE Name=location,Values=$AZ` |
| `ServiceIpv4Cidr` | String | `172.20.0.0/16` | CIDR the cluster allocates Service addresses from. Must not overlap `VpcCidr`. An input rather than a value left to EKS because a node group naming its own AMI has to repeat the value in its bootstrap configuration, and the value EKS picks on its own cannot be read back: `ServiceIpv6Cidr` is a readable cluster attribute and `ServiceIpv4Cidr` is not |
| `AdminRoleArn` | String | empty | An extra IAM principal that receives `AmazonEKSClusterAdminPolicy`. The principal that creates the stack always has it, so this is for the case where one principal provisions and another uses the cluster |

### GPU capacity

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `GpuInstanceType` | String | `g7e.12xlarge` | Instance type of the GPU node group, and through the `NicLayout` mapping the whole interface layout. See README section 3 for which types have been launched |
| `AmiType` | String | `AL2023_x86_64_NVIDIA` | EKS AMI type for the GPU nodes, used when no image input is given. Not an enumeration, so a type EKS adds later needs no template change. It has to be a type whose bootstrap is nodeadm, as the AL2023 family is: the GPU launch template's user data is a nodeadm `NodeConfig` |
| `SystemAmiType` | String | `AL2023_x86_64_STANDARD` | EKS AMI type for the system nodes. Not an enumeration either, and with no launch template and no user data on that node group, any type EKS validates works |
| `NodeAmiId` | String | empty | Node AMI for the GPU nodes. Leave the `NodeImage` inputs empty when using it. Any source: `awslabs/amazon-eks-ami`, EC2 Image Builder, or your own pipeline. It has to carry `nodeadm`, a driver that enumerates the instance type's GPUs, and the NVIDIA container toolkit |
| `NodeImagePackages` | String | empty | Comma-separated packages to build into the node image, each pinned to a version. Setting it builds an image and boots the GPU nodes from it. The build does not interpret the packages |
| `NodeImageRepoPackages` | String | empty | Refused without `NodeImagePackages`. Comma-separated packages whose job is to make the others resolvable, such as a vendor's `-release` package. Installed one at a time before anything else, and before the metadata refresh. A package that enables a repository has to be in place before a name from it can be resolved, which is what keeps install order out of `NodeImagePackages` |
| `NodeImageRepoFiles` | String | empty | Refused without `NodeImagePackages`. Comma-separated URLs of repository definitions the packages need |
| `NodeImageRepoKeys` | String | empty | Refused without `NodeImagePackages`. Comma-separated URLs of signing keys to import before installing |
| `NodeImageAssertPaths` | String | empty | Comma-separated paths the build requires to exist before publishing the image. Required with `NodeImagePackages`: a build with nothing to assert publishes an image whose contents were never checked |
| `NodeImageAssertCommands` | String | empty | Refused without `NodeImagePackages`. Semicolon-separated commands the build runs on the produced image, each of which has to exit zero. `NodeImageAssertPaths` proves a file arrived; these prove it works, which is a different claim, because a kernel module can be present as a file and fail to load |
| `NodeImageRecipeArn` | String | empty | An existing EC2 Image Builder recipe to build. Use it when you already maintain one: the stack contributes the build environment and nothing about the contents |
| `NodeImageBuildInstanceProfile` | String | empty | Name of the instance profile the image build runs with, not its ARN. Empty creates one whose role carries `EC2InstanceProfileForImageBuilder` and `AmazonSSMManagedInstanceCore`, which is what a build that authenticates for nothing it fetches needs. Supply one when the payload comes from a private bucket or registry: the permissions are yours to declare, and where a resource policy in another account names the role, that name has to exist before the build. Its role has to grant what Image Builder, Systems Manager and the payload's source require |
| `NodeImageVersion` | String | `1.0.0` | Semantic version of both the component and the recipe the build composes. Image Builder resources are immutable per version, and the recipe carries the payload as parameter values, so raise this whenever the `NodeImage` inputs change; reusing a version with different contents is rejected |
| `NodeImageBuildInstanceType` | String | `m5.large` | Instance type that builds the image. The build does not need a GPU, so it is deliberately not a GPU type; the node-side check is what proves the image drives its GPUs |
| `GpuNodeCount` | Number | `2` | Minimum, desired and maximum of the GPU node group, all the same value. A prefill/decode split needs at least 2. `0` creates the cluster and installs the device plugins with no GPU capacity, for testing template changes; a managed node group rejects a maximum of 0, so that case asks for 0 out of 1 |
| `GpuRootVolumeSize` | Number | `300` | Root EBS volume in GiB. Inference images are large, and they land on the root volume unless containerd is pointed at the NVMe volume |
| `CapacityReservationId` | String | empty | A targeted On-Demand Capacity Reservation or a Capacity Block. Empty launches On-Demand and consumes an open reservation whose attributes match |
| `CapacityReservationType` | String | `targeted-odcr` | `targeted-odcr` targets the reservation. `capacity-block` sets `MarketType=capacity-block` and `CapacityType=CAPACITY_BLOCK`. With either, the stack creates no cluster placement group, because the reserved capacity is not inside one it creates; without a reservation the nodes go into a new one. `capacity-block` with an empty id is rejected at submit time |

### Optional

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrePullImage` | String | empty | An image pulled onto every GPU node by a DaemonSet, after the nodes are verified. The pull is started and not waited for: a multi-gigabyte pull must not be able to roll back a cluster. Watch it with `kubectl rollout status daemonset/prepull-$NODE_GROUP_NAME -n kube-system`, where `NODE_GROUP_NAME` is the node group name in lowercase with `_` as `-`. With `GpuNodeCount=0` there is no node to pull onto, and the pull starts once the count is raised |
| `DeployFsxLustre` | String | `false` | `true` creates an FSx for Lustre filesystem and installs the `aws-fsx-csi-driver` add-on. Off by default because the GPU types carry local NVMe. The `FsxFileSystemId`, `FsxDnsName` and `FsxMountName` outputs are what a static `PersistentVolume` binds to |
| `FsxStorageCapacity` | Number | `1200` | Filesystem size in GiB, as an enumeration rather than a minimum: FSx accepts only certain sizes and refuses the rest minutes into the deploy. Add a size to the template if you need one the list does not cover |

### Template location

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `S3BucketName` | String | `awsome-distributed-ai` | Bucket the child templates are fetched from |
| `S3KeyPrefix` | String | `templates/amazon-eks/` | Key prefix of the child templates, trailing slash included |

Override both to deploy a copy that is not published yet, or to serve the templates from a bucket you
control.

## `eks-cluster-prerequisites.yaml`

`PrimarySubnetAZ`, `SecondarySubnetAZ`, `VpcCidr`, `DeployFsxLustre`, `FsxStorageCapacity` — same
meaning as above.

## `eks-cluster.yaml`

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `ClusterName` | String | required | Name of the cluster. The root passes its own stack name |
| `KubernetesVersion`, `AdminRoleArn`, `SystemInstanceType`, `SystemAmiType` | | | As above |
| `SystemNodeCount` | Number | `2` | Number of nodes carrying CoreDNS and the node-feature-discovery master, neither of which can run on a tainted GPU node. Not exposed by the root template. Two so that replacing one node leaves the other serving; it does not spread CoreDNS, whose anti-affinity is a preference |
| `ServiceIpv4Cidr` | String | `172.20.0.0/16` | As above |
| `PrivateSubnetId`, `ControlPlaneSubnetId`, `NodeSecurityGroupId` | ids | required | Outputs of the prerequisites stack |
| `FsxFileSystemId` | String | empty | A non-empty value installs the FSx CSI driver add-on |

Custom-AMI outputs: `ClusterEndpoint`, `ClusterCertificateAuthority` and
`ClusterServiceCidr`. A node group that names its own AMI needs all three, because EKS merges no
bootstrap user data once a launch template carries an `ImageId`.

## `eks-add-gpu-nodegroup.yaml`

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `ClusterName` | String | required | Cluster the node group joins |
| `KubernetesVersion` | String | `1.36` | The cluster's version. The AMI EKS resolves follows the cluster rather than this value; changing it, like changing `KubectlVersion`, re-runs the bootstrap and its verification. `KubectlVersion` is separate |
| `PrivateSubnetId` | id | required | Subnet for the GPU nodes, in the zone of the reservation |
| `NodeSecurityGroupId` | id | required | A security group that allows all traffic between its own members, which is EFA's requirement |
| `ClusterSecurityGroupId` | id | required | The cluster's own security group. EKS stops attaching it once the launch template names any security group, and a node without it never joins |
| `NodeGroupName` | String | `gpu` | Name of the managed node group. Change it to add a second GPU node group to a cluster that already has one; the nodes carry the label `role=gpu` either way |
| `AmiType` | String | `AL2023_x86_64_NVIDIA` | As above. Ignored when `NodeAmiId` is set, because that switches the node group to `CUSTOM` |
| `NodeRoleArn` | String | empty | Node IAM role. Empty creates one with the four managed policies a GPU node needs |
| `BootstrapVpcId` | String | empty | VPC of `PrivateSubnetId`. Empty runs the bootstrap's CodeBuild build outside the VPC, which needs the cluster API reachable from the internet. Set, the build runs in `PrivateSubnetId` with `ClusterSecurityGroupId`, the group EKS puts on the private endpoint's network interfaces, and uses the subnet's route to the internet for its downloads and its answer to CloudFormation. A parameter for the VPC alone rather than for a subnet and a security group, because the node subnet and the cluster security group already reach the endpoint |
| `NodeAmiId` | String | empty | As above. Setting it switches the node group to `AmiType: CUSTOM`, which is why the three cluster values below are then required |
| `KubectlVersion` | String | `1.36.4` | `kubectl` the bootstrap downloads. Has to stay within one minor version of the cluster, so a cluster on another version needs this set to match. A parameter rather than a table keyed by `KubernetesVersion`, because such a table is a set of versions to maintain here for something a caller can read off their own cluster |
| `HelmVersion` | String | `3.19.0` | Helm the bootstrap downloads to install the device plugins |
| `NvidiaDevicePluginChartVersion` | String | `0.20.0` | Chart version of `nvidia-device-plugin`. Pinned rather than resolved at install time, because the plugin's correctness depends on the driver on the node and nothing outside this repository states which chart version goes with which driver. One release serves the whole cluster, so every stack sharing a cluster passes the same value |
| `EfaDevicePluginChartVersion` | String | `v0.5.32` | Chart version of `aws-efa-k8s-device-plugin`. Pinned for the same reason |
| `GpuPciVendorId` | String | `0x10de` | PCI vendor id of the accelerators, read only by the diagnosis that runs when a node comes up without them |
| `ClusterEndpoint`, `ClusterCertificateAuthority`, `ClusterServiceCidr` | String | empty | Required with `NodeAmiId`, and rejected as a set at submit time when one is missing. Read them from `aws eks describe-cluster` |
| `GpuInstanceType`, `GpuNodeCount`, `GpuRootVolumeSize`, `CapacityReservationId`, `CapacityReservationType`, `PrePullImage` | | | As above |

## `eks-gpu-node-ami.yaml`

Deployed by the root when `NodeImagePackages` or `NodeImageRecipeArn` is set, or on its own. Either
way exactly one of `Packages` and `RecipeArn` is required, and `Packages` requires `AssertPaths`,
both asserted at submit time.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `VpcId`, `PrivateSubnetId` | Id | required | Where the build instance runs. The subnet needs outbound internet access for the package repositories |
| `BuildInstanceType` | String | `m5.large` | The build needs no GPU |
| `KubernetesVersion` | String | `1.36` | Selects the EKS-optimised AL2023 **standard** parent AMI |
| `RepoPackages` | String | empty | Packages that make the others resolvable, installed first, one at a time |
| `Packages` | String | empty | Comma-separated packages, each pinned to a version, installed in one transaction |
| `RepoFiles`, `RepoKeys` | String | empty | Comma-separated URLs of repository definitions and signing keys, reachable from the build subnet |
| `AssertPaths` | String | empty | Comma-separated paths the produced image must contain |
| `AssertCommands` | String | empty | Semicolon-separated commands that must exit zero on the produced image |
| `RecipeArn` | String | empty | An existing EC2 Image Builder recipe to build instead of the one this template composes. Its parent image needs the Systems Manager agent, which Image Builder uses to reach the build instance |
| `BuildInstanceProfileName` | String | empty | Name of the build instance's instance profile. Empty creates a role carrying the two managed policies Image Builder needs with a profile around it. A build fetching from a private bucket or registry supplies its own, because the permissions are specific to that payload and, where a resource policy names the role, a role this template creates cannot be granted before it exists |
| `ComponentVersion`, `RecipeVersion` | String | `1.0.0` | Image Builder resources are immutable per version. The recipe carries the inputs above as component parameter values, so changing the payload changes the recipe and needs `RecipeVersion` raised; the root passes one value to both |
