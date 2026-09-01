# Amazon Machine Images for Self-Managed ML Workloads

This package contains a Packer script to build Amazon Machine Images for self-managed Ml training and inference. The images can be built for difference AWS ParallelCluster, EKS), platforms (CPU, GPU, Neuron) for training and inference workloads.

### Initial setup

To build images you will need:
- **GNU Make**: install it via `yum` or `apt` if using Linux, via [`brew`](https://formulae.brew.sh/formula/make) if using OSX or [Chocolatey](https://community.chocolatey.org/packages/make) on MS Windows.
- **Packer**: it can be downloaded via [Hashicorp](https://www.packer.io/)'s website, you can also use [`brew`](https://formulae.brew.sh/formula/packer#default) on OSX.
- **Ansible**: get it via your package manager, we recommend via [`brew`](https://formulae.brew.sh/formula/ansible#default) if using OSX.

Initialize Packer (will install plugins):
```bash
packer init -upgrade packer-ami.pkr.hcl 
```

### Build a custom AMI

Assuming that GNU Make, Packer and Ansible installed, you can build AMIs by typing `make` in your terminal with an argument corresponding to the desired AMI you want to build.

Here is an example to build a AMI for training or inference on GPU with AWS ParallelCluster:

```bash
make ami_pcluster_gpu
```

> **Note**: If you encounter an error because Packer could not find the source AMI with the error `InvalidAMIID.NotFound` then prepend by `AWS_REGION` with the target region. For example, `AWS_REGION=us-east-2 make ami_pcluster_gpu`. By default the script use `us-west-2`.

The list of arguments you can use is shown in the table below with the AMI origin (what are we starting our custom AMI from) and notes regarding their content.

| Argument           | Source AMI | Notes                                                                              |
|--------------------|------------|------------------------------------------------------------------------------------|
| `ami_pcluster_cpu` | [ParallelCluster AMI](https://docs.aws.amazon.com/parallelcluster/latest/ug/pcluster.list-official-images-v3.html) | Creates a custom ParallelCluter AMI for CPU based workloads                        |
| `ami_pcluster_gpu` | [ParallelCluster AMI](https://docs.aws.amazon.com/parallelcluster/latest/ug/pcluster.list-official-images-v3.html) | Creates a custom ParallelCluter AMI for GPU based workloads, training or inference |
| `ami_pcluster_neuron` | [ParallelCluster AMI](https://docs.aws.amazon.com/parallelcluster/latest/ug/pcluster.list-official-images-v3.html) | Creates a custom ParallelCluter AMI for [Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) (Trn, Inf) based workloads, training or inference |
| `ami_base`         | [EC2 AL2 AMI](https://aws.amazon.com/amazon-linux-2/) | EC2 AMI with updates, Docker, Lustre, EFA, Pyxis and Enroot (everything)                        |
| `ami_dlami_gpu`    | [DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) | DL AMI with updated drivers, Pyxis, enroot, Lustre module client and Docker.       |
| `ami_dlami_neuron` | [DLAMI-Neuron](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) | DL AMI for Neuron, same as above without the Nvidia stack                          |
| `ami_eks_gpu`      | [EKS AMI](https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami) | EKS GPU AMI with Lustre, EFA                                                       |
| `ami`              |AMI dependent| Build all the images                                                               |


Once a build is launched, Packer will create an instance and install packages for a period of 10-25 minutes depending on how much software is installed.

### Software stack: Ansible roles

Each image is build using a base image and different Ansible roles used to install and configure the software stack installed on the AMI. The stack for each AMI is defined into *playbooks* files containing each a list of packages.

You will find below the list of images you can build and which roles are deployed in these. The `ami` argument will build all of these images.

| Ansible Roles         | `ami_pcluster_cpu` | `ami_pcluster_gpu`| `ami_base` | `ami_dlami_gpu` | `ami_dlami_neuron` | `ami_eks_gpu` |
|-----------------------|--------------------|-------------------|------------|-----------------|--------------------|---------------|
| `base`                |         ✅         |         ✅        |     ✅     |       ✅        |        ✅         |      ❌       |
| `packages`            |         ✅         |         ✅        |     ✅     |       ✅        |        ✅         |      ✅       |
| `aws_cliv2`           |         ✅         |         ✅        |     ✅     |       ✅        |        ✅         |      ✅       |
| `aws_lustre`          |         ✅         |         ✅        |     ✅     |       ✅        |        ✅         |      ✅       |
| `nvidia_enroot_pyxis` |         ✅         |         ✅        |     ✅     |       ✅        |        ✅         |      ❌       |
| `docker`              |         ✅         |         ✅        |     ✅     |       ✅        |        ❌         |      ❌       |
| `nvidia_docker`       |         ❌         |         ✅        |     ✅     |       ✅        |        ✅         |      ❌       |
| `nvidia_driver`       |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ✅       |
| `nvidia_cuda`         |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ❌       |
| `nvidia_gdrcopy`      |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ❌       |
| `nvidia_nccl`         |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ❌       |
| `aws_efa`             |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ❌       |
| `aws_efa_ofi`         |         ❌         |         ✅        |     ✅     |       ❌        |        ❌         |      ❌       |


## Customizing your AMIs

You can customize your AMIs by:
- Modifying existing roles to install specific software versions: for example a specific version of the EFA driver, Nvidia CUDA or Nvidia GPU driver.
- Add new roles to install or configure new software or libraries.

Before going further, let's see how we defined our Ansible Roles.

#### More on roles

Our Ansible roles consist of 3 components: `defaults`, `files` and `tasks`.

- `defaults`: contain default values for conditionals and versions of software being installed.
- `files`: hold files that will be copied to the custom AMI such as config files.
- `tasks`: is the list of tasks executed by Ansible to install and configure software.


#### Example

To illustrate that, here's an example using the Nvidia Driver. By looking at the structure below you will see the 3 role components: `defaults`, `files` and `tasks`

```
├── nvidia_driver
│   ├── defaults
│   │   └── main.yml
│   ├── files
│   │   └── nvidia-persistenced-override.service
│   └── tasks
│       └── main.yml
```

##### `defaults`

The defaults contain variables for the role and default values. In the case of the Nvidia driver we set the version to a default with `nvidia_driver_version` and if needed we can change it to a newer or older version. Then you will find two booleans that'll be used in the tasks as conditionals on whether to install the Nvidia Fabric Manager (required A100,H100) via `install_nvidia_fabric_manager` and allow a reboot after installing the driver using the variable `allow_reboot`.

```yaml
nvidia_driver_version: "535.54.03"
install_nvidia_fabric_manager: true
allow_reboot: true
```

##### `files`

In the case of the Nvidia driver we have 1 file in `files` named `nvidia-persistenced-override.service`. It is an SystemD service module that we use to force driver persistence. This file is copied to the custom AMI through one of the `tasks`.

##### `tasks`

The tasks are a list of instructions that Ansible will run through to deploy the role and will be based of Ansible default modules. Here's an excerpt of task below, feel free to open the original file to see the full list of tasks.

```yaml
- name: "Install additional dependencies"
  ansible.builtin.yum:
    name:
      - gcc10
      - kernel-devel
      - kernel-headers
      - dkms
    state: present

- name: "Blacklist nouveau"
  community.general.kernel_blacklist:
    name: nouveau
    state: present
```

### Modify the roles

As shared earlier, you can modify the roles and add new ones. Most users would modify the roles defaults to change the default versions of software being installed. If you need to modify the installation or configuration process you may want to modify the `tasks` file.

Alternatively, you can add a new role to install a new software component, ensure that you respect the structure used by other roles. Don't forget to list your role in the playbook you want to use, for example `playbook-eks-gpu.yaml`, to add the role as part of your custom AMI deployment.

## Notes
* Review `packer-ami.pkr.hcl` for all available variables.
* For Enroot, we are using shared filesystem (`/fsx`) for container cache, set this accordingly to your cluster in `roles/nvidia_enroot_pyxis/templates/enroot.conf` variable `ENROOT_CACHE_PATH`.
* Review variables (dependency versions) in `./roles/*/defaults/main.yml` according to [Ansible directory structure](https://docs.ansible.com/ansible/latest/tips_tricks/sample_setup.html).
* These are based upon using the default VPCs found in the account.  If this does not exist, the default VPC can be recreated with `aws ec2 create-default-vpc`.
* If packer can't find the AMI with the following message `Error querying AMI: InvalidAMIID.NotFound`, force the region by prepending your `make` command by the region `AWS_REGION=us-east-1`.
