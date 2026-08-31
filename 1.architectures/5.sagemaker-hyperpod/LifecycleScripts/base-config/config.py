
# Basic configuration parameters
class Config:

    # Default is true to install Docker/Enroot/Pyxis.
    enable_docker_enroot_pyxis = True

    # Set true if you want to install metric exporters and OpenTelemetry collector for observability
    # For more information, see https://catalog.workshops.aws/sagemaker-hyperpod/en-US/09-observability
    enable_observability = False

    # Set true if you want to:
    # - fix Slurm slurmctld being not responsive at restart
    # - install pam_slurm_adopt PAM module to:
    #    - Limit host memory usage at 99% MaxRAMPercent using cgroup enforcement
    #    - Prevent user to ssh without jobs running on that node
    enable_pam_slurm_adopt = False

    # Set true if you want to install SSSD for ActiveDirectory/LDAP integration.
    # You need to configure parameters in SssdConfig as well.
    enable_sssd = False

    # Set true if you want to use mountpoint for s3 on cluster nodes. 
    # If enabled, a systemctl mount-s3.service file will be writen that will mount at /mnt/<BucketName>.
    # requires s3 permissions to be added to cluster execution role. 
    enable_mount_s3 = False

    # Set true if you want to use FSx OpenZFS in addition to FSxL. 
    enable_fsx_openzfs = False

    # Set false if you want to disable log rotation of Slurm daemon logs
    enable_slurm_log_rotation = True

    s3_bucket = "" # required when enable_mount_s3 = True, replace with your actual data bucket name in quotes, ie. "my-dataset-bucket"

    if enable_mount_s3 and not s3_bucket:
        raise ValueError("Error: A bucket name must be specified when enable_mount_s3 is True")


# Configuration parameters for observability
# For more information, see https://catalog.workshops.aws/sagemaker-hyperpod/en-US/09-observability
class ObservabilityConfig:

    # Prometheus remote write URL
    prometheus_remote_write_url = "https://aps-workspaces.us-west-2.amazonaws.com/workspaces/ws-abcd1234-abcd-1234-ab12-1234abcd1234/api/v1/remote_write"

    # Set true if you want to collect advanced metrics
    advanced_metrics = False

    # NCCL Inspector metrics via node_exporter textfile collector.
    # Requires the NCCL Inspector profiler plugin (.so) to be pre-built and installed on compute nodes.
    # Build from NCCL source (post-v2.28.3): plugins/profiler/inspector/
    # See: https://github.com/NVIDIA/nccl/tree/master/plugins/profiler/inspector
    nccl_metrics_enabled = False
    nccl_metrics_dump_interval_seconds = 30
    nccl_profiler_plugin_path = "/opt/nccl-inspector/libnccl-profiler-inspector.so"


# Configuration parameters for ActiveDirectory/LDAP/SSSD
class SssdConfig:

    # Name of domain. Can be default if you are not sure.
    domain = "default"

    # Comma separated list of LDAP server URIs
    ldap_uri = "ldaps://nlb-ds-xyzxyz.elb.us-west-2.amazonaws.com"

    # The default base DN to use for performing LDAP user operations
    ldap_search_base = "dc=hyperpod,dc=abc123,dc=com"

    # The default bind DN to use for performing LDAP operations
    ldap_default_bind_dn = "CN=ReadOnly,OU=Users,OU=hyperpod,DC=hyperpod,DC=abc123,DC=com"

    # "password" or "obfuscated_password". Obfuscated password is recommended.
    ldap_default_authtok_type = "obfuscated_password"

    # You need to modify this parameter with the obfuscated password, not plain text password
    ldap_default_authtok = "placeholder"

    # SSH authentication method - "password" or "publickey"
    ssh_auth_method = "publickey"

    # Home directory. You can change it to "/home/%u" if your cluster doesn't use FSx volume.
    override_homedir = "/fsx/%u"

    # Group names to accept SSH login
    ssh_allow_groups = {
        "controller" : ["ClusterAdmin", "ubuntu"],
        "compute" : ["ClusterAdmin", "ClusterDev", "ubuntu"],
        "login" : ["ClusterAdmin", "ClusterDev", "ubuntu"],
    }

    # Group names for sudoers
    sudoers_groups = {
        "controller" : ["ClusterAdmin", "ClusterDev"],
        "compute" : ["ClusterAdmin", "ClusterDev"],
        "login" : ["ClusterAdmin", "ClusterDev"],
    }
