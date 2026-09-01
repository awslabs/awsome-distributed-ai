variable "resource_name_prefix" {
  description = "Prefix to be used for all resources created by this module"
  type        = string
}

variable "eks_cluster_name" {
  description = "The name of the EKS cluster"
  type        = string
}

variable "helm_repo_path" {
  description = "The path to the HyperPod Helm chart in the Helm repo"
  type        = string
  default     = "helm_chart/HyperPodHelmChart"
}

variable "namespace" {
  description = "The namespace to deploy the HyperPod Helm chart into"
  type        = string
  default     = "kube-system"
}

variable "helm_release_name" {
  description = "The name of the Helm release"
  type        = string
  default     = "hyperpod-dependencies"
}

variable "rig_mode" {
  description = "Whether restricted instance groups are configured"
  type        = bool
}

variable "helm_repo_revision" {
  description = "Git revision for normal mode"
  type        = string
}

variable "helm_repo_revision_rig" {
  description = "Git revision for RIG mode"
  type        = string
}

variable "enable_gpu_operator" {
  description = "Whether to enable the GPU operator"
  type        = bool
}

variable "enable_mlflow" {
  description = "Whether to enable the MLFlow"
  type        = bool
}

variable "enable_kubeflow_training_operators" {
  description = "Whether to enable the Kubeflow training operators"
  type        = bool
}

variable "enable_cluster_role_and_bindings" {
  description = "Whether to enable the cluster role and bindings"
  type        = bool
}

variable "enable_namespaced_role_and_bindings" {
  description = "Whether to enable the namespaced role and bindings"
  type        = bool
}

variable "enable_team_role_and_bindings" {
  description = "Whether to enable the team role and binding"
  type        = bool
}

variable "enable_nvidia_device_plugin" {
  description = "Whether to enable the NVIDIA device plugin"
  type        = bool
}

variable "enable_neuron_device_plugin" {
  description = "Whether to enable the Neuron device plugin"
  type        = bool
}

variable "enable_mpi_operator" {
  description = "Whether to enable the MPI operator"
  type        = bool
}

variable "enable_deep_health_check" {
  description = "Whether to enable the deep health check"
  type        = bool
}

variable "enable_job_auto_restart" {
  description = "Whether to enable the job auto restart"
  type        = bool
}

variable "enable_hyperpod_patching" {
  description = "Whether to enable the hyperpod patching"
  type        = bool
}

variable "rig_script_path" {
  description = "The path to the RIG script"
  type        = string
  default     = "helm_chart/install_rig_dependencies.sh"
}
