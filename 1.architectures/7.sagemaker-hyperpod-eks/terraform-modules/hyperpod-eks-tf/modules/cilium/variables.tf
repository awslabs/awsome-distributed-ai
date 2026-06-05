variable "cilium_mode" {
  description = "Cilium operating mode: overlay, chaining, or custom."
  type        = string
  validation {
    condition     = contains(["overlay", "chaining", "custom"], var.cilium_mode)
    error_message = "cilium_mode must be one of: overlay, chaining, custom."
  }
}

variable "cilium_version" {
  description = "Cilium Helm chart version to deploy."
  type        = string
  default     = "1.19.4"
}

variable "cilium_helm_repository" {
  description = "Helm chart repository for Cilium. Override for closed-network deployments using a private mirror (e.g. an OCI registry: oci://<account>.dkr.ecr.<region>.amazonaws.com). Leave empty when cilium_helm_chart already contains a fully qualified oci:// reference."
  type        = string
  default     = "https://helm.cilium.io/"
}

variable "cilium_helm_chart" {
  description = "Helm chart name for Cilium, or a fully qualified OCI reference (e.g. oci://<account>.dkr.ecr.<region>.amazonaws.com/cilium) when cilium_helm_repository is empty."
  type        = string
  default     = "cilium"
}

variable "cilium_helm_values" {
  description = "Custom Helm values merged on top of mode-specific defaults. In custom mode, this IS the entire config."
  type        = any
  default     = {}
}
