provider "aws" {
  region = var.aws_region
}

provider "awscc" {
  region = var.aws_region
}

provider "helm" {
  kubernetes = {
    host                   = var.create_eks_module ? module.eks_cluster[0].eks_cluster_endpoint : data.aws_eks_cluster.existing_eks_cluster[0].endpoint
    cluster_ca_certificate = base64decode(var.create_eks_module ? module.eks_cluster[0].eks_cluster_certificate_authority : data.aws_eks_cluster.existing_eks_cluster[0].certificate_authority[0].data)
    exec = {
      api_version = "client.authentication.k8s.io/v1beta1"
      args        = ["eks", "get-token", "--cluster-name", var.create_eks_module ? module.eks_cluster[0].eks_cluster_name : var.existing_eks_cluster_name]
      command     = "aws"
    }
  }
}

provider "kubernetes" {
  host                   = var.create_eks_module ? module.eks_cluster[0].eks_cluster_endpoint : data.aws_eks_cluster.existing_eks_cluster[0].endpoint
  cluster_ca_certificate = base64decode(var.create_eks_module ? module.eks_cluster[0].eks_cluster_certificate_authority : data.aws_eks_cluster.existing_eks_cluster[0].certificate_authority[0].data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    args        = ["eks", "get-token", "--cluster-name", var.create_eks_module ? module.eks_cluster[0].eks_cluster_name : var.existing_eks_cluster_name]
    command     = "aws"
  }
}

provider "grafana" {
  url  = var.create_observability_module && local.is_amp_allowed ? "https://${module.observability[0].grafana_workspace_endpoint}" : null
  auth = var.create_observability_module && local.is_amp_allowed ? module.observability[0].grafana_service_account_token : null
}
