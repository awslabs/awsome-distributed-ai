output "storage_class_name" {
  description = "StorageClass name for dynamic provisioning"
  value       = var.create_new_filesystem ? "fsx-sc" : null
}

output "fsx_csi_driver_addon_arn" {
  description = "ARN of the FSx CSI driver EKS addon"
  value       = aws_eks_addon.fsx_lustre_csi_driver.arn
}