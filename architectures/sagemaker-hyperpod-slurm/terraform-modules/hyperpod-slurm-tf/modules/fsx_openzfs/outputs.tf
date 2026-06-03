output "fsx_openzfs_id" {
  description = "ID of the FSx OpenZFS file system"
  value       = aws_fsx_openzfs_file_system.main.id
}

output "fsx_openzfs_dns_name" {
  description = "DNS name of the FSx OpenZFS file system"
  value       = aws_fsx_openzfs_file_system.main.dns_name
}
