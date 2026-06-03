output "fsx_lustre_id" {
  description = "ID of the FSx Lustre file system"
  value       = aws_fsx_lustre_file_system.main.id
}

output "fsx_lustre_dns_name" {
  description = "DNS name of the FSx Lustre file system"
  value       = aws_fsx_lustre_file_system.main.dns_name
}

output "fsx_lustre_mount_name" {
  description = "Mount name of the FSx Lustre file system"
  value       = aws_fsx_lustre_file_system.main.mount_name
}