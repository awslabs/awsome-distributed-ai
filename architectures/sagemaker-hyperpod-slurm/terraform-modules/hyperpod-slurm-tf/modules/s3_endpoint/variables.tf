variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_route_table_id" {
  description = "ID of the private route table"
  type        = string
}

variable "public_route_table_id" {
  description = "ID of the public route table"
  type        = string
}