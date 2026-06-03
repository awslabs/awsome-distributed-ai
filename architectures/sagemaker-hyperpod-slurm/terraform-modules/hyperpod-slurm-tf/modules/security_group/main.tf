resource "aws_security_group" "hyperpod" {
  name_prefix = "${var.resource_name_prefix}-sg"
  description = "Security group for SageMaker HyperPod with EFA support"
  vpc_id      = var.vpc_id

  tags = {
    Name = "${var.resource_name_prefix}-security-group"
  }
}

# Self-referencing ingress rule for EFA communication
resource "aws_security_group_rule" "efa_ingress" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 65535
  protocol                 = "-1"
  source_security_group_id = aws_security_group.hyperpod.id
  security_group_id        = aws_security_group.hyperpod.id
  description              = "All to all communication for EFA Ingress within Security Group"
}

# Self-referencing egress rule for EFA communication
resource "aws_security_group_rule" "efa_egress" {
  type                     = "egress"
  from_port                = 0
  to_port                  = 65535
  protocol                 = "-1"
  source_security_group_id = aws_security_group.hyperpod.id
  security_group_id        = aws_security_group.hyperpod.id
  description              = "All to all communication for EFA Egress within Security Group"
}

# FSx Lustre LNET traffic rules
resource "aws_security_group_rule" "fsx_lustre_lnet_tcp" {
  type              = "ingress"
  from_port         = 988
  to_port           = 988
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre LNET TCP traffic on port 988"
}

resource "aws_security_group_rule" "fsx_lustre_lnet_udp" {
  type              = "ingress"
  from_port         = 988
  to_port           = 988
  protocol          = "udp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre LNET UDP traffic on port 988"
}

# FSx Lustre LNET egress traffic rules
resource "aws_security_group_rule" "fsx_lustre_lnet_tcp_egress" {
  type              = "egress"
  from_port         = 988
  to_port           = 988
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre LNET TCP egress traffic on port 988"
}

resource "aws_security_group_rule" "fsx_lustre_lnet_udp_egress" {
  type              = "egress"
  from_port         = 988
  to_port           = 988
  protocol          = "udp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre LNET UDP egress traffic on port 988"
}

# FSx OpenZFS NFS traffic rules
resource "aws_security_group_rule" "fsx_openzfs_nfs_tcp" {
  type              = "ingress"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx OpenZFS NFS traffic on port 2049"
}

resource "aws_security_group_rule" "fsx_openzfs_nfs_tcp_egress" {
  type              = "egress"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx OpenZFS NFS egress traffic on port 2049"
}

# Additional FSx Lustre ports (1018-1023)
resource "aws_security_group_rule" "fsx_lustre_client_tcp" {
  type              = "ingress"
  from_port         = 1018
  to_port           = 1023
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre client communication ports 1018-1023"
}

resource "aws_security_group_rule" "fsx_lustre_client_tcp_egress" {
  type              = "egress"
  from_port         = 1018
  to_port           = 1023
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.hyperpod.id
  description       = "FSx Lustre client egress communication ports 1018-1023"
}

# Egress rule for internet access
resource "aws_security_group_rule" "internet_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 65535
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.hyperpod.id
  description       = "All to all communication for Egress to internet"
}

