variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.30"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "stock-pred-eks"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "stock-pred"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS nodes"
  type        = string
  default     = "t3.xlarge"
}

variable "node_disk_size" {
  description = "Node disk size in GB"
  type        = number
  default     = 30
}

variable "node_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 3
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "artifact_bucket_name" {
  description = "S3 bucket name for project artifacts"
  type        = string
}
