output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_certificate_authority" {
  description = "EKS cluster CA cert (base64)"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "account_id" {
  description = "AWS account id"
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "ecr_frontend_url" {
  description = "ECR repo URL for frontend"
  value       = aws_ecr_repository.frontend.repository_url
}

output "ecr_fastapi_url" {
  description = "ECR repo URL for fastapi"
  value       = aws_ecr_repository.fastapi.repository_url
}

output "ecr_monitoring_url" {
  description = "ECR repo URL for monitoring"
  value       = aws_ecr_repository.monitoring.repository_url
}

output "ecr_login_command" {
  description = "Command to login to ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.main.name}"
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnets" {
  description = "Public subnet ids"
  value       = aws_subnet.public[*].id
}

output "private_subnets" {
  description = "Private subnet ids"
  value       = aws_subnet.private[*].id
}

output "artifact_bucket" {
  description = "Artifact bucket name"
  value       = aws_s3_bucket.artifacts.bucket
}
