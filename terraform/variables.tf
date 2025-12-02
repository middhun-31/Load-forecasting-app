variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "eu-north-1"
}

variable "cluster_name" {
  description = "The name for your EKS cluster."
  type        = string
  default     = "refit-project-cluster"
}


