module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.10.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    ml_workers = {
      name           = "ml-workers"
      instance_types = ["m5.large"]
      min_size       = 2
      max_size       = 4
      desired_size   = 3
      subnet_ids     = module.vpc.private_subnets
    }
  }
}