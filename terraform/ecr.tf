resource "aws_ecr_repository" "fastapi" {
  name                 = "${var.project_name}-fastapi"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-fastapi"
  }
}

resource "aws_ecr_repository" "frontend" {
  name                 = "${var.project_name}-streamlit"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-streamlit"
  }
}

resource "aws_ecr_repository" "monitoring" {
  name                 = "${var.project_name}-monitoring"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-monitoring"
  }
}

resource "aws_ecr_lifecycle_policy" "fastapi" {
  repository = aws_ecr_repository.fastapi.name

  policy = jsonencode({
    rules = [
      {
        description  = "keep only last 10 imgs"
        rulePriority = 1
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "frontend" {
  repository = aws_ecr_repository.frontend.name

  policy = jsonencode({
    rules = [
      {
        description  = "keep only last 10 imgs"
        rulePriority = 1
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "monitoring" {
  repository = aws_ecr_repository.monitoring.name

  policy = jsonencode({
    rules = [
      {
        description  = "keep only last 10 imgs"
        rulePriority = 1
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
