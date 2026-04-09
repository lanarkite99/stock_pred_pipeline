$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$terraformDir = Join-Path $repoRoot "terraform"

Write-Host "Running Terraform apply for AWS infrastructure..." -ForegroundColor Cyan
Write-Host "This will create billable AWS resources." -ForegroundColor Yellow

Set-Location $terraformDir

terraform init
terraform apply
