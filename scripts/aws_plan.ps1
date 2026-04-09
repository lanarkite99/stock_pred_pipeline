$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$terraformDir = Join-Path $repoRoot "terraform"

Write-Host "Running Terraform plan for AWS infrastructure..." -ForegroundColor Cyan
Set-Location $terraformDir

terraform init
terraform plan
