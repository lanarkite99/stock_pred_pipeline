$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$terraformDir = Join-Path $repoRoot "terraform"
$tfvarsPath = Join-Path $terraformDir "terraform.tfvars"

if (-not (Test-Path $tfvarsPath)) {
    throw "terraform.tfvars not found at $tfvarsPath"
}

$tfvarsContent = Get-Content $tfvarsPath -Raw

function Get-TfVarValue {
    param(
        [string]$Content,
        [string]$Name
    )

    $pattern = '(?m)^\s*' + [regex]::Escape($Name) + '\s*=\s*"([^"]+)"\s*$'
    $match = [regex]::Match($Content, $pattern)
    if (-not $match.Success) {
        throw "Could not read terraform variable '$Name' from terraform.tfvars"
    }
    return $match.Groups[1].Value
}

$awsRegion = Get-TfVarValue -Content $tfvarsContent -Name "aws_region"
$clusterName = Get-TfVarValue -Content $tfvarsContent -Name "cluster_name"
$projectName = Get-TfVarValue -Content $tfvarsContent -Name "project_name"
$namespace = "stock-pred"
$ecrRepos = @(
    "$projectName-fastapi",
    "$projectName-streamlit",
    "$projectName-monitoring"
)

Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host "WARNING: This will destroy the AWS infrastructure for this project." -ForegroundColor Yellow
Write-Host "Region: $awsRegion"
Write-Host "Cluster: $clusterName"
Write-Host "Namespace: $namespace"
Write-Host "ECR repositories to empty: $($ecrRepos -join ', ')"
Write-Host "=================================================================" -ForegroundColor Yellow

$confirmation = Read-Host "Type YES to continue"
if ($confirmation -ne "YES") {
    Write-Host "Aborting destroy." -ForegroundColor Yellow
    exit 1
}

Write-Host "Step 1: Emptying ECR repositories..." -ForegroundColor Cyan
foreach ($repo in $ecrRepos) {
    Write-Host "  Cleaning $repo..."
    try {
        $images = aws ecr list-images --repository-name $repo --region $awsRegion --query "imageIds[*]" --output json 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($images) -or $images -eq "[]") {
            Write-Host "    Already empty or not found."
            continue
        }

        aws ecr batch-delete-image --repository-name $repo --region $awsRegion --image-ids $images | Out-Null
        Write-Host "    Images deleted."
    } catch {
        Write-Host "    Could not fully clean $repo. Continuing..." -ForegroundColor Yellow
    }
}

Write-Host "Step 2: Cleaning up Kubernetes resources..." -ForegroundColor Cyan
aws eks describe-cluster --name $clusterName --region $awsRegion *> $null
if ($LASTEXITCODE -eq 0) {
    aws eks update-kubeconfig --name $clusterName --region $awsRegion | Out-Null

    kubectl delete svc frontend -n $namespace --ignore-not-found=true
    kubectl delete deployment frontend -n $namespace --ignore-not-found=true
    kubectl delete deployment fastapi -n $namespace --ignore-not-found=true
    kubectl delete deployment redis -n $namespace --ignore-not-found=true
    kubectl delete deployment chroma -n $namespace --ignore-not-found=true
    kubectl delete deployment monitoring-app -n $namespace --ignore-not-found=true
    kubectl delete pvc --all -n $namespace --ignore-not-found=true
    kubectl delete namespace $namespace --ignore-not-found=true

    Write-Host "  Waiting for AWS load balancer and volumes to begin cleanup..."
    Start-Sleep -Seconds 30
} else {
    Write-Host "  Cluster not found or not reachable. Skipping kubectl cleanup." -ForegroundColor Yellow
}

Write-Host "Step 3: Running Terraform destroy..." -ForegroundColor Cyan
Set-Location $terraformDir
terraform init
terraform destroy

Write-Host "Step 4: Best-effort cleanup for leftover log group..." -ForegroundColor Cyan
aws logs delete-log-group --log-group-name "/aws/eks/$clusterName/cluster" --region $awsRegion 2>$null

Write-Host "Step 5: Final verification snapshot..." -ForegroundColor Cyan
Write-Host "Active EC2 instances:"
aws ec2 describe-instances --region $awsRegion --filters "Name=instance-state-name,Values=running,pending,stopping,stopped" --query "Reservations[*].Instances[*].[InstanceId,State.Name]" --output table

Write-Host "Load balancers:"
aws elbv2 describe-load-balancers --region $awsRegion --query "LoadBalancers[*].LoadBalancerName" --output table

Write-Host "NAT gateways:"
aws ec2 describe-nat-gateways --region $awsRegion --filter "Name=state,Values=available,pending,deleting" --query "NatGateways[*].NatGatewayId" --output table

Write-Host "EKS clusters:"
aws eks list-clusters --region $awsRegion --query "clusters" --output table

Write-Host "Cleanup finished. If any resources still appear above, wait a few minutes and check again." -ForegroundColor Green
