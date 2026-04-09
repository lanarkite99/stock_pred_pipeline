#!/bin/bash
set -e

CLUSTER_NAME=${1:-stock-pred-eks}
REGION=${2:-ap-south-1}

aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/volumes.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/chroma.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/fastapi.yaml
kubectl apply -f k8s/frontend.yaml
