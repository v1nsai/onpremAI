#!/bin/bash

set -e

# Install the helm repository
helm repo add go-skynet https://go-skynet.github.io/helm-charts/
# Update the repositories
helm repo update
# Install the helm chart
helm upgrade --install local-ai go-skynet/local-ai \
    --create-namespace \
    --namespace localai \
    --values projects/onpremAI/api/kubernetes/values.yaml

kubectl cp -r projects/onpremAI/pipelines/finetuning/google-bert/bert-base-cased.model -n localai $(kubectl get pods -n localai -l app.kubernetes.io/name=local-ai -o jsonpath='{.items[0].metadata.name}'):/models/bert-base-cased.model