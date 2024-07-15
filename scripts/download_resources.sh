#!/bin/bash

set -e

echo ">> Downloading resources"
# aws s3 sync --no-sign-request \
#   "s3://openproblems-data/resources/perturbation_prediction/" \
#   "resources" \
#   --delete

common/sync_resources/sync_resources \
  --input "s3://openproblems-data/resources/perturbation_prediction/" \
  --output "resources" \
  --delete
