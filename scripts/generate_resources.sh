#!/bin/bash

set -e

IN=resources/neurips-2023-raw
OUT=resources/neurips-2023-data

[[ ! -d $IN ]] && mkdir -p $IN

if [[ ! -f "$IN/sc_counts.h5ad" ]]; then
  echo ">> Downloading 'sc_counts.h5ad'"
  aws s3 cp --no-sign-request \
    s3://openproblems-bio/public/neurips-2023-competition/sc_counts_reannotated_with_counts.h5ad \
    "$IN/sc_counts.h5ad"
fi

echo ">> Run method"
viash run src/task/control_methods/sample/config.vsh.yaml -- \
  --de_train "$OUT/de_train.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.parquet"

echo ">> Run metric"
viash run src/task/metrics/mean_rowwise_error/config.vsh.yaml -- \
  --prediction "$OUT/prediction.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --output "$OUT/score.h5ad"

echo ">> Uploading results to S3"
aws s3 sync --profile op2 \
  --include "*" \
  --exclude "neurips-2023-raw/*" \
  --exclude "neurips-2023-public/*" \
  "resources" \
  "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
  --delete --dryrun
