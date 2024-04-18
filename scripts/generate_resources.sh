#!/bin/bash

set -e

IN=resources/neurips-2023-raw
OUT=resources/neurips-2023-data

[[ ! -d $IN ]] && mkdir -p $IN

if [[ ! -f "$IN/sc_counts.h5ad" ]]; then
  echo "Downloading 'sc_counts.h5ad'"
  aws s3 cp --no-sign-request \
    s3://openproblems-bio/public/neurips-2023-competition/sc_counts.h5ad \
    "$IN/sc_counts.h5ad"
fi
if [[ ! -f "$IN/lincs_id_compound_mapping.parquet" ]]; then
  echo "Downloading 'lincs_id_compound_mapping.parquet'"
  aws s3 cp --no-sign-request \
    s3://saturn-kaggle-datasets/open-problems-single-cell-perturbations-optional/lincs_id_compound_mapping.parquet \
    "$IN/lincs_id_compound_mapping.parquet"
fi

echo "Running 'process_dataset' workflow"
nextflow run \
  target/nextflow/process_dataset/workflow/main.nf \
  -profile docker \
  -resume \
  --sc_counts "$IN/sc_counts.h5ad" \
  --lincs_id_compound_mapping "$IN/lincs_id_compound_mapping.parquet" \
  --pseudobulk "pseudo_bulk.h5ad" \
  --de_train_h5ad "de_train.h5ad" \
  --de_train_parquet "de_train.parquet" \
  --de_test_h5ad "de_test.h5ad" \
  --de_test_parquet "de_test.parquet" \
  --id_map "id_map.csv" \
  --publish_dir "$OUT"

# run method
viash run src/task/control_methods/sample/config.vsh.yaml -- \
  --de_train "$OUT/de_train.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.parquet"

# run metric
viash run src/task/metrics/mean_rowwise_rmse/config.vsh.yaml -- \
  --prediction "$OUT/prediction.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --output "$OUT/score.h5ad"

echo "Uploading results to S3"
aws s3 sync --profile op2 \
  --include "*" \
  --exclude "neurips-2023-raw/*" \
  --exclude "neurips-2023-public/*" \
  "resources" \
  "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
  --delete --dryrun
