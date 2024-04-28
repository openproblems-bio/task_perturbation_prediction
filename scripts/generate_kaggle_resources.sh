#!/bin/bash

set -e

OUT=resources/neurips-2023-kaggle

[[ ! -d $OUT ]] && mkdir -p $OUT

aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_test.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_test.h5ad
aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_train.h5ad --no-sign-request $OUT/2023-09-12_de_by_cell_type_train.h5ad

# recompress h5ad files
python -c \
  "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_test.h5ad', compression='gzip')"
python -c \
  "import anndata as ad; ad.read_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad').write_h5ad('$OUT/2023-09-12_de_by_cell_type_train.h5ad', compression='gzip')"

viash run src/task/process_dataset/convert_kaggle_h5ad_to_parquet/config.vsh.yaml -- \
  --input_train "$OUT/2023-09-12_de_by_cell_type_train.h5ad" \
  --input_test "$OUT/2023-09-12_de_by_cell_type_test.h5ad" \
  --output_train "$OUT/de_train.parquet" \
  --output_test "$OUT/de_test.parquet" \
  --output_id_map "$OUT/id_map.csv"

echo ">> Run method"
viash run src/task/control_methods/sample/config.vsh.yaml -- \
  --de_train "$OUT/de_train.parquet" \
  --de_test "$OUT/de_test.parquet" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.parquet"

echo ">> Run metric"
viash run src/task/metrics/mean_rowwise_rmse/config.vsh.yaml -- \
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