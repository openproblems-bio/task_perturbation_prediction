#!/bin/bash

set -e

echo ">> Run $meta_functionality_name"
"$meta_executable" \
  --sc_counts "$meta_resources_dir/neurips-2023-raw/sc_counts.h5ad" \
  --lincs_id_compound_mapping "$meta_resources_dir/neurips-2023-raw/lincs_id_compound_mapping.parquet" \
  --de_train "de_train.parquet" \
  --de_test "de_test.parquet" \
  --id_map "id_map.csv"

echo ">> Checking output files"
if [ ! -f "de_train.parquet" ]; then
  echo "Error: Expected output file de_train.parquet not found"
  exit 1
fi
if [ ! -f "de_test.parquet" ]; then
  echo "Error: Expected output file de_test.parquet not found"
  exit 1
fi
if [ ! -f "id_map.csv" ]; then
  echo "Error: Expected output file id_map.csv not found"
  exit 1
fi

echo ">> Checking content of output files"
# TODO

echo ">> Done"
