#!/bin/bash

set -e

IN=resources/neurips-2023-raw
OUT=resources/neurips-2023-data

# create directory if it doesn't exist
[[ -d "$OUT" ]] || mkdir -p "$OUT"

echo "Clean single-cell counts"
viash run src/dge_perturbation_prediction/process_dataset/clean_sc_counts/config.vsh.yaml -- \
  --input "$IN/sc_counts.h5ad" \
  --lincs_id_compound_mapping "$IN/lincs_id_compound_mapping.parquet" \
  --output "$OUT/sc_counts_cleaned.h5ad"

echo "Compute pseudobulk"
viash run src/dge_perturbation_prediction/process_dataset/compute_pseudobulk/config.vsh.yaml -- \
  --input "$OUT/sc_counts_cleaned.h5ad" \
  --output "$OUT/pseudobulk.h5ad"

echo "Run limma on training set"
viash run src/dge_perturbation_prediction/process_dataset/run_limma/config.vsh.yaml -- \
  --input "$OUT/pseudobulk.h5ad" \
  --output "$OUT/de_train.h5ad" \
  --input_splits "train;control;public_test" \
  --output_splits "train;control;public_test"

echo "Run limma on test set"
viash run src/dge_perturbation_prediction/process_dataset/run_limma/config.vsh.yaml -- \
  --input "$OUT/pseudobulk.h5ad" \
  --output "$OUT/de_test.h5ad" \
  --input_splits "train;control;public_test;private_test" \
  --output_splits "private_test"

echo "Convert h5ad to parquet"
viash run src/dge_perturbation_prediction/process_dataset/convert_h5ad_to_parquet/config.vsh.yaml -- \
  --input_train "$OUT/de_train.h5ad" \
  --input_test "$OUT/de_test.h5ad" \
  --output_train "$OUT/de_train.parquet" \
  --output_test "$OUT/de_test.parquet" \
  --output_id_map "$OUT/id_map.csv"

echo "Convert kaggle h5ad to parquet"
viash run src/dge_perturbation_prediction/process_dataset/convert_kaggle_h5ad_to_parquet/config.vsh.yaml -- \
  --input_train "$IN/2023-09-12_de_by_cell_type_train.h5ad" \
  --input_test "$IN/2023-09-12_de_by_cell_type_test.h5ad" \
  --output_train "$OUT/de_train_kaggle.parquet" \
  --output_test "$OUT/de_test_kaggle.parquet" \
  --output_id_map "$OUT/id_map_kaggle.csv"

# # Alternatively:
# nextflow run \
#   target/nextflow/dge_perturbation_prediction/process_dataset/workflow/main.nf \
#   -profile docker \
#   --sc_counts "$IN/sc_counts.h5ad" \
#   --lincs_id_compound_mapping "$IN/lincs_id_compound_mapping.parquet" \
#   --pseudobulk "pseudo_bulk.h5ad" \
#   --de_train_h5ad "de_train.h5ad" \
#   --de_train_parquet "de_train.parquet" \
#   --de_test_h5ad "de_test.h5ad" \
#   --de_test_parquet "de_test.parquet" \
#   --id_map "id_map.csv" \
#   --publish_dir "$OUT"