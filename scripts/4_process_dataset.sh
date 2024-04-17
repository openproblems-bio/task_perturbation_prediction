#!/bin/bash

set -e

IN=resources/neurips-2023-raw
OUT=resources/neurips-2023-data

# create directory if it doesn't exist
[[ -d "$OUT" ]] || mkdir -p "$OUT"

echo "Clean single-cell counts"
viash run src/dge_perturbation_prediction/datasets/clean_sc_counts/config.vsh.yaml -- \
  --input "$IN/sc_counts.h5ad" \
  --lincs_id_compound_mapping "$IN/lincs_id_compound_mapping.parquet" \
  --output "$OUT/sc_counts_cleaned.h5ad"

echo "Compute pseudobulk"
viash run src/dge_perturbation_prediction/datasets/compute_pseudobulk/config.vsh.yaml -- \
  --input "$OUT/sc_counts_cleaned.h5ad" \
  --output "$OUT/pseudobulk.h5ad"

echo "Initial dataset split"
viash run src/dge_perturbation_prediction/datasets/initial_dataset_split/config.vsh.yaml -- \
  --input "$OUT/pseudobulk.h5ad" \
  --output_train "$OUT/pseudobulk_train.h5ad" \
  --output_test_celltypes "$OUT/pseudobulk_test_celltypes.h5ad"

echo "Run limma on all the samples with celltypes appearing in test set"
viash run src/dge_perturbation_prediction/datasets/run_limma/config.vsh.yaml -- \
  --input "$OUT/pseudobulk_test_celltypes.h5ad" \
  --output "$OUT/de_test_raw.h5ad"

echo "Run limma on training set"
viash run src/dge_perturbation_prediction/datasets/run_limma/config.vsh.yaml -- \
  --input "$OUT/pseudobulk_train.h5ad" \
  --output "$OUT/de_train_raw.h5ad"

echo "Split dataset"
viash run src/dge_perturbation_prediction/datasets/split_dataset/config.vsh.yaml -- \
  --input "$OUT/de.h5ad" \
  --output_train "$OUT/de_train.parquet" \
  --output_test "$OUT/de_test.parquet" \
  --output_id_map "$OUT/id_map.csv"