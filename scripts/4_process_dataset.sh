#!/bin/bash

set -e

# create directory if it doesn't exist
[[ -d resources/neurips-2023-data/ ]] || mkdir -p resources/neurips-2023-data/

# assuming that `viash ns build --setup cb --parallel` was run before
# target/docker/dge_perturbation_prediction/process_dataset/process_dataset \
#   ...

# or if it wasn't:
viash run src/dge_perturbation_prediction/process_dataset/config.vsh.yaml -- \
  --sc_counts resources/neurips-2023-raw/sc_counts.h5ad \
  --lincs_id_compound_mapping resources/neurips-2023-raw/lincs_id_compound_mapping.parquet \
  --de_train resources/neurips-2023-data/de_train.parquet \
  --de_test resources/neurips-2023-data/de_test.parquet \
  --id_map resources/neurips-2023-data/id_map.csv