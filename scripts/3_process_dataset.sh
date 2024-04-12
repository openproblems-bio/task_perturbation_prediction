#!/bin/bash

set -e

mkdir -p resources/neurips-2023-data/
viash run src/dge_perturbation_prediction/process_dataset/config.vsh.yaml \
  --sc_counts resources/neurips-2023-raw/sc_counts.h5ad \
  --de_train resources/neurips-2023-data/de_train.h5ad \
  --de_test resources/neurips-2023-data/de_test.h5ad