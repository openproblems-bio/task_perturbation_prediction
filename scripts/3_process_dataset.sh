#!/bin/bash

set -e

viash run src/dge_perturbation_prediction/process_dataset/config.vsh.yaml \
  --de_per_plate_by_celltype resources/neurips-2023-raw/de_per_plate_by_cell_type.h5ad \
  --de_per_plate resources/neurips-2023-raw/de_per_plate.h5ad \
  --de_train resources/neurips-2023-data/de_train.h5ad \
  --de_test resources/neurips-2023-data/de_test.h5ad