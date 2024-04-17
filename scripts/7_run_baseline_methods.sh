#!/bin/bash

set -e

viash run src/dge_perturbation_prediction/control_methods/baseline_zero/config.vsh.yaml -- \
  --de_train resources/neurips-2023-data/de_train.parquet \
  --de_test resources/neurips-2023-data/de_test.parquet \
  --id_map resources/neurips-2023-data/id_map.csv \
  --output resources/neurips-2023-data/output_baseline_zero.parquet

viash run src/dge_perturbation_prediction/control_methods/ground_truth/config.vsh.yaml -- \
  --de_train resources/neurips-2023-data/de_train.parquet \
  --de_test resources/neurips-2023-data/de_test.parquet \
  --id_map resources/neurips-2023-data/id_map.csv \
  --output resources/neurips-2023-data/output_ground_truth.parquet

viash run src/dge_perturbation_prediction/control_methods/sample/config.vsh.yaml -- \
  --de_train resources/neurips-2023-data/de_train.parquet \
  --de_test resources/neurips-2023-data/de_test.parquet \
  --id_map resources/neurips-2023-data/id_map.csv \
  --output resources/neurips-2023-data/output_sample.parquet
