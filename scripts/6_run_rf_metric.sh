#!/bin/bash

set -e

# run metric
viash run src/dge_perturbation_prediction/metrics/mean_rowwise_rmse/config.vsh.yaml -- \
  --prediction resources/neurips-2023-data/output_rf.parquet \
  --de_test resources/neurips-2023-data/de_test.parquet \
  --output resources/neurips-2023-data/score_rf.json
