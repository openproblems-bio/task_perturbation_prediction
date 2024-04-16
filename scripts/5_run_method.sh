#!/bin/bash

set -e

viash run src/dge_perturbation_prediction/methods/random_forest/config.vsh.yaml -- \
  --de_train resources/neurips-2023-data/de_train.parquet \
  --id_map resources/neurips-2023-data/id_map.csv \
  --output resources/neurips-2023-data/output_rf.csv
