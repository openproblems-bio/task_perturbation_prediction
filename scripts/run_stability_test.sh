#!/bin/bash

set -e

IN="resources"
OUT="output"

[[ ! -d "$OUT" ]] && mkdir -p "$OUT"

# run benchmark
# 'input_states' looks for state.yaml files corresponding to datasets
export NXF_VER=23.04.2

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  --publish_dir "$OUT" \
  -entry auto \
  --input_states "$IN/**/state.yaml" \
  --rename_keys 'de_train:de_train,de_train_h5ad:de_train_h5ad,de_test:de_test,de_test_h5ad:de_test_h5ad,id_map:id_map' \
  --settings '{"bootstrap": true, "bootstrap_num_replicates": 3, "bootstrap_obs_fraction": 0.99, "bootstrap_var_fraction": 0.99, "method_ids": ["zeros", "sample", "ground_truth", "mean_across_types", "mean_across_compounds"]}' \
  --output_state "state.yaml"