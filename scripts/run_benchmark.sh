#!/bin/bash

set -e

DATASETS_DIR="resources"
OUTPUT_DIR="output"

[[ ! -d "$OUTPUT_DIR" ]] && mkdir -p "$OUTPUT_DIR"

# run benchmark
export NXF_VER=23.04.2

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -entry auto \
  -with-trace \
  --input_states "$DATASETS_DIR/**/state.yaml" \
  --settings '{"output_scores": "scores.yaml", "output_dataset_info": "dataset_info.yaml", "output_method_configs": "method_configs.yaml", "output_metric_configs": "metric_configs.yaml", "output_task_info": "task_info.yaml"}' \
  --publish_dir "$OUTPUT_DIR" \
  --output_state 'state.yaml'