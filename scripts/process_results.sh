#!/bin/bash

set -e

IN="output"
OUT="../website/results/dge_perturbation_prediction/data"

[[ ! -d "$IN" ]] && mkdir -p "$IN"

# run benchmark
export NXF_VER=23.04.2

NXF_VER=23.10.0 nextflow run \
  openproblems-bio/openproblems-v2 \
  -r main_build \
  -main-script target/nextflow/common/process_task_results/run/main.nf \
  -profile docker \
  -resume \
  -latest \
  --id "process" \
  --input_scores "$IN/score_uns.yaml" \
  --input_dataset_info "$IN/dataset_uns.yaml" \
  --input_method_configs "$IN/method_configs.yaml" \
  --input_metric_configs "$IN/metric_configs.yaml" \
  --input_execution "$IN/trace.txt" \
  --input_task_info "$IN/task_info.yaml" \
  --output_state "state.yaml" \
  --publish_dir "$OUT"
