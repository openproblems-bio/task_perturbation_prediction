#!/bin/bash

RUN_ID="stability_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/perturbation_prediction/results/${RUN_ID}"

cat > /tmp/params.yaml << HERE
id: neurips-2023-data
sc_counts: s3://openproblems-bio/public/neurips-2023-competition/sc_counts_reannotated_with_counts.h5ad
scores: stability_uns.yaml
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_perturbation_prediction.git \
  --revision main_build \
  --pull-latest \
  --main-script target/nextflow/workflows/run_stability_analysis/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --config src/common/nextflow_helpers/labels_tw.config
