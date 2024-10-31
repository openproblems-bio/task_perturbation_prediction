#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

# generate a unique id
RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
resources_dir="s3://openproblems-data/resources/task_perturbation_prediction/datasets/"
publish_dir="s3://openproblems-data/resources/task_perturbation_prediction/results/${RUN_ID}"

# write the parameters to file
cat > /tmp/params.yaml << HERE
param_list:
  - id: neurips-2023-data
    de_train: "$resources_dir/neurips-2023-data/de_train.h5ad"
    de_test: "$resources_dir/neurips-2023-data/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-data/id_map.csv"
    layer: clipped_sign_log10_pval
  # - id: neurips-2023-kaggle
  #   de_train: "$resources_dir/neurips-2023-kaggle/de_train.h5ad"
  #   de_test: "$resources_dir/neurips-2023-kaggle/de_test.h5ad"
  #   id_map: "$resources_dir/neurips-2023-kaggle/id_map.csv"
  #   layer: sign_log10_pval
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_perturbation_prediction.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_perturbation_prediction,full