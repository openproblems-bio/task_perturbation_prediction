#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

# generate a unique id
RUN_ID="stability_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="s3://openproblems-data/resources/task_perturbation_prediction/results/${RUN_ID}"

cat > /tmp/params.yaml << HERE
id: neurips-2023-data
sc_counts: s3://openproblems-bio/public/neurips-2023-competition/sc_counts_reannotated_with_counts.h5ad
scores: stability_uns.yaml
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_perturbation_prediction.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_stability_analysis/main.nf \
  --workspace 53907369739130 \
  --compute-env 5DwwhQoBi0knMSGcwThnlF \
  --params-file /tmp/params.yaml \
  --config src/common/nextflow_helpers/labels_tw.config \
  --labels task_perturbation_prediction,stability
