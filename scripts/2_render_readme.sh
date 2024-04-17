#!/bin/bash

set -e

# create temp file and cleanup on exit
TMP_CONFIG=$(mktemp /tmp/nextflow.XXXXXX.config)
trap 'rm -f $TMP_CONFIG' EXIT

# create temporary nextflow config file
cat > $TMP_CONFIG <<EOF
process {
  errorStrategy = 'terminate'
}
EOF

# run nextflow to create the README.md file
nextflow run openproblems-bio/openproblems-v2 \
  -r main_build \
  -main-script target/nextflow/common/create_task_readme/main.nf \
  -profile docker \
  -latest \
  -c $TMP_CONFIG \
  --task "dge_perturbation_prediction" \
  --task_dir "src/dge_perturbation_prediction" \
  --github_url "https://github.com/openproblems-bio/task-dge-perturbation-prediction/tree/main/" \
  --output "README.md" \
  --viash_yaml "_viash.yaml" \
  --publish_dir src/dge_perturbation_prediction

# remove the unused state file
rm src/dge_perturbation_prediction/run.create_task_readme.state.yaml
