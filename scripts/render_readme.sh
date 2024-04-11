#!/bin/bash

set -e

nextflow run openproblems-bio/openproblems-v2 \
  -r main_build \
  -main-script target/nextflow/common/create_task_readme/main.nf \
  -profile docker \
  -latest \
  --task "dge-perturbation-prediction" \
  --task_dir "src" \
  --github_url "https://github.com/openproblems-bio/task-dge-perturbation-prediction/tree/main/" \
  --output "README.md" \
  --viash_yaml "_viash.yaml" \
  --publish_dir .

rm run.create_task_readme.state.yaml
