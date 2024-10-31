#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

# NOTE: depending on the the datasets and components, you may need to launch this workflow
# on a different compute platform (e.g. a HPC, AWS Cloud, Azure Cloud, Google Cloud).
# please refer to the nextflow information for more details:
# https://www.nextflow.io/docs/latest/

set -e

echo "Running benchmark on test data"
echo "  Make sure to run 'scripts/project/build_all_docker_containers.sh'!"

# generate a unique id
resources_dir="resources"
RUN_ID="run_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="resources/results/${RUN_ID}"

# write the parameters to file
cat > /tmp/params.yaml << HERE
param_list:
  - id: neurips-2023-data
    de_train: "$resources_dir/neurips-2023-data/de_train.h5ad"
    de_test: "$resources_dir/neurips-2023-data/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-data/id_map.csv"
    layer: clipped_sign_log10_pval
  - id: neurips-2023-kaggle
    de_train: "$resources_dir/neurips-2023-kaggle/de_train.h5ad"
    de_test: "$resources_dir/neurips-2023-kaggle/de_test.h5ad"
    id_map: "$resources_dir/neurips-2023-kaggle/id_map.csv"
    layer: sign_log10_pval
output_state: "state.yaml"
publish_dir: "$publish_dir"
HERE

# run the benchmark
nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -c common/nextflow_helpers/labels_ci.config \
  -params-file /tmp/params.yaml
