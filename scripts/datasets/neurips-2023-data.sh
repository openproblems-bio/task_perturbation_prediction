#!/bin/bash

set -e

IN=resources/datasets_raw/neurips-2023-raw
OUT=resources/datasets/neurips-2023-data

[[ ! -d $IN ]] && mkdir -p $IN

if [[ ! -f "$IN/sc_counts.h5ad" ]]; then
  echo ">> Downloading 'sc_counts.h5ad'"
  aws s3 cp --no-sign-request \
    s3://openproblems-bio/public/neurips-2023-competition/sc_counts_reannotated_with_counts.h5ad \
    "$IN/sc_counts_reannotated_with_counts.h5ad"
fi

echo ">> Running 'process_dataset' workflow"
nextflow run \
  target/nextflow/workflows/process_dataset/main.nf \
  -profile docker \
  -resume \
  --id neurips-2023-data \
  --sc_counts "$IN/sc_counts_reannotated_with_counts.h5ad" \
  --dataset_id "neurips-2023-data" \
  --dataset_name "NeurIPS2023 scPerturb DGE" \
  --dataset_url "TBD" \
  --dataset_reference "TBD" \
  --dataset_summary "Differential gene expression sign(logFC) * -log10(p-value) values after 24 hours of treatment with 144 compounds in human PBMCs" \
  --dataset_description "For this competition, we designed and generated a novel single-cell perturbational dataset in human peripheral blood mononuclear cells (PBMCs). We selected 144 compounds from the Library of Integrated Network-Based Cellular Signatures (LINCS) Connectivity Map dataset (PMID: 29195078) and measured single-cell gene expression profiles after 24 hours of treatment. The experiment was repeated in three healthy human donors, and the compounds were selected based on diverse transcriptional signatures observed in CD34+ hematopoietic stem cells (data not released). We performed this experiment in human PBMCs because the cells are commercially available with pre-obtained consent for public release and PBMCs are a primary, disease-relevant tissue that contains multiple mature cell types (including T-cells, B-cells, myeloid cells, and NK cells) with established markers for annotation of cell types. To supplement this dataset, we also measured cells from each donor at baseline with joint scRNA and single-cell chromatin accessibility measurements using the 10x Multiome assay. We hope that the addition of rich multi-omic data for each donor and cell type at baseline will help establish biological priors that explain the susceptibility of particular genes to exhibit perturbation responses in difference biological contexts." \
  --dataset_organism "homo_sapiens" \
  --output_state "state.yaml" \
  --publish_dir "$OUT"

echo ">> Run method"
viash run src/control_methods/mean_across_compounds/config.vsh.yaml -- \
  --de_train_h5ad "$OUT/de_train.h5ad" \
  --de_test_h5ad "$OUT/de_test.h5ad" \
  --id_map "$OUT/id_map.csv" \
  --output "$OUT/prediction.h5ad"

echo ">> Run metric"
viash run src/metrics/mean_rowwise_error/config.vsh.yaml -- \
  --prediction "$OUT/prediction.h5ad" \
  --de_test_h5ad "$OUT/de_test.h5ad" \
  --output "$OUT/score.h5ad"

echo ">> Uploading results to S3"
# aws s3 sync --profile op2 \
#   --include "*" \
#   --exclude "neurips-2023-raw/*" \
#   --exclude "neurips-2023-public/*" \
#   "resources" \
#   "s3://openproblems-bio/public/neurips-2023-competition/workflow-resources/" \
#   --delete --dryrun
