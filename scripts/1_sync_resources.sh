#!/bin/bash

mkdir -p resources/neurips-2023-raw/
aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/sc_counts.h5ad --no-sign-request resources/neurips-2023-raw/sc_counts.h5ad
aws s3 cp s3://saturn-kaggle-datasets/open-problems-single-cell-perturbations-optional/lincs_id_compound_mapping.parquet --no-sign-request resources/neurips-2023-raw/lincs_id_compound_mapping.parquet