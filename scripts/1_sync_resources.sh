#!/bin/bash

mkdir -p resources/neurips-2023-raw/
aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/sc_counts.h5ad resources/neurips-2023-raw/sc_counts.h5ad