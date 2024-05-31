import anndata as ad
import numpy as np
import pandas as pd
import os

# VIASH START
par = {
    "input": "resources/neurips-2023-raw/sc_counts_reannotated_with_counts.h5ad",
    "output": "output/sc_counts_bootstrapped_*.h5ad",
    "num_replicates": 10,
    "obs_fraction": 0.95,
    "var_fraction": 1
}
# VIASH END

# workaround for list bug
if isinstance(par["output"], list):
    par["output"] = par["output"][0]

# Load data
input_data = ad.read_h5ad(par["input"])

for i in range(par["num_replicates"]):
    print(f"Generating replicate {i+1}")

    # Sample indices
    obs_ix = np.random.choice(
        input_data.obs_names, int(input_data.n_obs * par["obs_fraction"]), replace=False
    )
    var_ix = np.random.choice(
        input_data.var_names, int(input_data.n_vars * par["var_fraction"]), replace=False
    )

    # Subset AnnData object
    output_data = input_data[obs_ix, var_ix].copy()

    # Update dataset ID metadata
    original_dataset_id = output_data.uns.get("dataset_id", "unknown_dataset")
    dataset_id = f"{original_dataset_id}-bootstrap{i+1}"
    output_data.uns["dataset_id"] = dataset_id
    output_data.uns["original_dataset_id"] = original_dataset_id

    # Construct output path
    output_path = par["output"].replace("*", str(i+1))

    # Write output (with compression)
    output_data.write_h5ad(output_path, compression="gzip")
