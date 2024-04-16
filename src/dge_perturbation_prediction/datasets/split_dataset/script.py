import anndata as ad
import pandas as pd

## VIASH START
par = {
    "input": "resources/neurips-2023-data/de.h5ad",
    "output_train": "resources/neurips-2023-data/de_train.parquet",
    "output_test": "resources/neurips-2023-data/de_test.parquet",
    "output_id_map": "resources/neurips-2023-data/id_map.csv",
}
## VIASH END

print(">> Load dataset", flush=True)
input = ad.read_h5ad(par["input"])

print(">> Extract metadata", flush=True)
metadata = input.obs[['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'split', 'control']]

print(">> Extract sign log10 pval adj", flush=True)
sign_logfc_pval = pd.DataFrame(
  input.layers["sign_log10_pval_adj"],
  columns=input.var_names,
  index=input.obs.index
)

print(">> Merge metadata and sign log10 pval adj", flush=True)
full_data = pd.concat([metadata, sign_logfc_pval], axis=1).reset_index(drop=True)

print(">> Split data", flush=True)
de_train = full_data[full_data["split"] != "private_test"]
de_test = full_data[full_data["split"] == "private_test"].copy()

# reset index
de_test.reset_index(drop=True, inplace=True)

# add id column
de_test.reset_index(names="id", inplace=True)

# save id map
id_map = de_test[["id", "sm_name", "cell_type"]]

print(">> Save data", flush=True)
de_train.to_parquet(par["output_train"])
de_test.to_parquet(par["output_test"])
id_map.to_csv(par["output_id_map"], index=False)
