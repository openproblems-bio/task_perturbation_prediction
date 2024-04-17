import anndata as ad

## VIASH START
par = {
    "input": "resources/neurips-2023-data/pseudobulk.h5ad",
    "output_train": "resources/neurips-2023-data/pseudobulk_train.h5ad",
    "output_test_celtypes": "resources/neurips-2023-data/pseudobulk_test_celltypes.h5ad",
}
## VIASH END

print(">> Load dataset", flush=True)
pseudobulk = ad.read_h5ad(par["input"])

print(">> Split data", flush=True)
pseudobulk_train = pseudobulk[pseudobulk.obs["split"] != "private_test"]
celltypes_in_test = pseudobulk[pseudobulk.obs["split"] == "private_test"].obs["cell_type"].unique()
pseudobulk_test_celltypes = pseudobulk[pseudobulk.obs["cell_type"].isin(celltypes_in_test)]

print(">> Save data", flush=True)
pseudobulk_train.write_h5ad(par["output_train"])
pseudobulk_test_celltypes.write_h5ad(par["output_test_celltypes"])
