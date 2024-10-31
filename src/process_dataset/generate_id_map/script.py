import anndata as ad

## VIASH START
par = {
    "de_test": "resources/datasets/neurips-2023-data/de_test.h5ad",
    "id_map": "resources/datasets/neurips-2023-data/id_map.csv",
}
## VIASH END

print(">> Load dataset", flush=True)
de_test = ad.read_h5ad(par["de_test"])

print(">> Generate id_map file", flush=True)
id_map = de_test.obs[["sm_name", "cell_type"]]
id_map.reset_index(drop=True, inplace=True)
id_map.reset_index(names="id", inplace=True)

print(">> Save data", flush=True)
id_map.to_csv(par["id_map"], index=False)
