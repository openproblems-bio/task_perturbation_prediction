import anndata as ad

## VIASH START
par = {
    "sc_counts": "resources/neurips-2023-raw/sc_counts.h5ad",
    "lincs_id_compound_mapping": "resources/neurips-2023-raw/lincs_id_compound_mapping.parquet",
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "de_test": "resources/neurips-2023-data/de_test.parquet",
    "id_map": "resources/neurips-2023-data/id_map.csv"
}
## VIASH END

print(">> Load dataset", flush=True)
sc_counts = ad.read_h5ad(par["sc_counts"])

print(">> Split dataset", flush=True)
de_train = ...
de_test = ...

print(">> Write to disk", flush=True)
de_train.write_h5ad(par["de_train"])
de_test.write_h5ad(par["de_test"])
