import anndata as ad

## VIASH START
par = {
    "sc_counts": "resources/neurips-2023-raw/sc_counts.h5ad",
    "de_train": "resources/neurips-2023-data/de_train.h5ad",
    "de_test": "resources/neurips-2023-data/de_test.h5ad",
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
