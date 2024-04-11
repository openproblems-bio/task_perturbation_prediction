import anndata as ad

## VIASH START
par = {
    "de_per_plate_by_cell_type": "resources/neurips-2023-raw/de_per_plate_by_cell_type.h5ad",
    "de_per_plate": "resources/neurips-2023-raw/de_per_plate.h5ad",
    "de_train": "resources/neurips-2023-data/de_train.h5ad",
    "de_test": "resources/neurips-2023-data/de_test.h5ad",
}
## VIASH END

print(">> Load dataset", flush=True)
de_per_plate_by_cell_type = ad.read_h5ad(par["de_per_plate_by_cell_type"])
de_per_plate = ad.read_h5ad(par["de_per_plate"])

print(">> Split dataset", flush=True)
de_train = ...
de_test = ...

print(">> Write to disk", flush=True)
de_train.write_h5ad(par["de_train"])
de_test.write_h5ad(par["de_test"])
