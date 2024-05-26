import pandas as pd
import anndata as ad
import sys

## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "layer": "sign_log10_pval",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "resources/neurips-2023-data/output_mean_celltypes.parquet",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from anndata_to_dataframe import anndata_to_dataframe

de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train_h5ad.var_names)
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])

# compute mean celltype
mean_celltype = de_train.groupby("cell_type")[gene_names].mean()
mean_celltype = mean_celltype.loc[id_map.cell_type]

output = pd.DataFrame(mean_celltype.values, index=id_map["id"], columns=gene_names).reset_index()
output.to_parquet(par["output"])