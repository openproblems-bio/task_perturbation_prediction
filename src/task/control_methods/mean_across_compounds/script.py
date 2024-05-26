import pandas as pd
import anndata as ad
import sys

## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "layer": "sign_log10_pval",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "output": "resources/neurips-2023-data/output_mean_compounds.parquet",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from anndata_to_dataframe import anndata_to_dataframe

de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train_h5ad.var_names)
de_train = anndata_to_dataframe(de_train_h5ad, par["layer"])

mean_compound = de_train.groupby("sm_name")[gene_names].mean()
mean_compound = mean_compound.loc[id_map.sm_name]

output = pd.DataFrame(mean_compound.values, index=id_map["id"], columns=gene_names).reset_index()
output.to_parquet(par["output"])