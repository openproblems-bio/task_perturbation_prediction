import pandas as pd
import anndata as ad
import numpy as np
import sys

## VIASH START
par = {
  "de_train": "resources/datasets/neurips-2023-data/de_train.h5ad",
  "de_test": "resources/datasets/neurips-2023-data/de_test.h5ad",
  "layer": "clipped_sign_log10_pval",
  "id_map": "resources/datasets/neurips-2023-data/id_map.csv",
  "output": "resources/datasets/neurips-2023-data/output_mean.h5ad",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from anndata_to_dataframe import anndata_to_dataframe

de_train = ad.read_h5ad(par["de_train"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train.var_names)
de_train_df = anndata_to_dataframe(de_train, par["layer"])

mean_pred = de_train_df[gene_names].mean(axis=0)

# write output
output = ad.AnnData(
    layers={
        "prediction": np.vstack([mean_pred.values] * id_map.shape[0])
    },
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
      "dataset_id": de_train.uns["dataset_id"],
      "method_id": meta["name"]
    }
)
output.write_h5ad(par["output"], compression="gzip")