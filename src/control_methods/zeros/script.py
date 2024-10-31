import anndata as ad
import numpy as np
import pandas as pd

## VIASH START
par = {
  "de_train": "resources/datasets/neurips-2023-data/de_train.h5ad",
  "de_test": "resources/datasets/neurips-2023-data/de_test.h5ad",
  "layer": "clipped_sign_log10_pval",
  "id_map": "resources/datasets/neurips-2023-data/id_map.csv",
  "output": "resources/datasets/neurips-2023-data/output_mean.h5ad",
}
## VIASH END

de_train = ad.read_h5ad(par["de_train"])
id_map = pd.read_csv(par["id_map"])
gene_names = list(de_train.var_names)

prediction = np.zeros((id_map.shape[0], len(gene_names)))

# write output
output = ad.AnnData(
    layers={"prediction": prediction},
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
      "dataset_id": de_train.uns["dataset_id"],
      "method_id": meta["name"]
    }
)
output.write_h5ad(par["output"], compression="gzip")