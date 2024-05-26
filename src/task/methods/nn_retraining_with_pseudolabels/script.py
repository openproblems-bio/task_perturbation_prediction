# According to the author, the output is created by running two notebooks:
# 
#  * notebook 264: https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206
#  * notebook 266: https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153141755
# 
# This component was created by:
#  * Taking the code of both notebooks
#  * Moving the code corresponding to the weights and models from each notebook to a separate helper file
#  * Write this script:
#      - Load the data in this script
#      - Run notebook 264 on it
#      - Run notebook 266 on the combined training data and output of notebook 264

import sys
import pandas as pd
import anndata as ad
import warnings

warnings.filterwarnings("ignore")

## VIASH START
par = {
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "de_train_h5ad": "resources/neurips-2023-data/de_train.h5ad",
    "layer": "sign_log10_pval",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "output": "output.parquet",
    "reps": 2,
}
meta = {"resources_dir": "src/task/methods/nn_retraining_with_pseudolabels"}
## VIASH END

# load helper functions in notebooks
sys.path.append(meta["resources_dir"])

from anndata_to_dataframe import anndata_to_dataframe
from notebook_264 import run_notebook_264
from notebook_266 import run_notebook_266

# load train data
de_train_h5ad = ad.read_h5ad(par["de_train_h5ad"])
train_df = anndata_to_dataframe(de_train_h5ad, par["layer"])

train_df = train_df.sample(frac=1.0, random_state=42)
train_df = train_df.reset_index(drop=True)

# load test data
test_df = pd.read_csv(par["id_map"])

# determine gene names
gene_names = list(de_train_h5ad.var_names)

# clean up train data
train_df = train_df.loc[:, ["cell_type", "sm_name"] + gene_names]

# run notebook 264
pseudolabel = run_notebook_264(train_df, test_df, gene_names, par["reps"])

# add metadata
pseudolabel = pd.concat(
    [test_df[["cell_type", "sm_name"]], pseudolabel.loc[:, gene_names]], axis=1
)

# run notebook 266
df = run_notebook_266(train_df, test_df, pseudolabel, gene_names, par["reps"])

df.to_parquet(par["output"])
