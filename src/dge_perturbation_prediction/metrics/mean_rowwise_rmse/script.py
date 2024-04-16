import pandas as pd
import json

## VIASH START
par = {
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "prediction": "resources/neurips-2023-data/output_rf.csv",
  "output": "resources/neurips-2023-data/score_rf.json",
}
## VIASH END

de_test = pd.read_parquet(par["de_test"]).set_index('id')
prediction = pd.read_csv(par["prediction"]).set_index('id')

# subset to the same columns
genes = list(set(de_test.columns) - set(["cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control"]))
de_test = de_test.loc[:, genes]
prediction = prediction[genes]

# compute mean_rowwise_rmse
mean_rowwise_rmse = 0
for i in de_test.index:
    mean_rowwise_rmse += ((de_test.iloc[i] - prediction.iloc[i])**2).mean()

mean_rowwise_rmse /= de_test.shape[0]

output = {
    "mean_rowwise_rmse": mean_rowwise_rmse
}

# write to file
with open(par["output"], 'w') as f:
    json.dump(output, f)
