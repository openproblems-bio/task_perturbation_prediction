import anndata as ad
import numpy as np

## VIASH START
par = {
    "de_test_h5ad": "resources/neurips-2023-data/de_test.h5ad",
    "de_test_layer": "sign_log10_pval",
    "prediction": "resources/neurips-2023-data/prediction.h5ad",
    "prediction_layer": "prediction",
    "method_id": "foo",
    "output": "output.h5ad",
}
## VIASH END

print("Load data", flush=True)
de_test = ad.read_h5ad(par["de_test_h5ad"])
prediction = ad.read_h5ad(par["prediction"])

print("Select genes", flush=True)
genes = list(de_test.var_names)
de_test_X = de_test.layers[par["de_test_layer"]]
prediction_X = prediction.layers[par["prediction_layer"]]

# transform to ranks
de_test_r = np.argsort(de_test_X, axis=1).argsort(axis=1)
prediction_r = np.argsort(prediction_X, axis=1).argsort(axis=1)

print("Calculate mean pearson", flush=True)
mean_pearson = np.mean(
    [np.corrcoef(de_test_X[i,], prediction_X[i,])[0, 1] for i in range(de_test_X.shape[0])]
)
mean_spearman = np.mean(
    [np.corrcoef(de_test_r[i,], prediction_r[i,])[0, 1] for i in range(de_test_X.shape[0])]
)

print("Create output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": de_test.uns["dataset_id"],
        "method_id": prediction.uns["method_id"],
        "metric_ids": ["mean_pearson", "mean_spearman"],
        "metric_values": [mean_pearson, mean_spearman]
    }
)

print("Write output", flush=True)
output.write_h5ad(par["output"], compression="gzip")