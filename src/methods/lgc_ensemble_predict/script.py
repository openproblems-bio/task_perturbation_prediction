import sys
import torch
import json
import anndata as ad
import numpy as np
import pandas as pd
import time
if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print('using device: cpu', flush=True)

## VIASH START
par = {
    "id_map": "resources/datasets/neurips-2023-data/id_map.csv",
    "train_data_aug_dir": "output/train_data_aug_dir",
    "model_files": [
        "output/models/pytorch_lstm_light_fold0.pt"
    ],
    "prediction": "output/prediction.h5ad"
}
## VIASH END

print(f"par: {par}", flush=True)

# import helper functions
sys.path.append(meta['resources_dir'])
from helper_functions import combine_features, lazy_load_trained_models, average_prediction, weighted_average_prediction

print("\nReading data...")
train_config = json.load(open(f'{par["train_data_aug_dir"]}/config.json'))
test_config = {
    "MODEL_COEFS": [0.29, 0.33, 0.38],
    "FOLD_COEFS": [0.25, 0.15, 0.2, 0.15, 0.25],
    "KF_N_SPLITS": train_config["KF_N_SPLITS"]
}

## Read train, test and sample submission data # train data is needed for columns
print("\nReading data...")

id_map = pd.read_csv(par["id_map"])

with open(f'{par["train_data_aug_dir"]}/gene_names.json', 'r') as f:
    gene_names = json.load(f)

## Build input features
mean_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}/mean_cell_type.csv')
std_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}/std_cell_type.csv')
mean_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}/mean_sm_name.csv')
std_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}/std_sm_name.csv')
quantiles_df = pd.read_csv(f'{par["train_data_aug_dir"]}/quantiles_cell_type.csv')
test_chem_feat = np.load(f'{par["train_data_aug_dir"]}/chemberta_test.npy')
test_chem_feat_mean = np.load(f'{par["train_data_aug_dir"]}/chemberta_test_mean.npy')
one_hot_test = pd.DataFrame(np.load(f'{par["train_data_aug_dir"]}/one_hot_test.npy'))

test_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
            [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
test_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
test_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                [test_chem_feat,test_chem_feat_mean], id_map, one_hot_test, quantiles_df)

## Load trained models
print("\nLoading trained models...")
trained_models = lazy_load_trained_models(
    par["train_data_aug_dir"],
    par["model_files"],
    kf_n_splits=test_config["KF_N_SPLITS"]
)
fold_weights = test_config["FOLD_COEFS"] \
    if test_config["KF_N_SPLITS"] == len(test_config["FOLD_COEFS"]) \
    else [1.0/test_config["KF_N_SPLITS"]]*test_config["KF_N_SPLITS"]

## Start predictions
print("\nStarting predictions...")
t0 = time.time()
if "light" in train_config["SCHEMES"]:
    print("\nPredicting light models...")
    pred1 = average_prediction(test_vec_light, trained_models['light'])
    pred2 = weighted_average_prediction(test_vec_light, trained_models['light'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
if "initial" in train_config["SCHEMES"]:
    print("\nPredicting initial models...")
    pred3 = average_prediction(test_vec, trained_models['initial'])
    pred4 = weighted_average_prediction(test_vec, trained_models['initial'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
if "heavy" in train_config["SCHEMES"]:
    print("\nPredicting heavy models...")
    pred5 = average_prediction(test_vec_heavy, trained_models['heavy'])
    pred6 = weighted_average_prediction(test_vec_heavy, trained_models['heavy'],\
                                    model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
t1 = time.time()
print("Prediction time: ", t1-t0, " seconds")
print("\nEnsembling predictions and writing to file...")

df_sub_ix = id_map.set_index(["cell_type", "sm_name"])
submission = pd.DataFrame(index=df_sub_ix.index, columns=gene_names)

submission[gene_names] = 0
weight = 0
if "light" in train_config["SCHEMES"]:
    submission[gene_names] += 0.23*pred1 + 0.15*pred2
    weight += 0.23 + 0.15
if "initial" in train_config["SCHEMES"]:
    submission[gene_names] += 0.18*pred3 + 0.15*pred4
    weight += 0.18 + 0.15
if "heavy" in train_config["SCHEMES"]:
    submission[gene_names] += 0.15*pred5 + 0.14*pred6
    weight += 0.15 + 0.14

submission[gene_names] /= weight
df1 = submission.copy()

submission[gene_names] = 0
weight = 0
if "light" in train_config["SCHEMES"]:
    submission[gene_names] += 0.13*pred1 + 0.15*pred2
    weight += 0.13 + 0.15
if "initial" in train_config["SCHEMES"]:
    submission[gene_names] += 0.23*pred3 + 0.15*pred4
    weight += 0.23 + 0.15
if "heavy" in train_config["SCHEMES"]:
    submission[gene_names] += 0.20*pred5 + 0.16*pred6
    weight += 0.20 + 0.16

submission[gene_names] /= weight
df2 = submission.copy()

submission[gene_names] = 0
weight = 0
if "light" in train_config["SCHEMES"]:
    submission[gene_names] += 0.17*pred1 + 0.16*pred2
    weight += 0.17 + 0.16
if "initial" in train_config["SCHEMES"]:
    submission[gene_names] += 0.17*pred3 + 0.16*pred4
    weight += 0.17 + 0.16
if "heavy" in train_config["SCHEMES"]:
    submission[gene_names] += 0.18*pred5 + 0.16*pred6
    weight += 0.18 + 0.16

submission[gene_names] /= weight
df3 = submission.copy()

df_sub = 0.34*df1 + 0.33*df2 + 0.33*df3 # Final ensembling
df_sub.reset_index(drop=True, inplace=True)

# write output
method_id = meta["name"].replace("_predict", "")
output = ad.AnnData(
    layers={"prediction": df_sub.to_numpy()},
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
        "dataset_id": train_config["DATASET_ID"],
        "method_id": method_id
    }
)
print(output)
output.write_h5ad(par["output"], compression="gzip")
print("\nDone.")