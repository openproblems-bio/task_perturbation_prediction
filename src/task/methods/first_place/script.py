from argparse import Namespace
import pandas as pd
import numpy as np
import json
import sys
import time
import os
import fastparquet, anndata
# import ipdb; ipdb.set_trace()
## VIASH START
par = {
  "de_train": "resources/neurips-2023-data/de_train.parquet",
  "de_test": "resources/neurips-2023-data/de_test.parquet",
  "id_map": "resources/neurips-2023-data/id_map.csv",
  "train_data_aug_dir": "resources/neurips-2023-data/prepared_data/",
  "model_dir": "resources/neurips-2023-data/trained_models/",
  "logs_dir": "resources/neurips-2023-data/results/",
  "submission": "resources/neurips-2023-data/sample_submission.csv",
#   "output_dir": "resources/neurips-2023-data/output_rf.parquet",
}
## VIASH END

## CUSTOM CODE START
# import helper function
# meta = {"resources_dir": "src/task/methods/first_place"}
sys.path.append(meta['resources_dir'])

# print(f"META PATH:{meta['resources_dir']}")
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(f"THIS IS THE PATH: {os.getcwd()}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

from helper_script import seed_everything, combine_features, train_validate
from helper_script import one_hot_encode, save_ChemBERTa_features
from helper_script import combine_features, load_trained_models, average_prediction, weighted_average_prediction


def prepare_data():
    seed_everything()
    # with open("./par.json") as file:
    #     par = json.load(file)
    ## Read data
    print("\nPreparing data...")
    print(par)
    de_train = pd.read_parquet(par["de_train"])
    de_train = de_train.drop(columns=['split'])
    id_map = pd.read_csv(par["id_map"])
    ## Create data augmentation
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()
    cell_types = de_cell_type.groupby('cell_type').quantile(0.1).reset_index()['cell_type'] # This is just to get cell types in the right order for the next line
    quantiles_cell_type = pd.concat([pd.DataFrame(cell_types)]+[de_cell_type.groupby('cell_type')[col]\
    .quantile([0.25, 0.50, 0.75], interpolation='linear').unstack().reset_index(drop=True) for col in list(de_train.columns)[5:]], axis=1)
    ## Save data augmentation features
    print(par["train_data_aug_dir"])
    if not os.path.exists(par["train_data_aug_dir"]):
        os.makedirs(par["train_data_aug_dir"], exist_ok=True)

    # if not os.path.exists(par["model_dir"]):
    #     print("There wasnt model_dir")
    #     os.makedirs(par["model_dir"], exist_ok=True)
    mean_cell_type.to_csv(f'{par["train_data_aug_dir"]}mean_cell_type.csv', index=False)
    std_cell_type.to_csv(f'{par["train_data_aug_dir"]}std_cell_type.csv', index=False)
    mean_sm_name.to_csv(f'{par["train_data_aug_dir"]}mean_sm_name.csv', index=False)
    std_sm_name.to_csv(f'{par["train_data_aug_dir"]}std_sm_name.csv', index=False)
    quantiles_cell_type.to_csv(f'{par["train_data_aug_dir"]}quantiles_cell_type.csv', index=False)
    ## Create one hot encoding features
    one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], out_dir=par["train_data_aug_dir"])
    ## Prepare ChemBERTa features
    save_ChemBERTa_features(de_train["SMILES"].tolist(), out_dir=par["train_data_aug_dir"], on_train_data=True)
    sm_name2smiles = {smname:smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
    test_smiles = list(map(sm_name2smiles.get, id_map['sm_name'].values))
    save_ChemBERTa_features(test_smiles, out_dir=par["train_data_aug_dir"], on_train_data=False)
    print("### Done.")

def train():
    # with open("./par.json") as file:
    #     par = json.load(file)
    # with open("./config/train_config.json") as file:
    #     train_config = json.load(file)
        
    train_config = {"LEARNING_RATES": [0.001, 0.001, 0.0003], "CLIP_VALUES": [5.0, 1.0, 1.0], "EPOCHS": 1, "KF_N_SPLITS": 2} #TODO change epochs to 250
    print("\nRead data and build features...")
    de_train = pd.read_parquet(par["de_train"])
    de_train = de_train.drop(columns=['split'])
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(f'{par["train_data_aug_dir"]}one_hot_train.npy'))
    y = de_train.drop(columns=ylist)
    mean_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{par["train_data_aug_dir"]}quantiles_cell_type.csv')
    train_chem_feat = np.load(f'{par["train_data_aug_dir"]}chemberta_train.npy')
    train_chem_feat_mean = np.load(f'{par["train_data_aug_dir"]}chemberta_train_mean.npy')
    X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [train_chem_feat,train_chem_feat_mean], de_train, one_hot_train, quantiles_df)
    ## Start training
    print(f"Mean cell type:{mean_cell_type.shape}")
    print(f"Std cell type:{std_cell_type.shape}")
    print(f"Mean_sm_name:{mean_sm_name.shape}")
    print(f"Std_sm_name:{std_sm_name.shape}")
    print(f"Quantiles_df:{quantiles_df.shape}")
    print(f"Train_chem_feat:{train_chem_feat.shape}")
    print(f"Train_chem_feat_mean:{train_chem_feat_mean.shape}")
    print(f"X_vec:{X_vec.shape}")
    print(f"X_vec_light:{X_vec_light.shape}")
    print(f"X_vec_heavy:{X_vec_heavy.shape}")
    print(f"de_train:{de_train.shape}")
    print(f"Y:{y.shape}")
    cell_types_sm_names = de_train[['cell_type', 'sm_name']]
    print("\nTraining starting...")
    train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, train_config)
    print("\nDone.")


def read_data(par):
    de_train = pd.read_parquet(par["de_train"])
    de_train = de_train.drop(columns=['split'])
    id_map = pd.read_csv(par["id_map"])
    print(par)
    # sample_submission = pd.read_csv(par["submission"], index_col='id')
    return de_train, id_map#, sample_submission

def predict():
    # with open("./par.json") as file:
    #     par = json.load(file)
    # /home/ttunjic/Code/task-dge-perturbation-prediction/resources/neurips-2023-data/sample_submission.csv
    # /home/ttunjic/Code/task-dge-perturbation-prediction/resources/neurips-2023-data/sample_submission.csv
    test_config = {"MODEL_COEFS": [0.29, 0.33, 0.38], "FOLD_COEFS": [0.25, 0.15, 0.2, 0.15, 0.25], "KF_N_SPLITS": 5}
    ## Read train, test and sample submission data # train data is needed for columns
    print("\nReading data...")
    # de_train, id_map, sample_submission = read_data(par)
    de_train, id_map = read_data(par)
    
    ## Build input features
    mean_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{par["train_data_aug_dir"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{par["train_data_aug_dir"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{par["train_data_aug_dir"]}quantiles_cell_type.csv')
    test_chem_feat = np.load(f'{par["train_data_aug_dir"]}chemberta_test.npy')
    test_chem_feat_mean = np.load(f'{par["train_data_aug_dir"]}chemberta_test_mean.npy')
    one_hot_test = pd.DataFrame(np.load(f'{par["train_data_aug_dir"]}one_hot_test.npy'))
    
    test_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [test_chem_feat,test_chem_feat_mean], id_map, one_hot_test, quantiles_df)
    
    ## Load trained models
    print("\nLoading trained models...")
    trained_models = load_trained_models(path=f'{par["model_dir"]}')
    fold_weights = test_config["FOLD_COEFS"] if test_config["KF_N_SPLITS"] == 5 else [1.0/test_config["KF_N_SPLITS"]]*test_config["KF_N_SPLITS"]
    ## Start predictions
    print("\nStarting predictions...")
    t0 = time.time()
    pred1 = average_prediction(test_vec_light, trained_models['light'])
    pred2 = weighted_average_prediction(test_vec_light, trained_models['light'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    pred3 = average_prediction(test_vec, trained_models['initial'])
    pred4 = weighted_average_prediction(test_vec, trained_models['initial'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    
    # pred5 = average_prediction(test_vec_heavy, trained_models['heavy'])
    # pred6 = weighted_average_prediction(test_vec_heavy, trained_models['heavy'],\
    #                                 model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    t1 = time.time()
    print("Prediction time: ", t1-t0, " seconds")
    print("\nEnsembling predictions and writing to file...")
    col = list(de_train.columns[5:])

    df_sub_ix = id_map.set_index(["cell_type", "sm_name"])
    submission = pd.DataFrame(index=df_sub_ix.index, columns=de_train.columns)
    # submission = sample_submission.copy()
    
    submission[col] = 0.23*pred1 + 0.15*pred2 + 0.18*pred3 + 0.15*pred4 #+ 0.15*pred5 + 0.14*pred6
    df1 = submission.copy()
    
    submission[col] = 0.13*pred1 + 0.15*pred2 + 0.23*pred3 + 0.15*pred4 #+ 0.20*pred5 + 0.14*pred6
    df2 = submission.copy()
    
    submission[col] = 0.17*pred1 + 0.16*pred2 + 0.17*pred3 + 0.16*pred4 #+ 0.18*pred5 + 0.16*pred6
    df3 = submission.copy()
    df_sub = 0.34*df1 + 0.33*df2 + 0.33*df3 # Final ensembling
    # if not os.path.exists(par["submission"]):
    #     os.mkdir(par["submission"])
    df_sub = df_sub.reset_index(drop=True)
    # df_sub.to_csv(f'{par["submission"]}')
    print(df_sub.head())
    # df_sub.to_parquet(par["output"])
    fastparquet.write(par['output'], df_sub)
    print("\nDone.")

prepare_data()
train()
predict()
