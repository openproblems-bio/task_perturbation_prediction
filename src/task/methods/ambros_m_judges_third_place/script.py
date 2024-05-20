# This script is based on an IPython notebook:
# https://github.com/Ambros-M/Single-Cell-Perturbations-2023/blob/main/notebooks/scp-26-py-boost-recommender-system-and-et.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style
from numpy.polynomial import Polynomial
from scipy.stats import norm, skew, kurtosis
import scipy
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from itertools import combinations
import cupy as cp
from py_boost import GradientBoosting
import warnings
import zipfile

warnings.simplefilter('ignore', FutureWarning)
np.set_printoptions(linewidth=195, edgeitems=3)
pd.set_option("min_rows", 6)

## VIASH START
par = dict(
    de_train = "resources/neurips-2023-data/de_train.parquet",
    id_map = "resources/neurips-2023-data/id_map.csv",
    output = "output.parquet",
)
## VIASH END


## Utility functions

def mean_rowwise_rmse(y_true, y_pred):
    """Competition metric
    
    Calling convention like in sklearn.metrics
    """
    mrrmse = np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()
    return mrrmse

def de_to_t_score(de):
    """Convert log10pvalues to t-scores
    
    Parameter:
    de: array or DataFrame of log10pvalues
    
    Return value:
    t_score: array or DataFrame of t-scores
    """
    p_value = 10 ** (-np.abs(de))
#     return - scipy.stats.t.ppf(p_value / 2, df=420) * np.sign(de)
    return - norm.ppf(p_value / 2) * np.sign(de)

def t_score_to_de(t_score):
    """Convert t-scores to log10pvalues (inverse of de_to_t_score)
    
    Parameter:
    t_score: array or DataFrame of t-scores
    
    Return value:
    de: array or DataFrame of log10pvalues
    """
#     p_value = scipy.stats.t.cdf(- np.abs(t_score), df=420) * 2
    p_value = norm.cdf(- np.abs(t_score)) * 2
    p_value = p_value.clip(1e-180, None)
    return - np.log10(p_value) * np.sign(t_score)

## Loading data

de_train = pd.read_parquet(par['de_train'])
id_map = pd.read_csv(par['id_map'], index_col = 0)
# display(id_map)

# 18211 genes
genes = de_train.columns[6:] 
de_train_indexed = de_train.set_index(['cell_type', 'sm_name'])[genes]

# All 146 sm_names
sm_names = sorted(de_train.sm_name.unique())
# Determine the 17 compounds (including the two control compounds) with data for almost all cell types
train_sm_names = de_train.query("cell_type == 'B cells'").sm_name.sort_values().values
# The other 129 sm_names
test_sm_names = [sm for sm in sm_names if sm not in train_sm_names]
# The three control sm_names
controls3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']

# All 6 cell types
cell_types = ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells', 'B cells', 'Myeloid cells']
# Determine the 4 cell types with data for almost all compounds
# ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells']
train_cell_types = de_train.query("sm_name == 'Vorinostat'").cell_type.sort_values().values
# The other 2 cell types: ['B cells', 'Myeloid cells']
test_cell_types = [ct for ct in cell_types if ct not in train_cell_types]
        
with zipfile.ZipFile(par['train_obs_zip']) as z:
    with z.open('train_obs.csv') as f:
        adata_obs = pd.read_csv(f)

# Cell counts
cell_count = adata_obs.groupby(['cell_type', 'sm_name']).size()
avg_cell_count = cell_count[~cell_count.index.get_level_values('sm_name').isin(controls3)].groupby('cell_type').mean()

# Cell type ratios (extrapolated from 17 train_sm_names)
temp = adata_obs.groupby(['cell_type', 'sm_name']).size().unstack().loc[cell_types]
cell_type_ratio = temp[list(train_sm_names) + ['Dimethyl Sulfoxide']].sum(axis=1)
cell_type_ratio /= cell_type_ratio.sum()

## Model fitting functions

def fit_predict_py_boost(de_tr, id_map):
    """Fit the model and predict.
    
    Parameters:
    de_tr: training dataframe of shape (n_samples, 18211), MultiIndex (cell_type, sm_name)
    id_map: two-column dataframe indicating the validation or test samples (cell_type, sm_name)
    
    Returns:
    de_pred: prediction dataframe of shape (n_samples, 18211), double index matching id_map
    
    https://pypi.org/project/py-boost/
    """
    # Hyperparameters
    n_components = 50
    max_depth = 10
    ntrees = 1000
    subsample = 1
    colsample = 0.2
    lr = 0.01

    # Determine the training cell types (3 or 4)
    cell_types_tr = de_tr.index[de_tr.index.get_level_values('sm_name') == 'Oxybenzone'].get_level_values('cell_type')

    X_train_categorical = de_tr.index.to_frame()

    #  Dimension reduction
    reducer = PCA(n_components=n_components, random_state=1)
    Yt_train_red = reducer.fit_transform(de_to_t_score(de_tr))
    Yt_train_red = pd.DataFrame(Yt_train_red, index=de_tr.index) # no specific column names

    # Target-encode the two categorical features column-wise
    # We encode the features with the t-score rather than the log10pvalue
    # ct_mean has shape (6, 18211) and contains the means of 13 or 14 values each
    ct_mean = Yt_train_red[X_train_categorical['sm_name'].isin(train_sm_names)].groupby('cell_type').mean()
    # sm_mean has shape (143, 18211) and contains the means of 3 or 4 values each
    sm_mean = Yt_train_red[X_train_categorical['cell_type'].isin(cell_types_tr)].groupby('sm_name').mean()
    X_train_encoded = np.hstack([ct_mean.reindex(X_train_categorical['cell_type']).values,
                                 sm_mean.reindex(X_train_categorical['sm_name']).values])
    X_test_encoded =  np.hstack([ct_mean.reindex(id_map['cell_type']).values,
                                 sm_mean.reindex(id_map['sm_name']).values])
    
    # Fit the model
    model = GradientBoosting('mse',
                             ntrees=ntrees, 
                             lr=lr, 
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample=colsample,
                             min_data_in_leaf=1,
                             min_gain_to_split=0,
                             verbose=10000)           
    model.fit(X_train_encoded, Yt_train_red)

    # Predict
    Y_test_pred_red = model.predict(X_test_encoded)
    Y_test_pred = t_score_to_de(reducer.inverse_transform(Y_test_pred_red))
    de_pred = pd.DataFrame(Y_test_pred, index=pd.MultiIndex.from_frame(id_map), columns=genes)

    return de_pred

def fit_predict_ridge_recommender(de_tr, id_map):
    """Fit the model and predict.
    
    Parameters:
    de_tr: training dataframe of shape (n_samples, 18211), MultiIndex (cell_type, sm_name)
    id_map: two-column dataframe indicating the validation or test samples (cell_type, sm_name)
    
    Returns:
    de_pred: prediction dataframes of shape (n_samples, 18211), double index matching id_map
             If a compound occurs in id_map but not in de_tr, the corresponding row
             of de_pred will be filled with np.nan
    """
    # Hyperparameters
    n_components_in, n_components_out = 7, 70
    factor_ct = 0.34
    factor_sm = 0.66
    
    # Determine the training cell types (3 or 4)
    cell_types_tr = de_tr.index[de_tr.index.get_level_values('sm_name') == 'Oxybenzone'].get_level_values('cell_type')
    
    # Denoising and dimensionality reduction  
    reducer_t = make_pipeline(StandardScaler(), PCA(n_components=n_components_out, svd_solver='full', random_state=1))
    Yt_train_red = reducer_t.fit_transform(de_to_t_score(de_tr))
    Yt_train_red = pd.DataFrame(Yt_train_red, index=de_tr.index) # no specific column names
    
    X_train_categorical = Yt_train_red.index.to_frame()

    # Even more dimensionality reduction
    Yt_train_red_in = Yt_train_red.iloc[:, :n_components_in]
    
    # ct-based model
    # The model fits a ridge regression to all cell types which have been treated with
    # the compound to be predicted
    model_ct = make_pipeline(StandardScaler(), Ridge(3e3))
    X_train = Yt_train_red_in[X_train_categorical.sm_name.isin(train_sm_names)]
    X_train = X_train.unstack('sm_name') # 6 rows, index is cell_type
    X_train.fillna(value=X_train.mean(), inplace=True)
    X_test = X_train.reindex(id_map['cell_type'])
    temp_list = []
    for i in range(len(id_map)):
        # Y_train has 3 or 4 rows and n_components_out columns
        Y_train = Yt_train_red[X_train_categorical.sm_name == id_map['sm_name'].iloc[i]]
        if len(Y_train) > 0:
            model_ct.fit(X_train.reindex(Y_train.index.get_level_values('cell_type')), Y_train)
            temp_list.append(model_ct.predict(X_test.iloc[[i]]))
        else: # compound has been dropped as outlier
            temp_list.append(np.full((1, Y_train.shape[1]), np.nan))
    Y_pred_ct = np.vstack(temp_list)

    # sm-based model
    # The model fits a ridge regression to all (at most 17) compounds which have been applied to
    # the cell type to be predicted
    model_sm = make_pipeline(StandardScaler(), Ridge(1e1))
    X_train = Yt_train_red_in[X_train_categorical.cell_type.isin(cell_types_tr)]
    X_train = X_train.unstack('cell_type') # 147 rows, index is sm_name
    X_train.fillna(value=X_train.mean(), inplace=True)
    X_test = X_train.reindex(id_map['sm_name'])
    temp_list = []
    for i in range(len(id_map)):
        if ~X_test.iloc[i].isna().any():
            # Y_train has 15 or 17 rows and n_components_out columns
            Y_train = Yt_train_red[X_train_categorical.cell_type == id_map['cell_type'].iloc[i]]
            model_sm.fit(X_train.reindex(Y_train.index.get_level_values('sm_name')), Y_train)
            temp_list.append(model_sm.predict(X_test.iloc[[i]]))
        else: # compound has been dropped as outlier
            temp_list.append(np.full((1, Y_train.shape[1]), np.nan))
    Y_pred_sm = np.vstack(temp_list)

    # Bring the two predictions together
    Y_test_pred_red = factor_ct * Y_pred_ct + factor_sm * Y_pred_sm
    Y_test_pred = t_score_to_de(reducer_t.inverse_transform(Y_test_pred_red))
    de_pred = pd.DataFrame(Y_test_pred, index=pd.MultiIndex.from_frame(id_map), columns=genes)
    return de_pred

def fit_predict_knn_recommender(de_tr, id_map):
    """Fit the model and predict.
    
    Parameters:
    de_tr: training dataframe of shape (n_samples, 18211), MultiIndex (cell_type, sm_name)
    id_map: two-column dataframe indicating the validation or test samples (cell_type, sm_name)
    
    Returns:
    de_pred: prediction dataframes of shape (n_samples, 18211), double index matching id_map
             If a compound occurs in id_map but not in de_tr, the corresponding row
             of de_pred will be filled with np.nan
    """
    # Hyperparameters
    n_components_in, n_components_out = 7, 70
    factor_ct = 0.32
    factor_sm = 0.74

    # Determine the training cell types (3 or 4)
    cell_types_tr = de_tr.index[de_tr.index.get_level_values('sm_name') == 'Oxybenzone'].get_level_values('cell_type')
    
    # Denoising and dimensionality reduction  
    reducer_t = make_pipeline(StandardScaler(), PCA(n_components=n_components_out, svd_solver='full', random_state=1))
    Yt_train_red = reducer_t.fit_transform(de_to_t_score(de_tr))
    Yt_train_red = pd.DataFrame(Yt_train_red, index=de_tr.index) # no specific column names
    
    # Data augmentation
    # We use two kinds of data augmentation:
    # 1. Scaled t-scores (t-score is a quotient of log-fold-change and standard deviation;
    #    if the variance changes, the t-scores are scaled)
    # 2. Mixture of two compounds
    for ct1 in cell_types_tr:
        for factor in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]:
            a = Yt_train_red.query("cell_type == @ct1").reset_index('cell_type', drop=True) * factor
            a = pd.concat([a], keys=[f"{ct1}*{factor}"], names=['cell_type'])
            Yt_train_red = pd.concat([Yt_train_red, a])

    for sm1 in train_sm_names:
        for factor in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]:
            a = Yt_train_red.query("sm_name == @sm1").reset_index('sm_name', drop=True) * factor
            a = pd.concat([a], keys=[f"{sm1}*{factor}"], names=['sm_name'])
            a = a.reorder_levels(['cell_type', 'sm_name'])
            Yt_train_red = pd.concat([Yt_train_red, a])

    for sm1, sm2 in combinations(train_sm_names, 2):
        a = (2 * Yt_train_red.query("sm_name == @sm1").reset_index('sm_name', drop=True)
             + Yt_train_red.query("sm_name == @sm2").reset_index('sm_name', drop=True)) / 3
        a.dropna(inplace=True)
        a = pd.concat([a], keys=[f"{sm1}+{sm2} a"], names=['sm_name'])
        a = a.reorder_levels(['cell_type', 'sm_name'])
        b = (Yt_train_red.query("sm_name == @sm1").reset_index('sm_name', drop=True)
             + 2 * Yt_train_red.query("sm_name == @sm2").reset_index('sm_name', drop=True)) / 3
        b.dropna(inplace=True)
        b = pd.concat([b], keys=[f"{sm1}+{sm2} b"], names=['sm_name'])
        b = b.reorder_levels(['cell_type', 'sm_name'])
        Yt_train_red = pd.concat([Yt_train_red, a, b])
        
    X_train_categorical = Yt_train_red.index.to_frame()

    # Even more dimensionality reduction
    Yt_train_red_in = Yt_train_red.iloc[:, :n_components_in]
    
    # ct-based model
    # The model finds similar cell types which have been treated with
    # the compound to be predicted and averages their t scores
    model_ct = KNeighborsRegressor(n_neighbors=7, weights='distance')
    X_train = Yt_train_red_in[X_train_categorical.sm_name.isin(train_sm_names)]
    X_train = X_train.unstack('sm_name') # 6 rows * 51 columns (without augmentation), index is cell_type
    X_train.fillna(value=X_train.mean(), inplace=True)
    X_test = X_train.reindex(id_map['cell_type'])
    temp_list = []
    for i in range(len(id_map)):
        # Find similar cell types which have a rating for the given sm_name
        Y_train = Yt_train_red[X_train_categorical.sm_name == id_map['sm_name'].iloc[i]]
        if len(Y_train) > 0:
            model_ct.fit(X_train.reindex(Y_train.index.get_level_values('cell_type')), Y_train)
            temp_list.append(model_ct.predict(X_test.iloc[[i]]))
        else: # compound has been dropped as outlier
            temp_list.append(np.full((1, Y_train.shape[1]), np.nan))
    Y_pred_ct = np.vstack(temp_list)

    # sm-based model
    # The model finds similar sm_names which have been measured with
    # the cell type to be predicted and averages their t scores
    model_sm = KNeighborsRegressor(n_neighbors=9, weights='distance', p=1)
    X_train = Yt_train_red_in[X_train_categorical.cell_type.isin(cell_types_tr)]
    X_train = X_train.unstack('cell_type') # 146 rows * 9 columns (without augmentation), index is sm_name
    X_train.fillna(value=X_train.mean(), inplace=True)
    X_test = X_train.reindex(id_map['sm_name'])
    temp_list = []
    for i in range(len(id_map)):
        if ~X_test.iloc[i].isna().any():
            # Find similar sm_names which have a rating for the given cell type
            # Y_train has 15 or 17 rows if there is no data augmentation
            # Y_train has 153 rows if all sm_name pairs are augmented
            Y_train = Yt_train_red[X_train_categorical.cell_type == id_map['cell_type'].iloc[i]]
            model_sm.fit(X_train.reindex(Y_train.index.get_level_values('sm_name')), Y_train)
            temp_list.append(model_sm.predict(X_test.iloc[[i]]))
        else: # compound has been dropped as outlier
            temp_list.append(np.full((1, Y_train.shape[1]), np.nan))
    Y_pred_sm = np.vstack(temp_list)

    # Bring the two predictions together
    Y_test_pred_red = factor_ct * Y_pred_ct + factor_sm * Y_pred_sm
    Y_test_pred = t_score_to_de(reducer_t.inverse_transform(Y_test_pred_red))
    de_pred = pd.DataFrame(Y_test_pred, index=pd.MultiIndex.from_frame(id_map), columns=genes)
    return de_pred

def fit_predict_extratrees(de_tr, id_map):
    """Fit the model and predict.
    
    Parameters:
    de_tr: training dataframe of shape (n_samples, 18211), MultiIndex (cell_type, sm_name)
    id_map: two-column dataframe indicating the validation or test samples (cell_type, sm_name)
    
    Returns:
    de_pred: prediction dataframes of shape (n_samples, 18211), double index matching id_map
             If a compound occurs in id_map but not in de_tr, the corresponding row
             of de_pred will be filled with np.nan
    """
    # Hyperparameters
    n_components_in, n_components_out = 35, 200
    max_features = 0.2
    n_trees = 2000
    
    # Determine the training cell types (3 or 4)
    cell_types_tr = de_tr.index[de_tr.index.get_level_values('sm_name') == 'Oxybenzone'].get_level_values('cell_type')
    
    X_train_categorical = de_tr.index.to_frame()
    
    # Denoising and dimensionality reduction  
    reducer_t = make_pipeline(StandardScaler(), PCA(n_components=n_components_out, svd_solver='full', random_state=1))
    Yt_train_red = reducer_t.fit_transform(de_to_t_score(de_tr))
    Yt_train_red = pd.DataFrame(Yt_train_red, index=de_tr.index) # no specific column names

    # Even more dimensionality reduction
    Yt_train_red_in = Yt_train_red.iloc[:, :n_components_in]
    
    # Target-encode the two categorical features without smoothing
    # We encode the features with the t-score rather than the log10pvalue
    ct_mean = Yt_train_red_in[X_train_categorical['sm_name'].isin(train_sm_names)].groupby('cell_type').mean() # shape (6, n_components), means of 13 or 14 values each
    sm_mean = Yt_train_red_in[X_train_categorical['cell_type'].isin(cell_types_tr)].groupby('sm_name').mean() # shape (143, n_components), means of 3 or 4 values each
    X_train_encoded = np.hstack([ct_mean.reindex(X_train_categorical['cell_type']).values,
                                 sm_mean.reindex(X_train_categorical['sm_name']).values,
                                 sm_mean.reindex(X_train_categorical['sm_name']).values * np.sqrt(cell_type_ratio.reindex(X_train_categorical['cell_type']).values.reshape(-1, 1))
                                ])
    X_test_encoded =  np.hstack([ct_mean.reindex(id_map['cell_type']).values,
                                 sm_mean.reindex(id_map['sm_name']).values,
                                 sm_mean.reindex(id_map['sm_name']).values * np.sqrt(cell_type_ratio.reindex(id_map['cell_type']).values.reshape(-1, 1))
                                ])

    # Train the model
    # The model is trained to predict the PCA-transformed t-scores.
    model = ExtraTreesRegressor(n_estimators=n_trees, max_features=max_features, random_state=1)
    model.fit(X_train_encoded, Yt_train_red)

    # Predict
    X_test_encoded = pd.DataFrame(X_test_encoded, index=id_map)
    X_test_encoded.dropna(inplace=True)
    Y_test_pred_red = model.predict(X_test_encoded.values)
    Y_test_pred = t_score_to_de(reducer_t.inverse_transform(Y_test_pred_red))
    de_pred = pd.DataFrame(Y_test_pred, index=X_test_encoded.index, columns=genes)
    de_pred = de_pred.reindex(id_map)
    return de_pred


def cross_val_log10pvalue(predictor, noise=0):
    """Cross-validate a machine-learning model
    
    Parameters:
    predictor: function which takes two parameters
        (training data and id_map) and returns
        the predictions
    noise: standard deviation of noise to be added to the t-scores
        
    Globals:
    de_oof_dict: dictionary into which the oof predictions are inserted
    mrrmse_noise_list: list into which the noise level and the oof score are inserted
    removed_compounds: list of outlier compounds
    """
    mrrmse_list = []
    t_oof_list, de_oof_list = [], []
    for fold, val_cell_type in enumerate(train_cell_types):
        # Split the data into training and validation
        # mask_va: 127 or 129 validation rows per fold, total 514 in four folds
        mask_va = ((de_train['cell_type'] == val_cell_type) &
                   ~de_train['sm_name'].isin(list(train_sm_names) + ['Dimethyl Sulfoxide']))
        if mask_va.sum() == 0: continue
        mask_va = mask_va.values
        # mask_tr: 485 or 487 training rows
        mask_tr = ~mask_va
        
        de_tr = de_train_indexed[mask_tr] # shape (48x, 18211), double index
        de_va = de_train_indexed[mask_va] # shape (12x, 18211), double index
        
        # Drop outliers from training and validation
        # If removed_compounds is nonempty, some prediction rows will contain np.nan
        de_tr = de_tr.query("~sm_name.isin(@removed_compounds)")
#         de_va = de_va.query("~sm_name.isin(@removed_compounds)")

        # Add noise to the training t-scores
        if noise > 0:
            if fold == 0:
                print(f"{Fore.RED}{Style.BRIGHT}Adding noise of scale {noise:.2f}{Style.RESET_ALL}")
            rng = np.random.default_rng(1)
            de_tr = t_score_to_de(de_to_t_score(de_tr) + rng.normal(scale=noise, size=de_tr.shape))
    
        # Fit the model and predict validation log10pvalues
        de_pred = predictor(de_tr, de_va.index.to_frame())
        
        # Update out-of-fold predictions and score
        de_oof_list.append(de_pred)
        mrrmse = mean_rowwise_rmse(de_va, de_pred.values) # the competition metric
        print(f"# Fold {fold}: de_mrrmse={mrrmse:5.3f}   val='{val_cell_type}'")
        mrrmse_list.append(mrrmse)

    # Collect out-of-fold predictions and scores
    de_oof = pd.concat(de_oof_list, axis=0)
    mrrmse = mean_rowwise_rmse(de_train_indexed.reindex(de_oof.index),
                               de_oof)
    name = predictor.__name__[12:]
    print(f"{Fore.GREEN}{Style.BRIGHT}# Overall "
          f"de_mrrmse={mrrmse:5.3f} "
          f"{tuple((np.array(mrrmse_list) * 1000).round(0).astype(int))} "
          f"{name}{Style.RESET_ALL}")
    if noise == 0:
        de_oof_dict[name] = de_oof
    mrrmse_noise_list.append((name, noise, mrrmse))
    print()
    return

# Define outliers which are excluded from training and validation
# removed_compounds = ['AT13387', 'Alvocidib', 'BAY 61-3606', 'BMS-387032', 
#                      'Belinostat', 'CEP-18770 (Delanzomib)', 'CGM-097', 'CGP 60474', 
#                      'Dabrafenib', 'Ganetespib (STA-9090)', 'I-BET151', 'IN1451', 
#                      'LY2090314', 'MLN 2238', 'Oprozomib (ONX 0912)', 
#                      'Proscillaridin A;Proscillaridin-A', 'Resminostat',
#                      'Scriptaid', 'UNII-BXU45ZH6LI', 'Vorinostat']
removed_compounds = []

# Cross-validate the four models (saving the oof predictions)
predictors = [fit_predict_py_boost, fit_predict_ridge_recommender, fit_predict_knn_recommender, fit_predict_extratrees] 
de_oof_dict, mrrmse_noise_list = {}, []
for predictor in predictors:
    print(fit_predict_py_boost)
    cross_val_log10pvalue(predictor)

# Ensemble the oof predictions
de_oof = sum(de_oof_dict.values()) / len(de_oof_dict)
de_true = de_train_indexed.reindex_like(de_oof)
print(f"# Ensemble MRRMSE: {mean_rowwise_rmse(de_true, de_oof):.3f}")

def submit(output_path):
    """Refit the selected models and write the submission file."""
    
    # Drop outliers from training
    de_tr = de_train_indexed.query("~sm_name.isin(@removed_compounds)")
    
    # Fit all models and average their predictions
    pred_list = [fit_predict(de_tr, id_map) for fit_predict in predictors]
    de_pred = sum(pred_list) / len(pred_list)
        
    # Test for missing values
    if de_pred.isna().any().any():
        print("Warning: This submission contains missing values. "
              "Don't submit it!")
        
    # Create the submission dataframe
    submission = pd.DataFrame(de_pred.values, columns=genes, index=id_map.index)
    #display(submission)
    print(f'Variance of submission: {submission.values.var():.2f},   min = {submission.values.min():.2f}, max = {submission.values.max():.2f}')

    # Write the files
    # de_oof.to_csv('de_oof.csv')
    submission.to_csv(output_path)        

predictors = [fit_predict_py_boost] 
submit(par['output'])