import sys
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tensorflow import keras

## VIASH START
par = {
  "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
  "de_test": "resources/neurips-2023-kaggle/de_test.parquet",
  "id_map": "resources/neurips-2023-kaggle/id_map.csv",
  "output": "output.parquet",
}
## VIASH END

## CUSTOM CODE START
# import helper function
sys.path.append(meta['resources_dir'])

from util import preprocess_data, train_nn, custom_mean_rowwise_rmse
from models import MODELS_STAGE_1, MODELS_STAGE_1_IDXS, MODELS_STAGE_2, MODELS_STAGE_2_IDXS, SIMPLE_MODEL, SIMPLE_MODEL_IDXS
from params_stage_1 import PARAMS_STAGE_1, SIMPLE_MODEL_PARAMS, WEIGHTS_SIMPLE_MODEL, WEIGHTS_STAGE_1
from params_stage_2 import PARAMS_STAGE_2, WEIGHTS_STAGE_2

REPS = 10

def load_config(stage: str):
    """
    Args:
        stage: one of simple, stage_1, stage_2
    Returns:
        models - a list of model constructors
        params - a list of hiperparameters
        model_idxs - a list of model idxs
        reps - number of repeated training.
    """
    if stage == 'stage_1':
        models = MODELS_STAGE_1
        params = PARAMS_STAGE_1
        model_idxs = MODELS_STAGE_1_IDXS
        reps = REPS
        weights = WEIGHTS_STAGE_1
    elif stage == 'stage_2':
        models = MODELS_STAGE_2
        params = PARAMS_STAGE_2
        model_idxs = MODELS_STAGE_2_IDXS
        reps = REPS
        weights = WEIGHTS_STAGE_2
    elif stage =='simple':
        models = SIMPLE_MODEL
        params = SIMPLE_MODEL_PARAMS
        model_idxs = SIMPLE_MODEL_IDXS
        reps = 1
        weights = WEIGHTS_SIMPLE_MODEL
    return models, params, model_idxs, reps, weights

def predict_single_model(x_test: np.array, y: np.array, model_path: Path) -> np.array:
    """
    Args:
        x_test - a test input
        y - labels of training dataset - needed for a truncated SVD
    Returns:
        preds - predictions in a numpy array
    """
    model = keras.models.load_model(model_path, 
                                    custom_objects={'custom_mean_rowwise_rmse': custom_mean_rowwise_rmse})
    preds = model.predict(x_test, batch_size=1)
    decomposition = TruncatedSVD(preds.shape[1])
    decomposition.fit(y)
    preds = decomposition.inverse_transform(preds)
    return preds

## CUSTOM CODE END

print('Reading input files', flush=True)
de_train = pd.read_parquet(par["de_train"])
id_map = pd.read_csv(par["id_map"])
gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]

for stage in ['stage_1', 'stage_2']:

  print('Preprocess data', flush=True)
  if stage == 'stage_2':
    pseudolabel = output
    pseudolabel = pd.concat([id_map[['cell_type', 'sm_name']], pseudolabel.loc[:, 'A1BG':]], axis=1)
    de_train = pd.concat([de_train, pseudolabel]).reset_index(drop=True)

  de_train = de_train.sample(frac=1.0, random_state=42)
  x, y, test_x = preprocess_data(de_train, id_map)

  print('Train model', flush=True)

  models, params, model_idxs, reps, weights = load_config(stage)
  model_paths = []
  for model_id, model_costructor, model_params in zip(model_idxs, models, params):
      for rep in range(reps):
          model_path = Path(f"models/model_{model_id}_{rep}.keras")
          print("Model path", model_path)
          train_nn(x, y, model_costructor, model_params, model_path)
          model_paths.append(model_path)

  print('Generate predictions', flush=True)
  grouped_models = defaultdict(list)
  for model_path in model_paths:
      stem = Path(model_path).stem
      model_id = stem.split('_')[-2]
      grouped_models[model_id].append(model_path)
  predictions = []
  sorted_keys = sorted(list(grouped_models.keys()))
  for k in sorted_keys:
      temp_preds = []
      for p in grouped_models[k]:
          preds = predict_single_model(test_x, y, p)
          temp_preds.append(preds)
      preds = np.median(temp_preds, axis=0)
      predictions.append(preds)
  preds = np.sum([w * p for w, p in zip(weights, predictions)], axis=0) / sum(weights)

  values = de_train.loc[:, 'A1BG':].values
  mins = values.min(axis=0)
  maxs = values.max(axis=0)
  clipped_preds = np.clip(preds, mins, maxs)

  output = pd.DataFrame(
    clipped_preds,
    index=id_map["id"],
    columns=gene_names
  ).reset_index()

print('Write output to file', flush=True)
output.to_parquet(par["output"])
