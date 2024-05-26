# This script is based an IPython notebook:
# https://github.com/AntoinePassemiers/Open-Challenges-Single-Cell-Perturbations/blob/master/op2-de-dl.ipynb

import sys
import pandas as pd
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Make PyTorch deterministic on GPU
from typing import List
import numpy as np
import pandas as pd
import tqdm
import torch

## VIASH START
par = {
    "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
    "id_map": "resources/neurips-2023-kaggle/id_map.csv",
    "output": "output.parquet",
    "n_replica": 1,
    "submission_names": ["dl40"]
}
meta = {
    "resources_dir": "src/task/methods/jn_ap_op2",
}
## VIASH END

sys.path.append(meta["resources_dir"])

from helper import plant_seed, MultiOutputTargetEncoder, train

print('Reading input files', flush=True)
de_train = pd.read_parquet(par["de_train"])
id_map = pd.read_csv(par["id_map"])

gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]

print('Preprocess data', flush=True)
SEED = 0xCAFE
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
    USE_GPU = False
    
# Make Python deterministic?
os.environ['PYTHONHASHSEED'] = str(int(SEED))

# Make PyTorch deterministic
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)
plant_seed(SEED, USE_GPU)

print('Data location', flush=True)
# Data location
cell_types = de_train['cell_type']
sm_names = de_train['sm_name']

data = de_train.drop(columns=["cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control"]).to_numpy(dtype=float)

print('Train model', flush=True)
# ... train model ...
encoder = MultiOutputTargetEncoder()

encoder.fit(np.asarray([cell_types, sm_names]).T, data)

X = torch.FloatTensor(encoder.transform(np.asarray([cell_types, sm_names]).T))
X_submit = torch.FloatTensor(encoder.transform(np.asarray([id_map.cell_type, id_map.sm_name]).T))

if USE_GPU:
    X = X.cuda()

print('Generate predictions', flush=True)
# ... generate predictions ...

Y_submit_ensemble = []
for SUBMISSION_NAME in par["submission_names"]:
  #train the models and store them
  models = []
  for i in range(par["n_replica"] ):
      seed = i
      if SUBMISSION_NAME == 'dl40':
          model = train(X, torch.FloatTensor(data), np.arange(len(X)), seed, n_iter=40, USE_GPU=USE_GPU)
      elif SUBMISSION_NAME == 'dl200':
          model = train(X, torch.FloatTensor(data), np.arange(len(X)), seed, n_iter=200, USE_GPU=USE_GPU)
      else:
          model = train(X, torch.FloatTensor(data), np.arange(len(X)), seed, n_iter=40, USE_GPU=USE_GPU)
      model.eval()
      models.append(model)
      torch.cuda.empty_cache()
  # predict 
  Y_submit =  []
  for i, x in tqdm.tqdm(enumerate(X_submit), desc='Submission'):
    # Predict on test sample using a simple ensembling strategy:
    # take the median of the predictions across the different models
    y_hat = []
    for model in models:
        model = model.cpu()
        y_hat.append(np.squeeze(model.forward(x.unsqueeze(0)).cpu().data.numpy()))
    y_hat = np.median(y_hat, axis=0)

    values = [f'{x:.5f}' for x in y_hat]
    Y_submit.append(values)
    
  Y_submit_ensemble.append(np.asarray(Y_submit).astype(np.float32))
    
Y_submit_final = np.mean(Y_submit_ensemble, axis=0)

print('Write output to file', flush=True)
output = pd.DataFrame(
  Y_submit_final,
  index=id_map["id"],
  columns=gene_names
).reset_index()
output.to_parquet(par["output"])





