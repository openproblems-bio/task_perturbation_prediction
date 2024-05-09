# Change history
# * Imported code from https://www.kaggle.com/code/jankowalski2000/3rd-place-solution
# * Derive gene_names from train_df to to avoid reading in the sample_submission.csv file
# * Make epochs and reps component arguments
# * Added a print statement to show progress
# * Auto-reformatted the code
# * Moved models, parameters and weights into a separate helper file
#
# NOTE: this script relies on "submission(14).csv", which according to the author
# ( https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/discussion/458750 )
# is created by version 264 of the same notebook
# ( https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206 )

# -----------------------------------------------------------------------------
# Load dependencies
# -----------------------------------------------------------------------------
import sys
import gc
import pandas as pd
import numpy as np
from random import shuffle

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

import warnings

warnings.filterwarnings("ignore")

## VIASH START
par = {
    # "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
    # "id_map": "resources/neurips-2023-kaggle/id_map.csv",
    "de_train": "resources/neurips-2023-data/de_train.parquet",
    "id_map": "resources/neurips-2023-data/id_map.csv",
    "output": "output.parquet",
    # lowering number of epochs and reps for testing purposes
    "epochs": 10,
    "reps": 2,
}
meta = {"resources_dir": "src/task/methods/third_place"}
## VIASH END

# -----------------------------------------------------------------------------
# Define helper functions and parameters
# -----------------------------------------------------------------------------

sys.path.append(meta["resources_dir"])

from helper import load_models, load_params, load_weights

def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()

def load_train_data():
    train_df = pd.read_parquet(par["de_train"])
    train_df = train_df.sample(frac=1.0, random_state=42)
    return train_df

def split_over_all_test(df, df_orig):
    valid_folds = []
    sm_names_B_M = pd.unique(
        df_orig[df_orig["cell_type"].isin(["B cells", "Myeloid cells"])]["sm_name"]
    )
    for cell_type in ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]:
        sub_df = df[
            (df["cell_type"] == cell_type) & (~df["sm_name"].isin(sm_names_B_M))
        ]
        valid_folds.append(sub_df.index.tolist())
    folds = []
    all_train = set(range(len(df)))
    for fold in valid_folds:
        fold_set = set(fold)
        train_rest = list(all_train - fold_set)
        shuffle(train_rest)
        shuffle(fold)
        folds.append((train_rest, fold))
    return folds


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

pseudolabel = pd.read_csv(meta["resources_dir"] + "/submission(14).csv")
test_df = pd.read_csv(par["id_map"])
pseudolabel = pd.concat(
    [test_df[["cell_type", "sm_name"]], pseudolabel.loc[:, "A1BG":]], axis=1
)

train_df = load_train_data().reset_index(drop=True)
columns = ["cell_type", "sm_name"] + train_df.loc[:, "A1BG":].columns.tolist()
train_df = train_df.loc[:, columns]

v = train_df.loc[:, "A1BG":]

train_df = pd.concat([train_df, pseudolabel]).reset_index(drop=True)

over_all_test = split_over_all_test(train_df, load_train_data().reset_index(drop=True))

original_x = train_df[["cell_type", "sm_name"]].values
original_y = train_df.loc[:, "A1BG":].values
le = LabelEncoder()
le.fit(original_x.flat)
new_names = le.transform(original_x.flat).reshape(-1, 2)


def split_params_to_training_model(model_params):
    model_params = model_params["params"]
    training_keys = ["epochs", "bs"]
    training_params = {k: model_params[k] for k in training_keys}
    model_params = {
        k: model_params[k] for k in model_params.keys() if k not in training_keys
    }
    return model_params, training_params


def fit_and_predict_embedding_nn(x, y, test_x, model_constructor, best_params):
    model_params, training_params = split_params_to_training_model(best_params)
    n_dim = model_params["n_dim"]
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    model.fit(
        x,
        y,
        epochs=training_params["epochs"],
        batch_size=training_params["bs"],
        verbose=0,
        shuffle=True,
    )
    return d.inverse_transform(model.predict(test_x, batch_size=1))


def predict(test_df, models, params, weights):
    x_test = le.transform(test_df[["cell_type", "sm_name"]].values.flat).reshape(-1, 2)

    preds = []
    # for model, param in zip(models, params):
    for modeli in range(len(models)):
        model = models[modeli]
        param = params[modeli]
        temp_pred = []
        for i in range(par["reps"]):
            print(
                f"Training model {modeli}/{len(models)}, repeat {i}/{par['reps']}",
                flush=True,
            )
            temp_pred.append(
                fit_and_predict_embedding_nn(
                    new_names, original_y, x_test, model, param
                )
            )
        temp_pred = np.median(temp_pred, axis=0)
        preds.append(temp_pred)

    pred = np.sum([w * p for w, p in zip(weights, preds)], axis=0) / sum(weights)
    return pred

models = load_models()
params = load_params(par["epochs"])
weights = load_weights()

pred = predict(test_df, models, params, weights)

mins = v.min(axis=0).values
maxs = v.max(axis=0).values

clipped_pred = np.clip(pred, mins, maxs)

gene_names = train_df.loc[:, "A1BG":].columns.tolist()
df = pd.DataFrame(clipped_pred, columns=gene_names)
df["id"] = range(len(df))
df = df.loc[:, ["id"] + gene_names]

df.to_parquet(par["output"])
