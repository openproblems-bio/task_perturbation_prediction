# Change history
# * Imported code from https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206
# * Derive gene_names from train_df to to avoid reading in the sample_submission.csv file
# * Make reps component arguments
# * Auto-reformatted the code
#

# -----------------------------------------------------------------------------
# Load dependencies
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import gc
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Embedding,
    Flatten,
    GaussianNoise,
)
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers.legacy import Adam
from sklearn.decomposition import TruncatedSVD

import warnings

warnings.filterwarnings("ignore")


## VIASH START
par = {
    "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
    "id_map": "resources/neurips-2023-kaggle/id_map.csv",
    # "de_train": "resources/neurips-2023-data/de_train.parquet",
    # "id_map": "resources/neurips-2023-data/id_map.csv",
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
def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()


def load_train_data():
    train_df = pd.read_parquet(par["de_train"])
    train_df = train_df.sample(frac=1.0, random_state=42)
    return train_df


def mean_rowwise_rmse(y_true, y_pred):
    rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    mrrmse_score = np.mean(rowwise_rmse)
    return mrrmse_score


def abs_error(true, pred):
    return np.abs(true - pred).mean()


def custom_mean_rowwise_rmse(y_true, y_pred):
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def model_1(lr, emb_out, n_dim):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(256),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.2),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_2(lr, emb_out, dense_1, dense_2, dropout_1, dropout_2, n_dim):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),  # 256 - 2048
            Dense(dense_2, activation="relu"),
            Activation("relu"),
            BatchNormalization(),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_3(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 256
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(dense_4),  # 16 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_4),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_4(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_5(lr, emb_out, n_dim, dropout_1, dropout_2):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(256),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(1024),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_6(lr, emb_out, dense_1, dense_2, n_dim, dropout_1, dropout_2):
    tf.random.set_seed(42)
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            BatchNormalization(),
            Dense(dense_1),  # 64 - 512
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_2),
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(n_dim, activation="linear"),
        ]
    )
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_7(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 256
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(dense_4),  # 16 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_4),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_8(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim
):
    model = Sequential(
        [
            Embedding(152, emb_out, input_length=2),
            Flatten(),
            Dense(dense_1),  # 128 - 1024
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_1),
            Dense(dense_2),  # 64 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_2),
            Dense(dense_3),  # 32 - 512
            BatchNormalization(),
            Activation("relu"),
            Dropout(dropout_3),
            Dense(n_dim, activation="linear"),
        ]
    )

    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
params_model_1 = {
    "params": {
        "epochs": 114,
        "bs": 128,
        "lr": 0.008457844054540857,
        "emb_out": 22,
        "n_dim": 50,
    },
    "value": 0.9060678655727635,
}

params_model_2 = {
    "params": {
        "epochs": 136,
        "bs": 64,
        "lr": 0.007787474024659863,
        "emb_out": 10,
        "dense_1": 384,
        "dense_2": 1280,
        "dropout_1": 0.4643149193312417,
        "dropout_2": 0.10101884612160547,
        "n_dim": 60,
    },
    "value": 0.9070240468092804,
}

params_model_3 = {
    "params": {
        "epochs": 157,
        "bs": 64,
        "lr": 0.004311857150745656,
        "emb_out": 62,
        "dense_1": 560,
        "dense_2": 480,
        "dense_3": 248,
        "dense_4": 224,
        "dropout_1": 0.4359908049836846,
        "dropout_2": 0.34432694543970555,
        "dropout_3": 0.01112409967333259,
        "dropout_4": 0.23133616975077548,
        "n_dim": 119,
    },
    "value": 0.9171315640806535,
}

params_model_4 = {
    "params": {
        "epochs": 147,
        "bs": 64,
        "lr": 0.005948541271442179,
        "emb_out": 46,
        "dense_1": 872,
        "dense_2": 264,
        "dense_3": 256,
        "dropout_1": 0.17543603718794346,
        "dropout_2": 0.3587657616370447,
        "dropout_3": 0.12077512068514727,
        "n_dim": 213,
    },
    "value": 0.9228638968500431,
}

params_model_5 = {
    "params": {
        "epochs": 122,
        "bs": 32,
        "lr": 0.004429076555977599,
        "emb_out": 32,
        "n_dim": 71,
        "dropout_1": 0.40604535344002984,
        "dropout_2": 0.178189970426619,
    },
    "value": 0.9083640103276015,
}

params_model_6 = {
    "params": {
        "epochs": 112,
        "bs": 128,
        "lr": 0.009773732221901085,
        "emb_out": 60,
        "dense_1": 436,
        "dense_2": 416,
        "n_dim": 126,
        "dropout_1": 0.4024659444883379,
        "dropout_2": 0.2573940194596736,
    },
    "value": 0.8909352668212382,
}

params_model_7 = {
    "params": {
        "epochs": 141,
        "bs": 128,
        "lr": 0.005530331519967936,
        "emb_out": 48,
        "dense_1": 712,
        "dense_2": 400,
        "dense_3": 232,
        "dense_4": 216,
        "dropout_1": 0.4903998136177629,
        "dropout_2": 0.032371643764537134,
        "dropout_3": 0.11138300987168903,
        "dropout_4": 0.019885384663655765,
        "n_dim": 100,
    },
    "value": 0.8978272722102707,
}

params_model_8 = {
    "params": {
        "epochs": 143,
        "bs": 192,
        "lr": 0.00971858172843266,
        "emb_out": 48,
        "dense_1": 312,
        "dense_2": 344,
        "dense_3": 248,
        "dropout_1": 0.10974777738609129,
        "dropout_2": 0.10106027333885811,
        "dropout_3": 0.09775833250663657,
        "n_dim": 100,
    },
    "value": 0.8885448573595669,
}

w1 = [
    0.15224443321212433,
    0.7152220796128623,
    0.7547606691460997,
    0.05786285275052854,
    0.9602177109190158,
    0.4968056740470425,
    0.9881673272809887,
]

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

train_df = load_train_data().reset_index(drop=True)
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


reps = 10

test_df = pd.read_csv(par["id_map"])


def predict(test_df):
    x_test = le.transform(test_df[["cell_type", "sm_name"]].values.flat).reshape(-1, 2)
    
    preds_model_1 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_1, params_model_1
        )
        for i in range(reps)
    ]
    preds_model_2 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_2, params_model_2
        )
        for i in range(reps)
    ]
    preds_model_3 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_3, params_model_3
        )
        for i in range(reps)
    ]
    preds_model_5 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_5, params_model_5
        )
        for i in range(reps)
    ]
    preds_model_6 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_6, params_model_6
        )
        for i in range(reps)
    ]
    preds_model_7 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_7, params_model_7
        )
        for i in range(reps)
    ]
    preds_model_8 = [
        fit_and_predict_embedding_nn(
            new_names, original_y, x_test, model_8, params_model_8
        )
        for i in range(reps)
    ]

    pred1 = np.median(preds_model_1, axis=0)
    pred2 = np.median(preds_model_2, axis=0)
    pred3 = np.median(preds_model_3, axis=0)
    pred5 = np.median(preds_model_5, axis=0)
    pred6 = np.median(preds_model_6, axis=0)
    pred7 = np.median(preds_model_7, axis=0)
    pred8 = np.median(preds_model_8, axis=0)

    pred = (
        w1[0] * pred1
        + w1[1] * pred2
        + w1[2] * pred3
        + w1[3] * pred5
        + w1[4] * pred6
        + w1[5] * pred7
        + w1[6] * pred8
    ) / sum(w1)

    return pred


mins = original_y.min(axis=0)
maxs = original_y.max(axis=0)

pred = predict(test_df)
clipped_pred = np.clip(pred, mins, maxs)

gene_names = train_df.loc[:, "A1BG":].columns.tolist()
df = pd.DataFrame(clipped_pred, columns=gene_names)
df["id"] = range(len(df))
df = df.loc[:, ["id"] + gene_names]

# df.to_parquet(par["output"])
