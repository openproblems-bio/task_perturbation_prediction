# import pandas as pd

# ## VIASH START
# par = {
#   "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
#   "de_test": "resources/neurips-2023-kaggle/de_test.parquet",
#   "id_map": "resources/neurips-2023-kaggle/id_map.csv",
#   "output": "output.parquet",
# }
# ## VIASH END

# print('Reading input files', flush=True)
# de_train = pd.read_parquet(par["de_train"])
# id_map = pd.read_csv(par["id_map"])
# gene_names = [col for col in de_train.columns if col not in {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "split", "control", "index"}]

# print('Preprocess data', flush=True)
# # ... preprocessing ...

# print('Train model', flush=True)
# # ... train model ...

# print('Generate predictions', flush=True)
# # ... generate predictions ...

# print('Write output to file', flush=True)
# output = pd.DataFrame(
#   # ... TODO: fill in data ...
#   index=id_map["id"],
#   columns=gene_names
# ).reset_index()
# output.to_parquet(par["output"])

import torch
import torch.nn as nn
import torch.optim
 

## VIASH START
par = {
  "de_train": "resources/neurips-2023-kaggle/de_train.parquet",
  "de_test": "resources/neurips-2023-kaggle/de_test.parquet",
  "id_map": "resources/neurips-2023-kaggle/id_map.csv",
  "output": "output.parquet",
}
## VIASH END


class CustomTransformer(nn.Module):
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6):  # num_heads=8
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        # Embedding layer for sparse features
        # self.embedding = nn.Embedding(num_features, d_model)

        # self.norm = nn.BatchNorm1d(d_model, affine=True)
        self.norm = nn.LayerNorm(d_model)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
        #                                 dropout=0.1, device='cuda')
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, device='cuda', dropout=0.3,
                                       activation=nn.GELU(),
                                       batch_first=True), enable_nested_tensor=True, num_layers=num_layers
        )
        # Dropout layer for regularization
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        x = self.embedding(x)

        # x = (self.transformer(x,x))
        x = self.transformer(x)
        x = self.norm(x)
        # x = self.fc(self.dropout(x))
        x = self.fc(x)
        return x


class CustomTransformer_v3(nn.Module):  # mean + std
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(CustomTransformer_v3, self).__init__()
        self.num_target_encodings = 18211 * 4
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x


class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.3, layer_norm=True):
        super(CustomMLP, self).__init__()
        layers = []

        for _ in range(num_layers):
            if layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x



import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim
from lion_pytorch import Lion
from sklearn.preprocessing import StandardScaler
import yaml
import pickle
from torch.utils.data import TensorDataset, DataLoader
# from models import *


# Evaluate the loaded model on the test data
def evaluate_model(model, dataloader, criterion=None):
    model.eval()
    total_output = []
    total_labels = []
    running_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            total_output.append(outputs)
            total_labels.append(labels)
            num_batches += 1
        total_mrrmse = calculate_mrrmse(torch.cat(total_labels, dim=0), torch.cat(total_output, dim=0))
    if not criterion:
        return total_mrrmse.detach().cpu().item()
    return total_mrrmse.detach().cpu().item(), running_loss / num_batches


def load_transformer_model(n_components, input_features, d_model, models_foler='trained_models', device='cuda'):
    # transformer_model = CustomTransformer(num_features=input_features, num_labels=n_components, d_model=d_model).to(
    #     device)
    transformer_model = CustomTransformer_v3(num_features=input_features, num_labels=n_components, d_model=d_model).to(
        device)
    # transformer_model = CustomDeeperModel(input_features, d_model, n_components).to(device)
    transformer_model.load_state_dict(torch.load(f'trained_models/transformer_model_{n_components}_{d_model}.pt'))
    transformer_model.eval()
    if n_components == 18211:
        return None, None, transformer_model
    label_reducer = pickle.load(open(f'{models_foler}/label_reducer_{n_components}_{d_model}.pkl', 'rb'))
    scaler = pickle.load(open(f'{models_foler}/scaler_{n_components}_{d_model}.pkl', 'rb'))
    return label_reducer, scaler, transformer_model


def reduce_labels(Y, n_components):
    if n_components == Y.shape[1]:
        return None, None, Y
    label_reducer = TruncatedSVD(n_components=n_components, n_iter=10)
    scaler = StandardScaler()

    Y_scaled = scaler.fit_transform(Y)
    Y_reduced = label_reducer.fit_transform(Y_scaled)

    return label_reducer, scaler, Y_reduced


def prepare_augmented_data(
        data_file="",
        id_map_file=""):
    de_train = pd.read_parquet(data_file)
    id_map = pd.read_csv(id_map_file)
    xlist = ['cell_type', 'sm_name']
    _ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    y = de_train.drop(columns=_ylist)

    # Combine train and test data for one-hot encoding
    combined_data = pd.concat([de_train[xlist], id_map[xlist]])

    dum_data = pd.get_dummies(combined_data, columns=xlist)

    # Split the combined data back into train and test
    train = dum_data.iloc[:len(de_train)]
    test = dum_data.iloc[len(de_train):]
    # uncommon = [f for f in train if f not in test]
    # X = train.drop(columns=uncommon)
    X = train
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]

    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()

    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()

    # Append mean and std for 'cell_type'
    rows = []
    for name in de_cell_type['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_cell_type = pd.concat(rows)
    tr_cell_type = tr_cell_type.reset_index(drop=True)

    # Append mean and std for 'sm_name'
    rows = []
    for name in de_sm_name['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    tr_sm_name = pd.concat(rows)
    tr_sm_name = tr_sm_name.reset_index(drop=True)

    # Similar process for test data
    rows = []
    for name in id_map['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        std_rows = std_cell_type[std_cell_type['cell_type'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_cell_type = pd.concat(rows)
    te_cell_type = te_cell_type.reset_index(drop=True)

    rows = []
    for name in id_map['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        std_rows = std_sm_name[std_sm_name['sm_name'] == name].copy()
        rows.append(pd.concat([mean_rows, std_rows.add_suffix('_std')], axis=1))

    te_sm_name = pd.concat(rows)
    te_sm_name = te_sm_name.reset_index(drop=True)

    # Join mean and std features to X0, test0
    X0 = X.join(tr_cell_type.iloc[:, 1:]).copy()
    X0 = X0.join(tr_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    X0 = X0.select_dtypes(exclude='object')

    y0 = y.iloc[:, :].copy()

    test0 = test.join(te_cell_type.iloc[:, 1:]).copy()
    test0 = test0.join(te_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    # Remove string columns
    test0 = test0.select_dtypes(exclude='object')
    return X0.astype(np.float32).to_numpy(), y0, test0.astype(np.float32).to_numpy()


def prepare_augmented_data_mean_only(
        data_file="",
        id_map_file=""):
    de_train = pd.read_parquet(data_file)
    id_map = pd.read_csv(id_map_file)
    xlist = ['cell_type', 'sm_name']
    _ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
    y = de_train.drop(columns=_ylist)
    # train = pd.get_dummies(de_train[xlist], columns=xlist)
    # test = pd.get_dummies(id_map[xlist], columns=xlist)
    # Combine train and test data for one-hot encoding
    combined_data = pd.concat([de_train[xlist], id_map[xlist]])

    dum_data = pd.get_dummies(combined_data, columns=xlist)

    # Split the combined data back into train and test
    train = dum_data.iloc[:len(de_train)]
    test = dum_data.iloc[len(de_train):]
    # uncommon = [f for f in train if f not in test]
    # X = train.drop(columns=uncommon)

    X = train
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    rows = []
    for name in de_cell_type['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        rows.append(mean_rows)
    tr_cell_type = pd.concat(rows)
    tr_cell_type = tr_cell_type.reset_index(drop=True)

    rows = []
    for name in de_sm_name['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        rows.append(mean_rows)
    tr_sm_name = pd.concat(rows)
    tr_sm_name = tr_sm_name.reset_index(drop=True)

    rows = []
    for name in id_map['cell_type']:
        mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
        rows.append(mean_rows)

    te_cell_type = pd.concat(rows)
    te_cell_type = te_cell_type.reset_index(drop=True)
    rows = []
    for name in id_map['sm_name']:
        mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
        rows.append(mean_rows)

    te_sm_name = pd.concat(rows)
    te_sm_name = te_sm_name.reset_index(drop=True)
    X0 = X.join(tr_cell_type.iloc[:, 1:]).copy()
    X0 = X0.join(tr_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    y0 = y.iloc[:, :].copy()
    test0 = test.join(te_cell_type.iloc[:, 1:]).copy()
    test0 = test0.join(te_sm_name.iloc[:, 1:], lsuffix='_cell_type', rsuffix='_sm_name')
    return X0.astype(np.float32).to_numpy(), y0, test0.astype(np.float32).to_numpy()


def load_and_print_config(config_file):
    # Load configurations from the YAML file
    config = load_config(config_file)

    # Print loaded configurations
    print("Configurations:")
    print(config)

    return config


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_data(train_features, targets, test_size=0.3, shuffle=False, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(train_features, targets, test_size=test_size,
                                                      shuffle=shuffle, random_state=random_state)

    return X_train, X_val, y_train.to_numpy(), y_val.to_numpy()


def calculate_mrrmse_np(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = np.sqrt(np.mean((outputs - labels.reshape(-1, outputs.shape[1])) ** 2, axis=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = np.mean(rmse_per_row)
    return mrmse


# Function to calculate MRRMSE
def calculate_mrrmse(outputs, labels):
    # Calculate the Root Mean Squared Error (RMSE) row-wise
    rmse_per_row = torch.sqrt(torch.mean((outputs - labels) ** 2, dim=1))
    # Calculate the Mean RMSE (MRMSE) across all rows
    mrmse = torch.mean(rmse_per_row)
    return mrmse


# Custom mrrmse Loss Function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = calculate_mrrmse(predictions, targets)  # + torch.abs(predictions-targets).mean()
        return loss.mean()


# Plot Loss and mrrmse
def plot_mrrmse(val_mrrmse):
    epochs = range(1, len(val_mrrmse) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_mrrmse, 'r', label='Validation mrrmse')
    plt.title('Validation mrrmse')
    plt.xlabel('Epochs')
    plt.ylabel('mrrmse')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Save Model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)



# For evaluation
def train_and_evaluate_model(X_train, y_train, X_val, y_val, num_epochs, batch_size, learning_rate, val_loss_path):
    device = 'cuda'

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    min_val_mrrmse = float('inf')
    min_loss = float('inf')
    num_features = X_train.shape[1]
    num_labels = y_train.shape[1]

    model = CustomTransformer(num_features, num_labels).to(device)
    # criterion_mse = nn.MSELoss()
    # criterion_mae = nn.L1Loss()  # nn.HuberLoss()#  # Mean Absolute Error
    criterion_mae = nn.HuberLoss(reduction='sum')
    # criterion = CustomLoss()
    #
    # weight_decay = 1e-4
    betas = (0.9, 0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-3, betas=betas)

    val_losses = []
    val_mrrmses = []
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0, verbose=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.9999, patience=250)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, targets)
            loss = criterion_mae(outputs, targets)  # criterion_mse(outputs,targets)#
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # epoch_loss = running_loss / num_batches
        with torch.no_grad():
            val_mrrmse, epoch_loss = evaluate_model(model, val_loader, criterion_mae)
        if val_mrrmse < min_val_mrrmse:
            min_val_mrrmse = val_mrrmse
            save_model(model, 'mrrmse_val_' + val_loss_path)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            save_model(model, 'loss_val_' + val_loss_path)
        val_losses.append(epoch_loss)
        val_mrrmses.append(val_mrrmse)

        print(f'Epoch {epoch + 1}/{num_epochs} - Val MRRMSE: {val_mrrmse:.4f} - Loss: {epoch_loss:.4f}')
        # Adjust learning rate based on validation MRRMSE
        # scheduler.step(epoch_loss)
        scheduler.step()
    # Plot validation MRRMSE and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(val_mrrmses, label='Validation MRRMSE')
    plt.xlabel('Epoch')
    plt.ylabel('MRRMSE')
    plt.title('Validation MRRMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.show()



import sys

# from utils import *
from sklearn.cluster import KMeans
import copy
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pickle
import argparse
# from models import CustomTransformer_v3  # Can be changed to other models in models.py
import os


def train_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, val_dataloader, criterion, label_reducer=None, scaler=None, device='cuda'):
    model.eval()
    val_loss = 0.0
    val_predictions_list = []
    val_targets_list = []
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_targets_list.append(val_targets.clone().cpu())
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_predictions = model(val_inputs)
            if label_reducer:
                val_targets = torch.tensor(
                    label_reducer.transform(scaler.transform(val_targets.clone().cpu().detach().numpy())),
                    dtype=torch.float32).to(device)
            val_loss += criterion(val_predictions, val_targets).item()
            val_predictions_list.append(val_predictions.cpu())

    val_loss /= len(val_dataloader)

    val_predictions_stacked = torch.cat(val_predictions_list, dim=0)
    val_targets_stacked = torch.cat(val_targets_list, dim=0)

    return val_loss, val_targets_stacked, val_predictions_stacked


def validate_sampling_strategy(sampling_strategy):
    allowed_strategies = ['k-means', 'random']
    if sampling_strategy not in allowed_strategies:
        raise ValueError(f"Invalid sampling strategy. Choose from: {', '.join(allowed_strategies)}")


def train_func(X_train, Y_reduced, X_val, Y_val, n_components, num_epochs, batch_size, label_reducer, scaler,
               d_model=128, early_stopping=5000, device='cuda', ):
    best_mrrmse = float('inf')
    best_model = None
    best_val_loss = float('inf')
    best_epoch = 0
    # model = CustomTransformer(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(device)
    model = CustomTransformer_v3(num_features=X_train.shape[1], num_labels=n_components, d_model=d_model).to(device)
    # model = CustomDeeperModel(X_train.shape[1], d_model, n_components).to(device)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                            torch.tensor(Y_reduced, dtype=torch.float32).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                              torch.tensor(
                                                  Y_val,
                                                  dtype=torch.float32).to(device)),
                                batch_size=batch_size, shuffle=False)
    if n_components < 18211:
        lr = 1e-3

    else:
        lr = 1e-5
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7, verbose=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9999, patience=500,
                                               verbose=True)
    criterion = nn.HuberLoss()
    # criterion = nn.L1Loss()
    # criterion = CustomLoss()
    # criterion = nn.MSELoss()
    model.train()
    counter = 0
    pbar = tqdm(range(num_epochs), position=0, leave=True)
    for epoch in range(num_epochs):
        _ = train_epoch(model, dataloader, optimizer, criterion)

        if counter >= early_stopping:
            break
        if scaler:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion,
                                                                              label_reducer, scaler)
            # Calculate MRRMSE for the entire validation set
            val_mrrmse = calculate_mrrmse_np(
                val_targets_stacked.cpu().detach().numpy(),
                scaler.inverse_transform((label_reducer.inverse_transform(
                    val_predictions_stacked.cpu().detach().numpy()))))
        else:
            val_loss, val_targets_stacked, val_predictions_stacked = validate(model, val_dataloader, criterion)
            val_mrrmse = calculate_mrrmse_np(val_targets_stacked.cpu().detach().numpy(),

                                             val_predictions_stacked.cpu().detach().numpy())

        if val_mrrmse < best_mrrmse:
            best_mrrmse = val_mrrmse
            # best_model = copy.deepcopy(model)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
            best_epoch = epoch
        else:
            counter += 1

        pbar.set_description(
            f"Validation best MRRMSE: {best_mrrmse:.4f} Validation best loss:"
            f" {best_val_loss:.4f} Last epoch: {best_epoch}")
        pbar.update(1)
        # scheduler.step()  # for cosine anealing
        scheduler.step(val_loss)
    return label_reducer, scaler, best_model


def train_transformer_k_means_learning(X, Y, n_components, num_epochs, batch_size,
                                       d_model=128, early_stopping=5000, device='cuda', seed=18):
    label_reducer, scaler, Y_reduced = reduce_labels(Y, n_components)
    Y_reduced = Y_reduced.to_numpy()
    Y = Y.to_numpy()
    num_clusters = 2
    validation_percentage = 0.1

    # Create a K-Means clustering model
    kmeans = KMeans(n_clusters=num_clusters, n_init=100)

    # Fit the model to your regression targets (Y)
    clusters = kmeans.fit_predict(Y)

    # Initialize lists to store the training and validation data
    X_train, Y_train = [], []
    X_val, Y_val = [], []

    # Iterate through each cluster
    for cluster_id in range(num_clusters):
        # Find the indices of data points in the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        print(len(cluster_indices))
        if len(cluster_indices) >= 20:
            # Split the data points in the cluster into training and validation
            train_indices, val_indices = train_test_split(cluster_indices, test_size=validation_percentage,
                                                          random_state=seed)

            # Append the corresponding data points to the training and validation sets
            X_train.extend(X[train_indices])
            Y_train.extend(Y_reduced[train_indices])  # Y_reduced for train Y for validation
            X_val.extend(X[val_indices])
            Y_val.extend(Y[val_indices])
        else:
            X_train.extend(X[cluster_indices])
            Y_train.extend(Y_reduced[cluster_indices])  # Y_reduced for train Y for validation
    # Convert the lists to numpy arrays if needed
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    transfromer_model = train_func(X_train, Y_train, X_val, Y_val, n_components, num_epochs, batch_size,
                                   label_reducer, scaler, d_model, early_stopping, device)

    return label_reducer, scaler, transfromer_model


def train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                           early_stopping, batch_size, device):
    # Training loop for k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = train_transformer_k_means_learning(
                one_hot_encode_features,
                targets,
                n_components,
                num_epochs=num_epochs,
                early_stopping=early_stopping,
                batch_size=batch_size,
                d_model=d_model, device=device)
            os.makedirs('trained_models_random', exist_ok=True)
            # Save the trained models
            with open(f'trained_models_random/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'trained_models_random/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)
                
            torch.save(transformer_model[2].state_dict(),
                       f'trained_models_random/transformer_model_{n_components}_{d_model}.pt')


def train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device, seed, validation_percentage):
    # Split the data for non-k_means sampling strategy
    X_train, X_val, y_train, y_val = split_data(one_hot_encode_features, targets, test_size=validation_percentage,
                                                shuffle=True, random_state=seed)

    # Training loop for non-k_means sampling strategy
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, Y_reduced = reduce_labels(y_train, n_components)
            transformer_model = train_func(X_train, y_train, X_val, y_val,
                                           n_components,
                                           num_epochs=num_epochs,
                                           early_stopping=early_stopping,
                                           batch_size=batch_size,
                                           d_model=d_model,
                                           label_reducer=label_reducer,
                                           scaler=scaler,
                                           device=device)

            # Save the trained models
            os.makedirs('trained_models_k-means', exist_ok=True)
            with open(f'trained_models_k-means/label_reducer_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(label_reducer, file)

            with open(f'trained_models_k-means/scaler_{n_components}_{d_model}.pkl', 'wb') as file:
                pickle.dump(scaler, file)

            torch.save(transformer_model.state_dict(),
                       f'trained_models_k-means/transformer_model_{n_components}_{d_model}.pt')


def main1():
    # # Set up command-line argument parser
    # parser = argparse.ArgumentParser(description="Your script description here.")
    # parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    # args = parser.parse_args()
    # print(args)
    # # Check if the config file is provided
    # if not args.config:
    #     print("Please provide a config file using --config.")
    #     return

    # # Load and print configurations
    # config_file = args.config
    # config = load_and_print_config(config_file)

    # Access specific values from the config
    # n_components_list = config.get('n_components_list', [])
    # d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    # batch_size = config.get('batch_size', 32)
    # sampling_strategy = config.get('sampling_strategy', 'random')
    # data_file = par['de_train']
    # id_map_file = par['id_map']
    # validation_percentage = config.get('validation_percentage', 0.2)
    # device = config.get('device', 'cuda')
    # seed = config.get('seed', None)
    # num_epochs = config.get('num_epochs', 20000)
    # early_stopping = config.get('early_stopping', 5000)

    n_components_list = [18211]
    d_models_list = [128]  # embedding dimensions for the transformer models
    batch_size = 32
    sampling_strategy = "k-means"
    data_file = par['de_train']
    id_map_file = par['id_map']
    validation_percentage = 0.1
    device = 'cpu'
    seed = None
    num_epochs = 20000
    early_stopping = 5000
    print('start training')

    # Validate the sampling strategy
    validate_sampling_strategy(sampling_strategy)

    # Prepare augmented data
    one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(data_file=data_file,
                                                                            id_map_file=id_map_file)

    targets = targets.drop(columns = ['split'])
    # one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(data_file=data_file,
    #                                                                         id_map_file=id_map_file)
    if sampling_strategy == 'k-means':
        train_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                               early_stopping, batch_size, device)
    else:
        train_non_k_means_strategy(n_components_list, d_models_list, one_hot_encode_features, targets, num_epochs,
                                   early_stopping, batch_size, device, seed, validation_percentage)
    print("Finish running stage 1!")


import copy

import argparse


@torch.no_grad()
def predict_test(data, models, n_components_list, d_list, batch_size, device='cuda'):
    num_samples = len(data)
    for i, n_components in enumerate(n_components_list):
        for j, d_model in enumerate(d_list):
            combined_outputs = []
            label_reducer, scaler, transformer_model = models[f'{n_components},{d_model}']
            transformer_model.eval()
            for i in range(0, num_samples, batch_size):
                batch_unseen_data = data[i:i + batch_size]
                transformed_data = transformer_model(batch_unseen_data)
                if scaler:
                    transformed_data = torch.tensor(scaler.inverse_transform(
                        label_reducer.inverse_transform(transformed_data.cpu().detach().numpy()))).to(device)
                print(transformed_data.shape)
                combined_outputs.append(transformed_data)

            # Stack the combined outputs
            combined_outputs = torch.vstack(combined_outputs)
            sample_submission = pd.read_csv(
                f"./sample_submission.csv")
            print(sample_submission)
            print(combined_outputs.cpu().detach().numpy().shape)
            sample_columns = sample_submission.columns
            sample_columns = sample_columns[1:]
            submission_df = pd.DataFrame(combined_outputs.cpu().detach().numpy(), columns=sample_columns)
            submission_df.insert(0, 'id', range(255))
            # output_path = "resources/neurips-2023-data/" + f"result_{n_components}_{d_model}"
            # par[f"result_{n_components}_{d_model}"] = output_path
            # submission_df.to_parquet(par[f"result_{n_components}_{d_model}"])
            submission_df.to_csv(f"result_{n_components}_{d_model}.csv", index=False)

    return


def main2():
    # Set up command-line argument parser
    # parser = argparse.ArgumentParser(description="Your script description here.")
    # parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    # args = parser.parse_args()

    # # Check if the config file is provided
    # if not args.config:
    #     print("Please provide a config file using --config.")
    #     return

    # # Load and print configurations
    # config_file = args.config
    # config = load_and_print_config(config_file)
    # # Access specific values from the config
    # n_components_list = config.get('n_components_list', [])
    # d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    # batch_size = config.get('batch_size', 32)
    # data_file =  par['de_test']
    # id_map_file =  par['id_map']
    # device = config.get('device', 'cuda')
    # models_dir = config.get('dir', 'model_1_mean_std_only')

    n_components_list = [18211]
    d_models_list = [128]  # embedding dimensions for the transformer models
    batch_size = 5
    data_file =  par['de_test']
    id_map_file =  par['id_map']
    device = 'cpu'
    models_dir = 'trained_models'


    # Prepare augmented data
    if 'std' in models_dir:
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(data_file=data_file,
                                                                                id_map_file=id_map_file)
    else:
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(data_file=data_file,
                                                                                          id_map_file=id_map_file)
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(device)  # Replace X_unseen with your new data
    transformer_models = {}
    for n_components in n_components_list:
        for d_model in d_models_list:
            label_reducer, scaler, transformer_model = load_transformer_model(n_components,
                                                                              input_features=
                                                                              one_hot_encode_features.shape[
                                                                                  1],
                                                                              d_model=d_model,
                                                                              models_foler=f'{models_dir}',
                                                                              device=device)
            transformer_model.eval()
            transformer_models[f'{n_components},{d_model}'] = (
                copy.deepcopy(label_reducer), copy.deepcopy(scaler), copy.deepcopy(transformer_model))
    predict_test(unseen_data, transformer_models, n_components_list, d_models_list, batch_size, device=device)
    print("Finish running stage 2!")


import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def normalize_weights(weights):
    total_weight = sum(weights)
    return [weight / total_weight for weight in weights]


def calculate_weighted_sum(dataframes, weights):
    weighted_dfs = [df * weight for df, weight in zip(dataframes, weights)]
    return sum(weighted_dfs)


def convert_to_consistent_dtype(df):
    df = df.astype(float)
    df['id'] = df['id'].astype(int)
    return df


def set_id_as_index(df):
    df.set_index('id', inplace=True)
    return df


def create_submission_df(weighted_sum):
    sample_submission = pd.read_csv(r"\kaggle_data\sample_submission.csv")
    sample_columns = sample_submission.columns[1:]
    submission_df = pd.DataFrame(weighted_sum.iloc[:, :].to_numpy(), columns=sample_columns)
    submission_df.insert(0, 'id', range(255))
    return submission_df


def save_submission_df(submission_df, file_path='weighted_submission.csv'):
    submission_df.to_csv(file_path, index=False)


def main3():
    # Load CSV DataFrames
    df1 = load_data(r"submissions/result (15).csv")
    df2 = load_data(r"submissions/result (9).csv")
    df3 = load_data(r"submissions/result (11).csv")
    df6 = load_data(r"submissions/result (8).csv")  # amplifier

    # Define weights for each DataFrame
    weights = [0.4, 0.1, 0.2, 0.3]

    # Normalize weights for df1, df2, and df3 to ensure their sum is 1
    normalized_weights = normalize_weights(weights[:-1]) + [weights[-1]]

    # Apply normalized weights to each DataFrame
    weighted_sum = calculate_weighted_sum([df1, df2, df3, df6], normalized_weights)

    # Convert all columns to a consistent data type (e.g., float)
    weighted_sum = convert_to_consistent_dtype(weighted_sum)

    # Set 'id' column as the index
    weighted_sum = set_id_as_index(weighted_sum)

    # Create and save the resulting weighted sum DataFrame
    submission_df = create_submission_df(weighted_sum)
    # save_submission_df(submission_df)
    submission_df.to_parquet(par['output'])

main1()
main2()
main3()