from utils import *
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
            submission_df.to_csv(f"result_{n_components}_{d_model}.csv", index=False)

    return


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument('--config', type=str, help="Path to the YAML config file.", default='config_train.yaml')
    args = parser.parse_args()

    # Check if the config file is provided
    if not args.config:
        print("Please provide a config file using --config.")
        return

    # Load and print configurations
    config_file = args.config
    config = load_and_print_config(config_file)

    # Access specific values from the config
    n_components_list = config.get('n_components_list', [])
    d_models_list = config.get('d_models_list', [])  # embedding dimensions for the transformer models
    batch_size = config.get('batch_size', 32)
    data_file = config.get('data_file', '')
    id_map_file = config.get('id_map_file', '')
    device = config.get('device', 'cuda')
    models_dir = config.get('dir', 'model_1_mean_std_only')
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


if __name__ == "__main__":
    main()
