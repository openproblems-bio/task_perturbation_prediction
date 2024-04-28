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


def main():
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
    save_submission_df(submission_df)


if __name__ == '__main__':
    main()
