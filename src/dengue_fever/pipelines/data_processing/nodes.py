import pandas as pd


def merge_features_and_label(feature_train: pd.DataFrame, feature_test: pd.DataFrame, label_train:pd.DataFrame) -> pd.DataFrame:
    """Concatenate feature tables and merge the train labels."""
    feature_train.loc[:, "data_type"] = "train"
    feature_test.loc[:, "data_type"] = "test"
    total_features = pd.concat([feature_train, feature_test], axis=0)
    total_data = pd.merge(total_features, label_train, on=["city", "year", "weekofyear"], how="left")
    return total_data

def encoding(data: pd.DataFrame) -> pd.DataFrame:
    """Perform encoding on the data."""
    import numpy as np

    # one-hot encoding for city
    data_encoded = pd.get_dummies(data, columns=["city"], drop_first=True)

    # cyclical encoding for weekofyear
    data_encoded['weekofyear_sin'] = np.sin(2*np.pi*data_encoded['weekofyear']/52)
    data_encoded['weekofyear_cos'] = np.cos(2*np.pi*data_encoded['weekofyear']/52)
    return data_encoded

def dropping_columns(data: pd.DataFrame, list_of_columns_to_drop: list[str]) -> pd.DataFrame:
    """Drop the columns from the data."""
    data_dropped = data.drop(columns=list_of_columns_to_drop)
    return data_dropped

def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in the data, with the exception of the 'train' column."""

    from sklearn.impute import KNNImputer

    # Separate the 'train' column from the rest of the DataFrame
    train_column = data[['data_type', 'total_cases']]
    data = data.drop(columns=['data_type', 'total_cases'])

    # Perform the imputation
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data)

    # Convert the imputed data back to a DataFrame and re-add the 'train' column
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
    data_imputed[['data_type', 'total_cases']] = train_column

    return data_imputed

def scale(data: pd.DataFrame) -> pd.DataFrame:
    """Scale the data using MinMaxScaler."""

    from sklearn.preprocessing import MinMaxScaler

    # Separate the 'train' column from the rest of the DataFrame

    train_column = data[['data_type', 'total_cases']]
    data = data.drop(columns=['data_type', 'total_cases'])

    # Perform the scaling

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame and re-add the 'train' column

    data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    data_scaled[['data_type', 'total_cases']] = train_column

    return data_scaled
