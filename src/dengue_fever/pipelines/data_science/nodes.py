from typing import Tuple
import numpy as np
import pandas as pd
# fix the line below
from sklearn.ensemble import RandomForestRegressor

def train_model(data: pd.DataFrame, model_options) -> RandomForestRegressor:
    """Trains the random forest regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.

        RandomForestRegressor(n_estimators=200,max_depth=7,min_samples_leaf=4,min_samples_split=5)
    """
    # Filter out test data and drop data_type column
    data = data[data['data_type'] == 'train']
    data = data.drop('data_type', axis=1)

    X = data.drop('total_cases',axis=1)
    y = data['total_cases']

    regressor = RandomForestRegressor(n_estimators=model_options['n_estimators'],
        max_depth=model_options['max_depth'],min_samples_leaf=model_options['min_samples_leaf'],
        min_samples_split=model_options['min_samples_split'])

    regressor.fit(X,y)

    return regressor

def make_prediction(
    data: pd.DataFrame, regressor: RandomForestRegressor, submission_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Make predictions using the trained model.

    Args:
        data: Data on which to make predictions.
        regressor: Trained model.
        submission_data: Data formatted for submission.

    Returns:
        The submission data DataFrame updated with the predictions.
    """
    # Filter out test data and drop data_type column
    data = data[data['data_type'] == 'test']
    data = data.drop(['data_type', 'total_cases'], axis=1)

    y_pred = np.round(regressor.predict(data))

    submission_data['total_cases'] = y_pred.astype(int)

    return submission_data

