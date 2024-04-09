from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def model_prediction(test_x,model):
    predicted=model.predict(test_x)
    return predicted

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Function to calculate Mean Absolute Percentage Error (MAPE).

    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.

    Returns:
    float: Mean Absolute Percentage Error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    """
    Function to evaluate a regression model using various metrics.

    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.

    Returns:
    dict: Dictionary containing evaluation metrics (MSE, R-squared, MAPE)
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    evaluation_metrics = {
        "Mean Squared Error": mse,
        "R-squared Score": r2,
        "Mean Absolute Percentage Error (MAPE)": mape
    }

    return evaluation_metrics

