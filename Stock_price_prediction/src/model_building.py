from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle
from src.model_evaluation import *

def perform_linear_regression(X_train, X_test, y_train, y_test):
    """
    Function to perform linear regression on stock data.

    Args:
    X_train (DataFrame): Features of the training set.
    X_test (DataFrame): Features of the testing set.
    y_train (DataFrame): Target variable of the training set.
    y_test (DataFrame): Target variable of the testing set.

    Returns:
    tuple: Tuple containing the trained model, training features, testing features,
           training target variable, and testing target variable.
    """

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test



def perform_tuned_regression(train_x, test_x, train_y, test_y):
    # Define the model
    regression_model = Ridge()
    
    # Define hyperparameters for tuning
    hyperparameters = {'alpha': [0.1, 1.0, 10.0]}
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(regression_model, hyperparameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train the best model
    best_model.fit(train_x, train_y)
    
    # Evaluate the model
    predicted_prices = best_model.predict(test_x)
    
    # Calculate confidence (R-squared)
    confidence_after_tuning = best_model.score(test_x, test_y)
    
    # Evaluate other metrics
    evaluation_metrics_after_tuning = evaluate_model(test_y, predicted_prices)
    
    return best_model, confidence_after_tuning, predicted_prices, evaluation_metrics_after_tuning


def train_xgboost_model(X_train, y_train, X_test, y_test):
    # Training the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror',reg_alpha=0.1)
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test 


def train_xgboost_model_tuned(X_train, y_train, X_test, y_test, params):
    model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


