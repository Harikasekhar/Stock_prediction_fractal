from flask import Flask, request, jsonify
import pickle,sys,os
import numpy as np
import pandas as pd
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.data_collection import get_stock_data
from src.model_building import perform_linear_regression, perform_tuned_regression,train_xgboost_model_tuned,train_xgboost_model
from src.data_preprocessing import preprocess_stock_data
from src.feature_engineering import cal_ex_moving_avg, calculate_rsi
from src.model_evaluation import evaluate_model
from src.utils import save_model,save_raw_data,save_processed_data


app = Flask(__name__)

# Define the directory where the models are stored
model_dir = 'D:\\Python\\Stock_price_prediction\\Stock_price_prediction\\models'

# Assuming you have the stock name and model type
stock_name = "NIO"
model_type = "Linear Before Tuning"

# Construct the file path
model_filename = f"{stock_name}_{model_type}_model.pkl"
model_path = os.path.join(model_dir, model_filename)

# Load the model
with open(model_path, 'rb') as f:
    best_model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON request data
    data = request.get_json()

    # Extract input features from the request data
    features = data['features']

    # Convert features to DataFrame
    features_df = pd.DataFrame(features)

    # Make predictions using the best model
    predictions = best_model.predict(features_df)

    # Return the predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000,debug=True)

