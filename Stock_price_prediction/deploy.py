from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

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

# Define the predict route
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
    app.run(host='127.0.0.1', port=5000, debug=True)
