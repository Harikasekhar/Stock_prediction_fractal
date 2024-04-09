from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Define the directory where the models are stored
model_path = 'D:\\Python\\Stock_price_prediction\\Stock_price_prediction\\models\\best_model.pkl'

# Load the model
with open(model_path, 'rb') as f:
    best_model = pickle.load(f)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON request data
    data = request.get_json()

    # Check if 'features' key is present in the JSON data
    if 'features' not in data:
        return jsonify({'error': 'No features found in the request data'})

    # Assuming data contains features as a list of dictionaries
    features = data['features']

    # Convert features to DataFrame
    features_df = pd.DataFrame(features)

    # Preprocess features if needed (e.g., scaling, encoding)
    # Ensure that the preprocessing steps match those used during training

    # Make predictions using the model
    predictions = best_model.predict(features_df)

    # Return the predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
