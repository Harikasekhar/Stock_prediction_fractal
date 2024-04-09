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
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Handle GET request
        return jsonify({'message': 'GET method not allowed'})

    # Handle POST request
    data = request.get_json()
    # Process the data and return predictions

    return jsonify({'predictions': data.tolist()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
