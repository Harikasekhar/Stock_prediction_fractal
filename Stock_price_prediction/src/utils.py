import pickle, os


import os
import pandas as pd

def save_raw_data(data, stock_name, data_dir):
    """
    Save raw data to a CSV file.

    Args:
    data (DataFrame): The raw data to be saved.
    stock_name (str): The name of the stock.
    data_dir (str): The directory where the data will be saved.
    """
    try:
        # Ensure that the directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Construct the filename
        filename = os.path.join(data_dir, f'{stock_name}_raw_data.csv')
        
        # Save the data
        data.to_csv(filename, index=False)
        
        print("Raw data saved successfully as:", filename)
    except Exception as e:
        print("An error occurred while saving the raw data:", str(e))


def save_processed_data(data, stock_name, data_dir):
    """
    Save processed data to a CSV file.

    Args:
    data (DataFrame): The processed data to be saved.
    stock_name (str): The name of the stock.
    data_dir (str): The directory where the data will be saved.
    """
    try:
        # Ensure that the directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Construct the filename
        filename = os.path.join(data_dir, f'{stock_name}_processed_data.csv')
        
        # Save the data
        data.to_csv(filename, index=False)
        
        print("Processed data saved successfully as:", filename)
    except Exception as e:
        print("An error occurred while saving the processed data:", str(e))


def save_model(model, model_dir):
    """
    Save the model to a file using pickle.

    Args:
    model (object): The trained model object.
    model_dir (str): The directory where the model will be saved.
    """
    try:
        # Ensure that the directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Construct the filename
        filename = os.path.join(model_dir, 'best_model.pkl')
        
        # Save the model
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        print("Model saved successfully as:", filename)
    except Exception as e:
        print("An error occurred while saving the model:", str(e))








