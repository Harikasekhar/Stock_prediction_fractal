import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.data_collection import *


def preprocess_stock_data(stock_data):
    """
    Function to preprocess stock data obtained from Yahoo Finance.

    Args:
    stock_data (DataFrame): Pandas DataFrame containing stock data.

    Returns:
    DataFrame: Preprocessed stock data.
    """
    try:
        # Drop rows with missing values
        stock_data.dropna(inplace=True)

        # Drop any additional columns that may not be needed for analysis
        # For example, if you only want to keep 'Open', 'High', 'Low', 'Close', 'Volume'
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Optionally, we can perform additional preprocessing steps such as:
        # - Feature scaling
        # - Handling categorical variables
        # - Handling outliers
        # - Feature engineering (creating new features)

        return stock_data
    except Exception as e:
        print("An error occurred during preprocessing:", str(e))
        return None

# df = preprocess_stock_data('TSLA','1mo')
# print(df.head(5))
