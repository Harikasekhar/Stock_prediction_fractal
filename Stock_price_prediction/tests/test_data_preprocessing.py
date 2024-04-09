import unittest
import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.data_preprocessing import *


class TestPreprocessStockData(unittest.TestCase):
    stock_data = get_stock_data('TSLA','2mo')
    def test_drop_missing_values(self):
        processed_data = preprocess_stock_data(self.stock_data)
        print(self.stock_data.shape)
        print(processed_data.shape)
        self.assertEqual(processed_data.shape[1],5)  # Expecting 2 rows after dropping missing values

    def test_columns_retained(self):
        # Test case: check if the function retains the correct columns
        processed_data = preprocess_stock_data(self.stock_data)
        self.assertListEqual(processed_data.columns.tolist(), ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_return_type(self):
        # Test case: check if the function returns a DataFrame
        processed_data = preprocess_stock_data(self.stock_data)
        self.assertIsInstance(processed_data, pd.DataFrame)

    def test_empty_input(self):
        # Test case: check if the function returns None for empty input
        data = pd.DataFrame()  # Empty DataFrame
        processed_data = preprocess_stock_data(self.stock_data)
        self.assertIsNotNone(processed_data)

if __name__ == '__main__':
    unittest.main()
