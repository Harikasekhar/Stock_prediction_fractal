import unittest
import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.data_collection import *

class TestGetStockData(unittest.TestCase):
    def test_get_stock_data(self, stock_name,period):
        # Test case: check if the function returns a DataFrame

        stock_data = get_stock_data(stock_name,period)
        self.assertIsInstance(stock_data, pd.DataFrame)
        

        # Test case: check if the returned DataFrame has expected columns
        expected_columns = ['Date','Open', 'High', 'Low', 'Close','Volume','Dividends','Stock Splits']
        self.assertListEqual(stock_data.columns.tolist(), expected_columns)
        

        # Test case: check if the returned DataFrame is not empty
        self.assertFalse(stock_data.empty)


if __name__ == '__main__':
    unittest.main()
