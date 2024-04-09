import unittest
import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.feature_engineering import cal_ex_moving_avg, calculate_rsi
from src.data_collection import get_stock_data

class TestMovingAverageAndRSI(unittest.TestCase):
    data = get_stock_data('TSLA','2mo')
    def test_cal_ex_moving_avg(self):
        # Test case for cal_ex_moving_avg function
        
        # Perform exponential moving average calculation
        ema_window = 5
        result_df = cal_ex_moving_avg(self.data, ema_window)
        
        # Check if the 'EMA' column is added to the DataFrame
        self.assertTrue('EMA' in result_df.columns)
        
        # Check if the length of the DataFrame remains the same
        self.assertEqual(len(result_df), len(self.data))
        
    def test_calculate_rsi(self):
        # Test case for calculate_rsi function
        
        # Perform RSI calculation
        window_days = 7
        result_df = calculate_rsi(self.data,window_days)
        
        # Check if the required columns are added to the DataFrame
        self.assertTrue(all(col in result_df.columns for col in ['Price_Change', 'Gain', 'Loss', 'Avg_Gain', 'Avg_Loss', 'RS', 'RSI']))

if __name__ == '__main__':
    unittest.main()
