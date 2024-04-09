import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.model_building import perform_linear_regression, save_linear_regression_model

class TestLinearRegression(unittest.TestCase):
    def test_perform_linear_regression(self):
        # Load test data from file
        file_path = 'C:\\Users\\sg1402-dsk05-user1\\Desktop\\Capstone_Project\\data\\processed\\Processed_data.csv'
        stock_data = pd.read_csv(file_path)

        # Perform linear regression
        model, X_train, X_test, y_train, y_test = perform_linear_regression(stock_data)

        # Check if model is instance of LinearRegression
        self.assertIsInstance(model, LinearRegression)

        # Check if X_train, X_test, y_train, y_test are DataFrames/Series
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

        # Check if training and testing data have correct shapes
        self.assertEqual(X_train.shape[1], 6)
        self.assertEqual(X_test.shape[1], 6)
        

    def test_save_linear_regression_model(self):
        # Load test data from file
        file_path = 'C:\\Users\\sg1402-dsk05-user1\\Desktop\\Capstone_Project\\data\\processed\\Processed_data.csv'
        stock_data = pd.read_csv(file_path)

        # Perform linear regression
        model, _, _, _, _ = perform_linear_regression(stock_data)
        filename = 'linear_regression_model.pkl'

        # Save the model
        save_linear_regression_model(model, filename)

        # Check if file exists
        self.assertTrue(os.path.exists(filename))

if __name__ == '__main__':
    unittest.main()