import unittest
import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.model_evaluation import evaluate_regression_model

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Load test data and predictions from another folder
        test_data_folder = 'C:\\Users\\sg1402-dsk05-user1\\Desktop\\Capstone_Project\\notebooks'
        self.test_x = np.load(os.path.join(test_data_folder, 'X_test.npy'))
        self.y_test = np.load(os.path.join(test_data_folder, 'y_test.npy'))
        self.y_pred = np.load(os.path.join(test_data_folder, 'y_pred.npy'))
        
    def test_evaluate_regression_model(self):
        # Test evaluation function
        metrics = evaluate_regression_model(self.y_test, self.y_pred)
        
        # Check if metrics dictionary is not empty
        self.assertNotEqual(len(metrics), 0)
        
        # Check if each metric has a value
        for metric_name, metric_value in metrics.items():
            with self.subTest(metric_name=metric_name):
                self.assertIsNotNone(metric_value)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
