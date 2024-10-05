import numpy as np
import pandas as pd

import logging

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """
        Evaluate the model's performance.
        """
        try:
            predictions = self.model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            r2 = r2_score(self.y_test, predictions)
            direction_actual = np.where(self.y_test.values > self.X_test['Close_Lag1'], 1, 0)
            direction_pred = np.where(predictions > self.X_test['Close_Lag1'], 1, 0)
            directional_accuracy = np.mean(direction_actual == direction_pred)
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Directional Accuracy': directional_accuracy
            }
            logging.info(f"Model Evaluation Metrics: {metrics}")
            return metrics, predictions
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            return None, None