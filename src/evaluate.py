import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract class for model evaluation"""
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass
    
class MSE(Evaluation):
    """MSE evaluation Strategy"""
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Evaluates the model using Mean Squared Error
        Args:
        y_true:true labels
        y_pred:predicted labels
        Returns:
        mse:Mean Squared Error
        """
        try:
            mse = mean_squared_error(y_pred,y_true)
            
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in MSE evaluation: {e}")
            raise e

class R2Score(Evaluation):
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Evaluates the model using R2 Score
        Args:
        y_true:true labels
        y_pred:predicted labels
        Returns:
        r2_score:R2 Score
        """
        try:
            r2 = r2_score(y_true,y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in R2 Score evaluation: {e}")
            raise e

class RMSE(Evaluation):
    """RMSE evaluation Strategy"""
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Evaluates the model using Root Mean Squared Error
        Args:
        y_true:true labels
        y_pred:predicted labels
        Returns:
        rmse:Root Mean Squared Error
        """
        try:
            rmse = (np.mean((y_true - y_pred) ** 2))**0.5
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in RMSE evaluation: {e}")
            raise e

class ModelEvaluator:
    """Class for evaluating models using different strategies"""
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, strategy:Evaluation):
        self.y_true = y_true
        self.y_pred = y_pred
        self.strategy = strategy

    def evaluate(self):
        """
        Evaluates the model using the provided strategy
        Returns:
        score:evaluation score
        """
        try:
            score = self.strategy.calculate_scores(self.y_true, self.y_pred)
            return score
        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            raise e