import logging
import mlflow
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_train(
    X_train:pd.DataFrame,
    y_train:pd.Series,
    X_test:pd.DataFrame,
    y_test:pd.Series,
    config: ModelNameConfig
)->RegressorMixin:
    """
    Trains the model
    Args:
    X_train:training features
    y_train:training labels
    X_test:testing features
    y_test:testing labels
    config:model configuration parameters
    
    Returns:
    model:trained model
    """
    try:
        model=None
        model_name=config.model_name
        if model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model_trainer = LinearRegressionModel(model_name)
            model = model_trainer.train(X_train, y_train)
            logging.info("Model training completed successfully")
            return model
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    except Exception as e:
        logging.error(f"Error in model_train step: {e}")
        raise e
    
