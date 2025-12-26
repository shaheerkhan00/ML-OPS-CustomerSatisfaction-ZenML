import logging
import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
from src.evaluate import ModelEvaluator,MSE,R2Score,RMSE
from typing import Tuple
from typing_extensions import Annotated


experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) ->Tuple[
    Annotated[float,"Mean Squared Error"],
    Annotated[float,"R2 Score"],    
    Annotated[float,"Root Mean Squared Error"]
]:
    
    """
    Evaluates the trained model
    Args:
    model:trained model
    X_test:testing features
    y_test:testing labels
    
    Returns:
    None
    """
    try:
        logging.info("Starting model evaluation")
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,predictions)
        mlflow.log_metric("Mean Squared Error", mse)
        r2_class = R2Score()
        r2 = r2_class.calculate_scores(y_test,predictions)
        mlflow.log_metric("R2 Score", r2)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,predictions)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        logging.info("Model evaluation completed successfully")
        return mse, r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluate_model step: {e}")
        raise e
