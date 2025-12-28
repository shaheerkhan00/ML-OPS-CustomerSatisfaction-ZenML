import numpy as np
import pandas as pd
import json
from zenml import pipeline, step
from zenml.client import Client  # <--- Added Client import
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.steps import BaseParameters
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import model_train
from .utils import get_data_for_test

# --- 1. Define Helper Steps ---

@step
def model_loader() -> object:
    """Loads the model artifact from the last training run."""
    client = Client()
    # Get the last run of the training pipeline
    last_run = client.get_pipeline("continous_deployment_pipeline").last_run
    # Fetch the model artifact (output of the 'model_train' step)
    return last_run.steps["model_train"].output.load()

@step
def local_predictor(
    model: object,
    data: str
) -> np.ndarray:
    """Runs prediction locally using the model object directly."""
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential", "payment_installments", "payment_value", "price",
        "freight_value", "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ]
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    predictions = model.predict(df)
    return predictions

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy: float, min_acc: float = 0.92) -> bool:
    return accuracy > min_acc

# --- 2. The Pipelines ---

@pipeline(enable_cache=True)
def local_continous_deployment_pipeline(
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    data_path: str = "data/olist_customers_dataset.csv"   
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    trained_model = model_train(X_train, y_train, X_test, y_test)
    mse, r2, rmse = evaluate_model(trained_model, X_test, y_test)
    deployment_decision = deployment_trigger(accuracy=r2, min_acc=min_accuracy)

@pipeline(enable_cache=False)
def local_inference_pipeline():
    # LINKING STEPS DIRECTLY HERE
    data = dynamic_importer()
    model = model_loader()  
    prediction = local_predictor(model=model, data=data)