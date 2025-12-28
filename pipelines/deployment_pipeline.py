import json
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import model_train
from steps.config import ModelNameConfig
from .utils import get_data_for_test


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.92


@step
def local_predictor(
    model: object,  # This expects the generic sklearn model object
    data: str       # JSON string data
) -> np.ndarray:
    """Runs prediction locally using the model object directly."""
    # Convert JSON string back to dataframe
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
    
    # Run prediction
    predictions = model.predict(df)
    return predictions



@step(enable_cache=False)
def dynamic_importer()->str:
    data = get_data_for_test()
    return data


    
@step
def deployment_trigger(
    accuracy: float,
    min_acc: float =2
) -> bool:  # <--- Added type hint (Required by ZenML)
    """Implements a deployment trigger based on model accuracy and decide whether to deploy or not."""
    if accuracy <= min_acc:
        return True
    return False

class MLFlowDeoploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str="continuous_deployment_pipeline"
    step_name: str="mlflow_model_deployer_step"
    running : bool=True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name:str,
    step_name:str,
    running:bool=True,
    model_name:str="model"
)->MLFlowDeploymentService:
    """Gets the prediction service started by the deployment pipeline.
    Args:
        pipeline_name (str): name of the deployment pipeline
        step_name (str): name of the deployment step
        running (bool): whether the service should be running or not
        model_name (str): name of the model
    Returns:
        MLFlowDeploymentService: the prediction service
    """
    #get the active model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No running service found for pipeline '{pipeline_name}', "
            f"step '{step_name}', and model '{model_name}'."
        )
    print(f"Found {len(existing_services)} running service(s).")
    # return the first service found    
    return existing_services[0]


@step
def predictor(
    service:MLFlowDeploymentService,
    data:np.ndarray
)->np.ndarray:
    """Makes predictions using the deployed model service.
    Args:
        MLFlow_service (MLFlowDeploymentService): the deployed model service
        Data (np.ndarray): input data for prediction
    Returns:
        np.ndarray: predictions from the model
    """
    service.start(timeout=10)
    data=json.loads(data)
    data.pop("columns")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df=pd.DataFrame([data],columns=columns_for_df)
    json_list= json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.ndarray(json_list)
    predictions = service.predict(data)
    return predictions



 
@pipeline(
    enable_cache=False, 
    settings={"docker": docker_settings}
)
def continous_deployment_pipeline(
    min_accuracy:float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    data_path:str = "data/olist_customers_dataset.csv"   
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    trained_model = model_train(X_train, y_train, X_test, y_test)
    mse, r2, rmse = evaluate_model(trained_model, X_test, y_test)
    trigger_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    deployment_decision = deployment_trigger(accuracy=mse)
   
    mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
    

@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings}
)
def inference_pipeline(
    pipeline_name:str,
    step_name:str,
):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        step_name=step_name
    )
    prediction = predictor(service=service,data=data)
    return prediction