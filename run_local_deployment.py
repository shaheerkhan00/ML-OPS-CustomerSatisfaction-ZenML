from pipelines.local_deployment_pipeline import (
    local_continous_deployment_pipeline, 
    local_inference_pipeline
)

import click 

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config", 
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Run 'deploy' for training, 'predict' for inference."
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy for the model."
)
def run_local_deployment(config: str, min_accuracy: float):
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        print("🚀 Running Training Pipeline...")
        local_continous_deployment_pipeline(min_accuracy=min_accuracy, workers=3, timeout=60)
        print("✅ Training finished!")

    if predict:
        print("🔮 Running Inference Pipeline (Local)...")
        # No arguments needed anymore!
        local_inference_pipeline()
        print("✅ Inference finished! Check the logs above for results.")

if __name__ == "__main__":
    run_local_deployment()