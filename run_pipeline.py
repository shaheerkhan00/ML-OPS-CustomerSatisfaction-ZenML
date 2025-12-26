from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")

#mlflow ui --backend-store-uri "file:C:\Users\SHAHEERk\AppData\Roaming\zenml\local_stores\29fcae73-1dff-445f-a6d1-5c51c0ad4b1d\mlruns"


