from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import model_train
from steps.evaluation import evaluate_model


@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    trained_model = model_train(X_train, y_train, X_test, y_test)
    mse, r2, rmse = evaluate_model(trained_model, X_test, y_test)
    
   

