import logging
import pandas as pd
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStrategy

def get_data_for_test():
    try:
        df=pd.read_csv("data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        preprocessing_strategy=DataPreProcessStrategy()
        data_cleaner=DataCleaning(df,preprocessing_strategy)
        df=data_cleaner.handle_data()
        df.drop(["review_score"],axis=1,inplace=True)
        result=df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e