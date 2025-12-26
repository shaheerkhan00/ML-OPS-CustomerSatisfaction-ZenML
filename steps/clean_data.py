import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_Train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"], 
    Annotated[pd.Series,"y_test"]
]:
    """
    Cleans the data and divides it into training and testing sets
    Args:
    df:Raw data frame
    Returns:
    X_train:training features
    X_test:testing features 
    y_train:training labels
    y_test:testing labels
    """
    try:
        #preprocessing
        logging.info("Starting data preprocessing")
        preprocessing_strategy=DataPreProcessStrategy()
        data_cleaner=DataCleaning(df,preprocessing_strategy)
        preprocessed_data=data_cleaner.handle_data()
        #dividing
        logging.info("Dividing data into train and test sets")
        dividing_strategy=DataDivideStrategy()
        data_divider=DataCleaning(preprocessed_data,dividing_strategy)
        X_train,X_test,y_train,y_test=data_divider.handle_data()
        logging.info("Data cleaning completed successfully")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in clean_df step: {e}")
        raise e



