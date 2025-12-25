import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from the data path
    """
    def __init__(self,data_path:str):
        """
        Args:
            data_path: path to dta
        """
        self.data_path=data_path

    def get_data(self):
        """
        Ingesting the data from data_path
        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)

@step
def ingest_df(datapath:str)->pd.DataFrame:
    """
    ingesting the data from data_path
    Args:
        datapath (str): the path of the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data=IngestData(datapath)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting the data : {e}')
        raise e

