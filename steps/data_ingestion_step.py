import pandas as pd
from zenml import step
from src.data_ingestion import DataIngestorFactory


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Ingest data from a given file path as a zenml step which can be either in zip or csv format.

    Args:
        file_path (str): The path to the file to be ingested.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the ingested data.
    """
    ingestor = DataIngestorFactory.get_data_ingestor(file_path)
    return ingestor.ingest(file_path)
