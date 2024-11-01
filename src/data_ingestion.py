import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        This Concrete method should be implemented by subclasses to ingest data from a given file path.

        Args:
            file_path (str): The path to the file to be ingested.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the ingested data.
        """
        pass


class ZipCSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from a given file path which can be either in zip or csv format.

        Args:
            file_path (str): The path to the file to be ingested.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the ingested data.
        """
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as z:
                # Assuming there is only one file in the zip
                file_name = z.namelist()[0]
                with z.open(file_name) as f:
                    return pd.read_csv(f)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            raise ValueError(
                "Unsupported file format. Only .zip and .csv are supported."
            )


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        """
        Factory method to create a DataIngestor based on the file extension.

        Args:
            file_path (str): The path to the file to be ingested.

        Returns:
            DataIngestor: An instance of a subclass of DataIngestor.
        """
        if file_path.endswith(".zip") or file_path.endswith(".csv"):
            return ZipCSVDataIngestor()
        else:
            raise ValueError(
                "Unsupported file format. Only .zip and .csv are supported."
            )
