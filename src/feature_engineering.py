from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import logging


# Abstract base class for Feature Engineering
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
            df (pd.DataFrame): The dataFrame containing features to transform.

        Returns:
            pd.DataFrame: A dataFrame with the applied transformations.
        """
        pass


# Concrete strategy to normalize features in the dataset.
class NormalizeFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def apply_transformation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalizes the dataset on every feature except for the prediction feature.

        Parameters:
            df (pd.DataFrame): The dataFrame containing features to normalize.

        Returns:
            pd.DataFrame: A dataFrame with all features normalized except for the prediction feature.
        """
        # Exclude the prediction feature from normalization
        logging.info("Starting feature normalization process.")

        # Normalize each feature
        norm = PowerTransformer()

        X_norm = norm.fit_transform(X=df)
        X_norm_df = pd.DataFrame(X_norm, columns=df.columns)

        logging.info("Feature engineering completed")
        return X_norm_df


class FeatureEngineeringHandler:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineeringContext with a specific strategy.

        Parameters:
            strategy (FeatureEngineeringStrategy): The strategy to use for feature engineering.
        """
        self._strategy = strategy

    @property
    def strategy(self) -> FeatureEngineeringStrategy:
        """
        Gets the current strategy.

        Returns:
           FeatureEngineeringStrategy: The current strategy.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets the strategy.

        Parameters:
            strategy (FeatureEngineeringStrategy): The new strategy.
        """
        self._strategy = strategy

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the feature engineering transformation using the current strategy.

        Parameters:
            df (pd.DataFrame): The dataFrame containing features to transform.
            prediction_feature (str): The name of the prediction feature.

        Returns:
            pd.DataFrame: A dataFrame with the applied transformations.
        """
        return self._strategy.apply_transformation(df)
