from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging


# Abstract base class to clean features from the dataset.
class FeatureCleaningStrategy(ABC):
    @abstractmethod
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to remove features from the dataset.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing unwanted features.

        Returns:
            pd.DataFrame: The DataFrame with model relevant features.
        """
        pass


# Concrete strategy to remove unwanted features.
class UnwantedFeatureCleaningStrategy(FeatureCleaningStrategy):
    def __init__(self, feature_list: list):
        """
        Initializes the UnwantedFeatureCleaningStrategy with specific parameters.

        Parameters:
            feature_list (list): List of unwanted features from the dataset.
        """
        self.feature_list = feature_list

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns with provided the feature list.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing unwanted features.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logging.info(f"Dropping unwanted features")
        for feature in self.feature_list:
            if feature in df.columns:
                df = df.drop(feature, axis=1)
                logging.info(f"Dropped feature: {feature}")
            else:
                logging.warning(f"Feature '{feature}' does not exist in the DataFrame.")
        logging.info("Unwanted values dropped.")
        return df


# Concrete strategy to remove categorical features.
class CategoricalFeatureCleaningStrategy(FeatureCleaningStrategy):
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops categorical features from the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing categorical features.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logging.info(f"Dropping categorical features")
        categorical_features = df.select_dtypes("O").columns.astype("category")
        cleaned_df = df.drop(categorical_features, axis=1)
        logging.info("categorical values dropped.")
        return cleaned_df


# Concrete strategy to remove features with less than 5 unique values.
class NoUniqueFeatureCleaningStrategy(FeatureCleaningStrategy):
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops features with less than 5 unique values from the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing features with less than 5 unique values.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logging.info(f"Dropping features with less than 5 unique values")
        zero_feature_column = [
            i for i in df.columns[1:] if len(df[i].value_counts()) <= 5
        ]
        cleaned_df = df.drop(zero_feature_column, axis=1)
        logging.info("Features with less than 5 unique values dropped.")
        return cleaned_df


# Concrete strategy to rename features with industry standards.
class RenameFeatureCleaningStrategy(FeatureCleaningStrategy):
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the features in the DataFrame according to industry standards.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing features to be renamed.

        Returns:
            pd.DataFrame: The DataFrame with features renamed.
        """
        logging.info(f"Renaming features with industry standards")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.columns = df.columns.str.replace("[^a-zA-Z0-9_]", "")
        logging.info("Features renamed.")
        return df


# Concrete strategy to identify and remove highly correlated features.
class CorrelatedFeatureCleaningStrategy(FeatureCleaningStrategy):
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes highly correlated features from the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing highly correlated features.

        Returns:
            pd.DataFrame: The DataFrame with highly correlated features removed.
        """
        logging.info(f"Removing highly correlated features")
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find index of feature with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

        # Drop features
        cleaned_df = df.drop(to_drop, axis=1)
        logging.info("Highly correlated features removed.")
        return cleaned_df


class ImbalanceFeatureCleaningStrategy(FeatureCleaningStrategy):
    def __init__(self, target_column: str):
        """
        Initializes the ImbalanceFeatureCleaningStrategy with the target column.

        Parameters:
            target_column (str): The name of the target column.
        """
        self.target_column = target_column

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the imbalance in the target column by downsampling the majority class.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the target column.

        Returns:
            pd.DataFrame: The DataFrame with the target column balanced.
        """
        logging.info(f"Reducing imbalance in target column '{self.target_column}'")

        # Find the unique value with the least number of values
        min_count = df[self.target_column].value_counts().min()

        # Downsample the majority class
        balanced_df = pd.concat(
            [
                df[df[self.target_column] == value].sample(min_count)
                for value in df[self.target_column].unique()
            ]
        )

        logging.info("Target column balanced.")
        return balanced_df


# Context class to set and implement the strategies.
class FeatureCleaning:
    def __init__(self, strategy: FeatureCleaningStrategy):
        """
        Initializes the FeatureCleaning context with a specific strategy.

        Parameters:
            strategy (FeatureCleaningStrategy): The strategy to use for feature cleaning.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureCleaningStrategy):
        """
        Set the strategy for feature cleaning.

        Parameters:
            strategy: The strategy to use for feature cleaning.
        """
        logging.info("Switching feature cleaning strategy.")
        self._strategy = strategy

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle feature cleaning in the DataFrame using the set strategy.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing feature cleaning.

        Returns:
            pd.DataFrame: The DataFrame with feature cleaning handled according to the strategy.
        """
        logging.info("Executing feature cleaning strategy.")
        return self._strategy.apply_strategy(df)
