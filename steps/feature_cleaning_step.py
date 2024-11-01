from zenml import step
import pandas as pd
from src.feature_cleaning import (
    FeatureCleaning,
    CategoricalFeatureCleaningStrategy,
    ImbalanceFeatureCleaningStrategy,
    NoUniqueFeatureCleaningStrategy,
    RenameFeatureCleaningStrategy,
    UnwantedFeatureCleaningStrategy,
    CorrelatedFeatureCleaningStrategy,
)


@step
def feature_cleaning_step(
    df: pd.DataFrame,
    target_column: str,
    unwanted_feature_list: list = [],
) -> pd.DataFrame:
    """
    Applies the specified feature cleaning strategy to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing features to be cleaned.

    Returns:
        pd.DataFrame: The DataFrame with features cleaned according to the specified strategy.
    """
    copy_df = df.copy()

    # Remove unwanted features
    feature_cleaning = FeatureCleaning(
        UnwantedFeatureCleaningStrategy(feature_list=unwanted_feature_list)
    )
    wanted_features = feature_cleaning.apply_strategy(copy_df)

    # Remove categorical features
    feature_cleaning.set_strategy(CategoricalFeatureCleaningStrategy())
    numerical_features = feature_cleaning.apply_strategy(wanted_features)

    # Remove features with less than 5 unique values
    feature_cleaning.set_strategy(NoUniqueFeatureCleaningStrategy())
    unique_features = feature_cleaning.apply_strategy(numerical_features)

    # Remove highly correlated features
    feature_cleaning.set_strategy(CorrelatedFeatureCleaningStrategy())
    non_correlated_features = feature_cleaning.apply_strategy(unique_features)

    # Fix imbalance in the dataset
    feature_cleaning.set_strategy(
        ImbalanceFeatureCleaningStrategy(target_column=target_column)
    )
    balanced_features = feature_cleaning.apply_strategy(non_correlated_features)

    # Rename features as per Industry Standards
    feature_cleaning.set_strategy(RenameFeatureCleaningStrategy())
    cleaned_df = feature_cleaning.apply_strategy(balanced_features)

    return cleaned_df
