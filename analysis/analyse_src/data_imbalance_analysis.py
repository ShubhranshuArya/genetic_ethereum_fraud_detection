from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Abstract Base Class for Data Imbalance Analysis
# -----------------------------------------------
# This class defines a template for analyzing imbalance in the data.
# Subclasses must implement the methods to identify and visualize missing values.
class DataImbalanceAnalysisTemplate(ABC):
    @abstractmethod
    def visualize_data_imbalance(self, df: pd.DataFrame):
        pass


# Concrete Class for Data Imbalance Identification
# -------------------------------------------------
# This class implements methods to identify and visualize imbalance in the dataset.
class DataImbalanceAnalysis(DataImbalanceAnalysisTemplate):
    def visualize_data_imbalance(self, df: pd.DataFrame):
        # Count the occurrences of each FLAG value
        flag_counts = df["FLAG"].value_counts().sort_index()

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=flag_counts.index, y=flag_counts.values)

        # Customize the plot
        plt.title("Distribution of FLAG Values", fontsize=16)
        plt.xlabel("FLAG", fontsize=12)
        plt.ylabel("Count", fontsize=12)

        # Add value labels on top of each bar
        for i, v in enumerate(flag_counts.values):
            plt.text(i, v, str(v), ha="center", va="bottom")

        # Improve readability
        plt.xticks(rotation=0)
        sns.despine()

        # Show the plot
        plt.tight_layout()
        plt.show()
