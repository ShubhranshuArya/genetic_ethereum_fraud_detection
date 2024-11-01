import pandas as pd
from abc import ABC, abstractmethod


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame):
        pass


class DataShapeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        print(data.shape)


class DataInfoInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        print(data.info())


class DataUniqueInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        for col in data:
            print(f"{col} : {len(data[col].unique())}")


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def execute_strategy(self, data: pd.DataFrame):
        self._strategy.inspect(data)
