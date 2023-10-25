from __future__ import annotations

import logging

import pandas as pd
import yaml

from ludwig.api import LudwigModel, TrainingResults
from ludwig.datasets import titanic


class TitanicModel:
    def __init__(self):
        titanic_config: dict = yaml.safe_load(
            """
            input_features:
                - name: Pclass
                  type: category
                - name: Sex
                  type: category
                - name: Age
                  type: numerical
                  preprocessing:
                    missing_value_strategy: fill_with_mean
                - name: SibSp
                  type: numerical
                - name: Parch
                  type: numerical
                - name: Fare
                  type: numerical
                  preprocessing:
                    missing_value_strategy: fill_with_mean
                - name: Embarked
                  type: category

            output_features:
                - name: Survived
                  type: binary
            """
        )
        self._model: LudwigModel = LudwigModel(config=titanic_config, logging_level=logging.INFO)

        self._titanic_config: dict = titanic_config

        self._df_training_set, self._df_test_set = self.get_train_and_test_dataframes()

    @property
    def config(self) -> dict:
        return self._titanic_config

    @property
    def df_training_set(self) -> pd.DataFrame:
        return self._df_training_set

    @property
    def df_test_set(self) -> pd.DataFrame:
        return self._df_test_set

    def train(self, df_dataset: pd.DataFrame) -> TrainingResults:
        return self._model.train(dataset=df_dataset)

    def predict(self, df_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        predictions_and_probabilities: tuple[pd.DataFrame, pd.DataFrame] = self._model.predict(dataset=df_dataset)
        return predictions_and_probabilities

    @staticmethod
    def get_train_and_test_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
        df_training_set: pd.DataFrame
        df_test_set: pd.DataFrame
        df_training_set, df_test_set, _ = titanic.load(split=True)
        return df_training_set, df_test_set
