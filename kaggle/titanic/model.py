import logging

# TODO: <Alex>ALEX</Alex>
# import numpy as np
# import pandas as pd
# TODO: <Alex>ALEX</Alex>
import yaml

# TODO: <Alex>ALEX</Alex>
# import ludwig
from ludwig.api import LudwigModel, TrainingResults

# TODO: <Alex>ALEX</Alex>
from ludwig.datasets import titanic


class TitanicModel:
    def __init__(self):
        titanic_config = yaml.safe_load(
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
        self._model = LudwigModel(config=titanic_config, logging_level=logging.INFO)

        self._titanic_config = titanic_config

        self._df_training_set, self._df_test_set = self.get_train_and_test_dataframes()

    @property
    def config(self):
        return self._titanic_config

    @property
    def df_training_set(self):
        return self._df_training_set

    @property
    def df_test_set(self):
        return self._df_test_set

    def train(self, df_dataset) -> TrainingResults:
        return self._model.train(dataset=df_dataset)

    def predict(self, df_dataset):
        return self._model.predict(dataset=df_dataset)

    @staticmethod
    def get_train_and_test_dataframes():
        df_training_set, df_test_set, _ = titanic.load(split=True)
        return df_training_set, df_test_set
