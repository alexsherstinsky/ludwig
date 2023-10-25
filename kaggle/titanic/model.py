from __future__ import annotations

import logging
import pathlib

import pandas as pd
import yaml

from ludwig.api import LudwigModel, TrainingResults
from ludwig.datasets import titanic


def convert_dictionary_to_pandas_dataframe(sample_dictionary: dict) -> pd.DataFrame:
    """Converts each saclar dictionary value to list of length one, containing that value.

    Then the resulting dictionary is used to instantiate a Pandas DataFrame, which is returned to be used as "dataset"
    argument to "model.predict()".
    """
    sample_data_items: list[tuple] = sample_dictionary.items()
    sample_data_items = [(item[0], [item[1]]) for item in sample_data_items]
    df_sample: pd.DataFrame = pd.DataFrame(data=dict(sample_data_items))

    return df_sample


class TitanicModel:
    def __init__(
        self,
        output_directory: str = "results",
        model_directory_relative_path: pathlib.Path = pathlib.Path("api_experiment_run/model"),
    ):
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

        self._titanic_inoput_features: list[str] = self._get_input_features()

        self._df_training_set, self._df_test_set = self.get_train_and_test_dataframes()

        self._output_directory: str = output_directory

        self._model_directory_path: pathlib.Path = pathlib.Path(self._output_directory) / model_directory_relative_path

    @property
    def config(self) -> dict:
        return self._titanic_config

    @property
    def full_config(self) -> dict:
        return self._model.config_obj.to_dict()

    @property
    def input_features(self) -> list[str]:
        return self._titanic_inoput_features

    @property
    def df_training_set(self) -> pd.DataFrame:
        return self._df_training_set

    @property
    def df_test_set(self) -> pd.DataFrame:
        return self._df_test_set

    @property
    def output_directory(self) -> str:
        return self._output_directory

    @property
    def model_directory_relative_path(self) -> pathlib.Path:
        return self._model_directory_relative_path

    def train(self, dataset: str | dict | pd.DataFrame) -> TrainingResults:
        return self._model.train(dataset=dataset, output_directory=self._output_directory)

    def predict(self, dataset: str | dict | pd.DataFrame) -> tuple[dict | pd.DataFrame, bool, str | None]:
        model_in_memory: bool = True
        model_directory_path_as_string: str | None = None

        predictions_and_probabilities: tuple[dict | pd.DataFrame, str]
        try:
            predictions_and_probabilities = self._model.predict(
                dataset=dataset, output_directory=self._output_directory
            )
        except ValueError:  # Model has not been trained or loaded.
            model_directory_path_as_string = self._model_directory_path.as_posix()
            self._model = LudwigModel.load(model_dir=model_directory_path_as_string)  # Use default saved model path.
            predictions_and_probabilities = self._model.predict(
                dataset=dataset, output_directory=self._output_directory
            )
            model_in_memory = False

        return predictions_and_probabilities[0], model_in_memory, model_directory_path_as_string

    @staticmethod
    def get_train_and_test_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
        df_training_set: pd.DataFrame
        df_test_set: pd.DataFrame
        df_training_set, df_test_set, _ = titanic.load(split=True)

        return df_training_set, df_test_set

    def _get_input_features(self) -> list[str]:
        input_features: list[str] = [feature.column for feature in self._model.config_obj.input_features]

        return input_features
