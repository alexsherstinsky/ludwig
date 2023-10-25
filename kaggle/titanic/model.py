from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd
import yaml

from ludwig.api import LudwigModel, TrainingResults
from ludwig.datasets import titanic

DEFAULT_RANDOM_STATE: int = 200


def convert_dictionary_to_pandas_dataframe(sample_dictionary: dict) -> pd.DataFrame:
    """Converts each saclar dictionary value to list of length one, containing that value.

    Then the resulting dictionary is used to instantiate a Pandas DataFrame, which is returned to be used as "dataset"
    argument to "model.predict()".

    # Inputs

    :param :sample_dictionary (dict): input dictionary with each value being a scalar (not Iterable)

    # Return

    :return (pd.DataFrame): resulting Pandas DataFrame (after each value of the dictionary is turned into a list of one)
    """
    sample_data_items: list[tuple] = sample_dictionary.items()
    # TODO: <Alex>Add an assertion that each value (item[1] in the next line) is not iterable.</Alex>
    sample_data_items = [(item[0], [item[1]]) for item in sample_data_items]
    df_sample: pd.DataFrame = pd.DataFrame(data=dict(sample_data_items))

    return df_sample


def convert_pandas_dataframe_to_batch_form_dictionary(
    df_data: pd.DataFrame, num_samples: int = 1, random_state: int = DEFAULT_RANDOM_STATE
) -> dict:
    """Converts input DataFrame to dictionary user "split" format so that it can be submitted to "batch_predict"
    API.

    # Inputs

    :param :df_data (pd.DataFrame): input Pandas DataFrame (e.g., from test/evaluation dataset)
    :param :num_samples (int, optional): number of random samples from input DataDrame to return (detault is 1)
    :param :random_state (int, optional): Seed for random number generation (default is DEFAULT_RANDOM_STATE).

    # Return

    :return (dict): resulting dictionary in the JSON format that can be submitted to "batch_predict" HTTP API.
    """
    # TODO: <Alex>Additional precautions against very large batch sizes requested would be highly advisable..</Alex>
    if num_samples > df_data.shape[0]:
        num_samples = df_data.shape[0]

    df_data = df_data.sample(n=num_samples, random_state=random_state)
    data: dict = df_data.to_dict(orient="split")

    post_submission: dict = {
        "dataset": (None, json.dumps(data), "application/json"),
    }

    return post_submission


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
