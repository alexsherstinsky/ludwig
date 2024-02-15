#!/usr/bin/env python

# # Simple Model Experiment Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

from __future__ import annotations

# Import required libraries
# TODO: <Alex>ALEX</Alex>
# import sys
# TODO: <Alex>ALEX</Alex>
import logging
import os
import shutil

import yaml

import pandas as pd

from ludwig.api import LudwigModel, PreprocessedDataset, TrainingStats
from ludwig.datasets import titanic

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
"""
The Titanic dataset, available on Kaggle, comes with "training_set" and "test_set", but does not have "val_set".  During
downloading and processing, "training_set", "test_set", and "val_set" are concatenated and a fixed "split" column is set
(0 -- training, 1 -- validation, 2 -- test).  Applying "DatasetLoader.load(split=True)" with "DatasetConfig" for Titanic
from Kaggle, splits the dataframe into three parts by the split value in the downloaded and processed Titanic data, and
returns three Pandas dataframes, corresponding to each of "training_set", "test_set", and "val_set" (empty for Titanic),
but without the "split" column, since it gets dropped by "DatasetLoader.split(split=True)".  Then "test_set" is held off
entirely for final evaluation, while "training_set" is used for model training with default split of [0.7, 0.1, 0.2] for
["training_set", "val_set", "test_set"].  There is apparent inefficiency in creating and removing the "split" column.
"""
training_set: pd.DataFrame
test_set: pd.DataFrame
val_set: pd.DataFrame
training_set, test_set, val_set = titanic.load(split=True)
"""
Resulting Pandas DataFrame shapes are:
* Entire Kaggle Titanic dataset: (1309, 13); extra column in original dataset is "split", which gets dropped upon load.
  - Titanic training_set: (891, 12); gets split into ["training_set"=0.7, "val_set"=0.1, "test_set"=0.2] during training
  - Titanic test_set: (418, 12); used for evaluation (post training phase)
  - Titanic val_set: (0, 12); discarded/ignored

* Dataset Dictionary
PassengerId   Passenger ID (integer)
Survived      Survival: 0 = No, 1 = Yes (float, due to NaN values in some rows; will be converted to binary)
Pclass        Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd (integer)
Name          Name of passenger (string)
Sex           Gender of passenger (string)
Age           Age: in years (float)
SibSp         Siblings/Spouses: # of siblings / spouses aboard the Titanic (integer)
Parch         Parents/Children: # of parents / children aboard the Titanic (integer)
Ticket        Ticket number (alphanumeric)
Fare          Passenger fare (float)
Cabin         Cabin number (alphanumeric)
Embarked      Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton (single-character string)
"""
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] TRAINING_SET:\n{training_set} ; TYPE: {str(type(training_set))} ; SHAPE: {training_set.shape}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] TEST_SET:\n{test_set} ; TYPE: {str(type(test_set))} ; SHAPE: {test_set.shape}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] VAL_SET:\n{val_set} ; TYPE: {str(type(val_set))} ; SHAPE: {val_set.shape}')

"""
Use 7 out 11 available columns ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", and "Embarked") as "input_features".
Columns "PassengerId", "Name", "Ticket", and "Cabin" are not used (as they would not make good features as predictors).
"""
config: dict = yaml.safe_load(
    """
input_features:
    - name: Pclass
      type: category
    - name: Sex
      type: category
    - name: Age
      type: number
      preprocessing:
          missing_value_strategy: fill_with_mean
    - name: SibSp
      type: number
    - name: Parch
      type: number
    - name: Fare
      type: number
      preprocessing:
          missing_value_strategy: fill_with_mean
    - name: Embarked
      type: category

output_features:
    - name: Survived
      type: binary

"""
)
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] CONFIG:\n{config} ; TYPE: {str(type(config))}')

# Define Ludwig model object that drive model training
model: LudwigModel = LudwigModel(config=config, logging_level=logging.INFO)
# TODO: <Alex>ALEX</Alex>
# full_config: dict = model.config
# print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] FULL_CONFIG:\n{full_config} ; TYPE: {str(type(full_config))}')
# TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>
# sys.exit()
# TODO: <Alex>ALEX</Alex>

# # TODO: <Alex>ALEX</Alex>
# # initiate model training
# train_stats: TrainingStats  # object containing training statistics (exhibits dict-like and tuple-like behavior)
# preprocessed_data: PreprocessedDataset  # collection of Ludwig Dataset objects of pre-processed training data
# output_directory: str  # location of training results stored on disk
# (
#     train_stats,
#     preprocessed_data,
#     output_directory,
# ) = model.train(
#     dataset=training_set,
#     experiment_name="simple_experiment",
#     model_name="simple_model",
#     skip_save_processed_input=True,
#     # TODO: <Alex>ALEX</Alex>
#     # logging_level=logging.INFO,
#     # TODO: <Alex>ALEX</Alex>
# )
# # TODO: <Alex>ALEX</Alex>
# # breakpoint()
# # TODO: <Alex>ALEX</Alex>
# # print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] TRAIN_STATS:\n{train_stats} ; TYPE: {str(type(train_stats))}')
# # print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] PREPROCESSED_DATA:\n{preprocessed_data} ; TYPE: {str(type(preprocessed_data))}')
# # print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] OUTPUT_DIRECTORY:\n{output_directory} ; TYPE: {str(type(output_directory))}')
# # TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>
# initiate experiment
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] INITIATING-LudwigModel.experiment()-SIMPLE_EXPERIMENT')
eval_stats: dict | None  # dictionary with evaluation performance statistics on the test_set (per output_feature)
train_stats: TrainingStats  # object containing training statistics (exhibits dict-like and tuple-like behavior)
preprocessed_data: PreprocessedDataset  # collection of Ludwig Dataset objects of pre-processed training data/metadata
output_directory: str  # location of training results stored on disk
experiment_output: tuple[dict | None, TrainingStats, PreprocessedDataset, str] = model.experiment(
    dataset=training_set,
    experiment_name="simple_experiment",
    model_name="simple_model",
    skip_save_processed_input=True,
    # TODO: <Alex>ALEX</Alex>
    # logging_level=logging.INFO,
    # TODO: <Alex>ALEX</Alex>
)
# TODO: <Alex>ALEX</Alex>
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] EXPERIMENT_OUTPUT:\n{experiment_output} ; TYPE: {str(type(experiment_output))}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] EXPERIMENT_OUTPUT_SIZE:\n{len(experiment_output)} ; TYPE: {str(type(len(experiment_output)))}')
# TODO: <Alex>ALEX</Alex>
for idx, exp_out in enumerate(experiment_output):
  print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] EXPERIMENT_OUTPUT[{idx}]:\n{exp_out} ; TYPE: {str(type(exp_out))}')
# TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>
eval_stats, train_stats, preprocessed_data, output_directory = experiment_output
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] EVAL_STATS:\n{eval_stats} ; TYPE: {str(type(eval_stats))}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] TRAIN_STATS:\n{train_stats} ; TYPE: {str(type(train_stats))}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] PREPROCESSED_DATA:\n{preprocessed_data} ; TYPE: {str(type(preprocessed_data))}')
print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] OUTPUT_DIRECTORY:\n{output_directory} ; TYPE: {str(type(output_directory))}')
# TODO: <Alex>ALEX</Alex>

# TODO: <Alex>ALEX</Alex>
# # TODO: <Alex>ALEX</Alex>
# # list contents of output directory
# # print("contents of output directory:", output_directory)
# # TODO: <Alex>ALEX</Alex>
# for item in os.listdir(output_directory):
#     # TODO: <Alex>ALEX</Alex>
#     # print("\t", item)
#     # TODO: <Alex>ALEX</Alex>
#     # TODO: <Alex>ALEX</Alex>
#     print(f'\n[ALEX_TEST] [TITANIC_EXPERIMENT] ITEM:\n{item} ; TYPE: {str(type(item))}')
#     # TODO: <Alex>ALEX</Alex>

# # batch prediction
# model.predict(dataset=test_set, skip_save_predictions=False)
# TODO: <Alex>ALEX</Alex>
