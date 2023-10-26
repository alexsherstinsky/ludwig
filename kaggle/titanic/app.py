from __future__ import annotations

import json
import threading
import uuid

import pandas as pd
from flask import Flask, jsonify, request

from kaggle.titanic.model import (
    convert_dictionary_to_pandas_dataframe,
    convert_pandas_dataframe_to_batch_form_dictionary,
    TitanicModel,
)

app = Flask(__name__)


# TODO: <Alex>ALEX</Alex>
import os

from kaggle.titanic.train_utils import evaluate, generate_random_config, preprocess, start_serve, stop_serve, train
from kaggle.titanic.training_handler import TrainingHandler
from ludwig.datasets import titanic

# TODO: <Alex>ALEX</Alex>


# TODO: <Alex>ALEX</Alex>
# titanic_model = TitanicModel()
# TODO: <Alex>ALEX</Alex>

# tmpdir = "/Users/alexsherstinsky/Development/ExternalGitRepositories/OpenSource/Ludwig/ludwig/kaggle/titanic"
# training_handler = TrainingHandler(tmpdir=tmpdir)
training_handler = TrainingHandler()


@app.route("/")
def site_root():
    return jsonify("Access Succeded")


# TODO: <Alex>ALEX</Alex>
# @app.route("/config")
# def get_config():
#     return jsonify(titanic_model.config)
#
#
# @app.route("/full-config")
# def get_full_config():
#     return jsonify(titanic_model.full_config)
#
#
# @app.route("/input-features")
# def get_input_features():
#     return jsonify(titanic_model.input_features)
#
#
# @app.route("/train")
# def train():
#     train_stats, _, _ = titanic_model.train(dataset=titanic_model.df_training_set)
#     return jsonify(train_stats)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     sample_data: dict = request.get_json()
#     df_sample: pd.DataFrame = convert_dictionary_to_pandas_dataframe(sample_dictionary=sample_data)
#
#     result: dict = _run_model_predictions_on_sample_pandas_dataframe(dataset=df_sample)
#
#     return jsonify(result), 200
#
#
# @app.route("/batch-predict", methods=["POST"])
# def batch_predict():
#     input_params: dict | None = request.get_json()
#
#     num_samples: int | None = None
#     if input_params:
#         num_samples = input_params.get("num_samples")
#
#     if not (num_samples and isinstance(num_samples, int)):
#         num_samples = 1
#
#     data_samples: dict = convert_pandas_dataframe_to_batch_form_dictionary(
#         df_data=titanic_model.df_test_set, num_samples=num_samples
#     )
#     data_samples = json.loads(data_samples["dataset"][1])
#     df_samples: pd.DataFrame = pd.DataFrame(**data_samples)
#
#     result: dict = _run_model_predictions_on_sample_pandas_dataframe(dataset=df_samples)
#
#     return jsonify(result), 200
#
#
# def _run_model_predictions_on_sample_pandas_dataframe(dataset: pd.DataFrame) -> dict:
#     df_predictions_and_probabilities: pd.DataFrame
#     model_in_memory: bool
#     model_directory_path: str | None
#     df_predictions_and_probabilities, model_in_memory, model_directory_path = titanic_model.predict(dataset=dataset)
#
#     result: dict = df_predictions_and_probabilities.to_dict()
#     result["model_in_memory"] = model_in_memory
#     result["model_load_path"] = model_directory_path
#
#     return result
# TODO: <Alex>ALEX</Alex>


@app.route("/interview-train", methods=["POST"])
def interview_train():
    input_params: dict | None = request.get_json()

    num_train_runs: int | None = None
    if input_params:
        num_train_runs = input_params.get("n")

    tmpdir = "/Users/alexsherstinsky/Development/ExternalGitRepositories/OpenSource/Ludwig/ludwig/kaggle/titanic"

    model_ids = []
    for _ in range(num_train_runs):
        model_id = generete_unique_model_id()
        model_ids.append(model_id)

    queue = training_handler.queue
    for model_id in model_ids:
        task_info = (model_id, tmpdir)
        queue.put(task_info)

    # training_threads = []
    # for model_id in model_ids:
    #     training_task_thread = ModelTrainingThreadedTask(model_id, tmpdir)
    #     app.logger.error(f'Created Thread object for model_id={model_id}')
    #     training_task_thread.start()
    #     training_threads.append(training_task_thread)

    # TODO: <Alex>ALEX</Alex>
    # dir_contents_from_n_runs = []
    # model_ids = []
    # for idx in range(num_train_runs):
    #     processed_dir = os.path.join(tmpdir, f"processed_{idx}")
    #     model_dir = os.path.join(tmpdir, f"model_{idx}")
    #     config = generate_random_config()
    #     preprocess(config, processed_dir)
    #     print(f'listing contents of {processed_dir}: {os.listdir(processed_dir)}')
    #
    #     train_loss = train(config, processed_dir, model_dir)
    #     eval_loss = evaluate(processed_dir, model_dir)
    #
    #     dir_contents = os.listdir(model_dir)
    #     dir_contents_from_n_runs.append(dir_contents)
    #
    #     model_id = generete_unique_model_id()
    #     model_ids.append(model_id)
    #
    #     # print(f'listing contents of {model_dir}: {dir_contents}')
    # TODO: <Alex>ALEX</Alex>

    # return jsonify(dir_contents_from_n_runs), 200
    return jsonify(model_ids), 200


def generete_unique_model_id(
    default_prefix: str = "model_id_",
    num_digits: int = 8,
) -> str:
    return f"{default_prefix}{str(uuid.uuid4())[:num_digits]}"


# class ModelTrainingThreadedTask(threading.Thread):
#     @staticmethod
#     def config_and_train_seqence(model_id, tmpdir):
#         app.logger.info(f'Beginning ModelTrainingThreadedTask.config_and_train_seqence() for model_id={model_id}')
#         processed_dir = os.path.join(tmpdir, f"processed_{model_id}")
#         model_dir = os.path.join(tmpdir, f"model_{model_id}")
#         config = generate_random_config()
#         preprocess(config, processed_dir)
#         print(f'listing contents of {processed_dir}: {os.listdir(processed_dir)}')
#
#         train_loss = train(config, processed_dir, model_dir)
#         eval_loss = evaluate(processed_dir, model_dir)
#
#         app.logger.info(f'Finished ModelTrainingThreadedTask.config_and_train_seqence() for model_id={model_id}')
#
#     def __init__(self, model_id, tmpdir, target=config_and_train_seqence):
#         super(ModelTrainingThreadedTask, self).__init__(target=target, args=(model_id, tmpdir,))
