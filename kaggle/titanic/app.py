from __future__ import annotations

import json

import pandas as pd
from flask import Flask, jsonify, request

from kaggle.titanic.model import (
    convert_dictionary_to_pandas_dataframe,
    convert_pandas_dataframe_to_batch_form_dictionary,
    TitanicModel,
)

app = Flask(__name__)


titanic_model = TitanicModel()


@app.route("/")
def site_root():
    return jsonify("Access Succeded")


@app.route("/config")
def get_config():
    return jsonify(titanic_model.config)


@app.route("/full-config")
def get_full_config():
    return jsonify(titanic_model.full_config)


@app.route("/input-features")
def get_input_features():
    return jsonify(titanic_model.input_features)


@app.route("/train")
def train():
    train_stats, _, _ = titanic_model.train(dataset=titanic_model.df_training_set)
    return jsonify(train_stats)


@app.route("/predict", methods=["POST"])
def predict():
    sample_data: dict = request.get_json()
    df_sample: pd.DataFrame = convert_dictionary_to_pandas_dataframe(sample_dictionary=sample_data)

    df_predictions_and_probabilities: pd.DataFrame
    model_in_memory: bool
    model_directory_path: str | None
    df_predictions_and_probabilities, model_in_memory, model_directory_path = titanic_model.predict(dataset=df_sample)

    result: dict = df_predictions_and_probabilities.to_dict()
    result["model_in_memory"] = model_in_memory
    result["model_load_path"] = model_directory_path

    return jsonify(result), 200


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    input_params: dict | None = request.get_json()

    num_samples: int | None = None
    if input_params:
        num_samples = input_params.get("num_samples")

    if not (num_samples and isinstance(num_samples, int)):
        num_samples = 1

    data_samples: dict = convert_pandas_dataframe_to_batch_form_dictionary(
        df_data=titanic_model.df_test_set, num_samples=num_samples
    )
    data_samples = json.loads(data_samples["dataset"][1])
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # data_samples = data_samples["dataset"][1]
    # # data_samples = data_samples["index"]
    # # data_samples = str(type(data_samples))
    # data_samples = json.loads(data_samples)
    # # TODO: <Alex>ALEX</Alex>
    # # data_samples = data_samples["columns"]
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    df_samples: pd.DataFrame = pd.DataFrame(**data_samples)
    # TODO: <Alex>ALEX</Alex>
    #
    df_predictions_and_probabilities: pd.DataFrame
    model_in_memory: bool
    model_directory_path: str | None
    df_predictions_and_probabilities, model_in_memory, model_directory_path = titanic_model.predict(dataset=df_samples)

    result: dict = df_predictions_and_probabilities.to_dict()
    result["model_in_memory"] = model_in_memory
    result["model_load_path"] = model_directory_path
    # TODO: <Alex>ALEX</Alex>

    # TODO: <Alex>ALEX</Alex>
    # result = data_samples
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    return jsonify(result), 200
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    # return jsonify(input_params), 200
    # TODO: <Alex>ALEX</Alex>
