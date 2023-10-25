from __future__ import annotations

import pandas as pd
from flask import Flask, jsonify, request

from kaggle.titanic.model import convert_dictionary_to_pandas_dataframe, TitanicModel

app = Flask(__name__)


titanic_model = TitanicModel()


@app.route("/")
def site_root():
    return jsonify("Access Succeded")


@app.route("/config")
def get_config():
    return jsonify(titanic_model.config)


@app.route("/train")
def train():
    train_stats, _, _ = titanic_model.train(dataset=titanic_model.df_training_set)
    return jsonify(train_stats)


@app.route("/predict", methods=["POST"])
def predict():
    sample_data: dict = request.get_json()
    df_sample: pd.DataFrame = convert_dictionary_to_pandas_dataframe(sample_dictionary=sample_data)
    df_predictions_and_probabilities: pd.DataFrame = titanic_model.predict(dataset=df_sample)
    result: dict = df_predictions_and_probabilities.to_dict()
    return jsonify(result), 200
