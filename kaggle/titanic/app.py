from __future__ import annotations

# TODO: <Alex>ALEX</Alex>
# from flask import Flask, jsonify, request
# TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>
from flask import Flask, jsonify

from kaggle.titanic.model import TitanicModel

# TODO: <Alex>ALEX</Alex>


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
    train_stats, _, _ = titanic_model.train(df_dataset=titanic_model.df_training_set)
    return jsonify(train_stats)
