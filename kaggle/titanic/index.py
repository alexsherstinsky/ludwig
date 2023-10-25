from flask import Flask, jsonify, request

from kaggle.titanic.model import TitanicModel

app = Flask(__name__)


# TODO: <Alex>ALEX</Alex>
class Incomes:
    def __init__(self):
        self.incomes = [{"description": "salary", "amount": 5000}]


incomes = Incomes()
# TODO: <Alex>ALEX</Alex>

titanic_model = TitanicModel()


@app.route("/")
def site_root():
    return jsonify("Access Succeded")


# TODO: <Alex>ALEX</Alex>
@app.route("/reset_incomes", methods=["POST"])
def reset_incomes():
    incomes.incomes = []
    # TODO: <Alex>ALEX</Alex>
    # incomes.clear()
    # TODO: <Alex>ALEX</Alex>
    return "", 204


@app.route("/incomes")
def get_incomes():
    return jsonify(incomes.incomes)


@app.route("/incomes", methods=["POST"])
def add_income():
    incomes.incomes.append(request.get_json())
    return "", 204


# TODO: <Alex>ALEX</Alex>


@app.route("/config")
def get_config():
    return jsonify(titanic_model.config)


@app.route("/train")
def train():
    train_stats, _, _ = titanic_model.train(df_dataset=titanic_model.df_training_set)
    return jsonify(train_stats)
