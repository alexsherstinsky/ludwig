import argparse
import base64
import hashlib
import json
import os
import pickle
import random
import shutil
import tempfile
from multiprocessing import Process

import numpy as np

from ludwig.api import LudwigModel
from ludwig.datasets import titanic
from ludwig.serve import run_server
from ludwig.utils.data_utils import save_json

TRAIN_PREPROCESSED = "train_ds.pkl"
TEST_PREPROCESSED = "test_ds.pkl"
METADATA = "metadata.json"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def random_numerical_preprocessing():
    return {
        "missing_value_strategy": "fill_with_mean",
        "normalization": "zscore" if random.random() < 0.5 else "minmax",
    }


def random_lr():
    return np.random.exponential(10) / 100


def generate_random_config():
    config = {
        "input_features": [
            {"name": "Pclass", "type": "category"},
            {"name": "Sex", "type": "category"},
            {"name": "Age", "type": "number", "preprocessing": random_numerical_preprocessing()},
            {"name": "SibSp", "type": "number", "preprocessing": random_numerical_preprocessing()},
            {"name": "Parch", "type": "number", "preprocessing": random_numerical_preprocessing()},
            {"name": "Fare", "type": "number", "preprocessing": random_numerical_preprocessing()},
            {"name": "Embarked", "type": "category"},
        ],
        "output_features": [
            {"name": "Survived", "type": "binary"},
        ],
        "preprocessing": {"split": {"type": "random", "probabilities": [0.8, 0.0, 0.2]}},
        "trainer": {
            "learning_rate": random_lr(),
            "epochs": 5,
        },
        "backend": {"type": "local"},
    }
    print(f"{json.dumps(config)}")
    return config


def hash_config(config):
    d = config["input_features"]
    s = json.dumps(d, sort_keys=True, ensure_ascii=True)
    h = hashlib.md5(s.encode())
    d = h.digest()
    hash_code = base64.b64encode(d, altchars=b"__").decode("ascii")
    print(f"{hash_code}")
    return hash_code


def preprocess(config, output_dir):
    train_data, _, _ = titanic.load(split=True)
    model = LudwigModel(config)
    train_ds, _, test_ds, training_set_metadata = model.preprocess(
        dataset=train_data,
    )
    print(len(train_ds))
    print(len(test_ds))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, TRAIN_PREPROCESSED), "wb") as f:
        pickle.dump(train_ds, f)
    with open(os.path.join(output_dir, TEST_PREPROCESSED), "wb") as f:
        pickle.dump(test_ds, f)
    save_json(os.path.join(output_dir, METADATA), training_set_metadata)


def train(config, processed_dir, output_dir):
    with open(os.path.join(processed_dir, TRAIN_PREPROCESSED), "rb") as f:
        train_ds = pickle.load(f)
    with open(os.path.join(processed_dir, METADATA)) as f:
        training_set_metadata = json.load(f)

    model = LudwigModel(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_stats, preprocessed_data, output_directory = model.train(
            training_set=train_ds,
            training_set_metadata=training_set_metadata,
            output_directory=tmpdir,
        )
        shutil.copytree(output_directory, output_dir)
    metric = min(train_stats["training"]["combined"]["loss"])
    print(f"{metric}")
    return float(metric)


def evaluate(processed_dir, model_dir):
    with open(os.path.join(processed_dir, TEST_PREPROCESSED), "rb") as f:
        test_ds = pickle.load(f)

    model = LudwigModel.load(os.path.join(model_dir, "model"))
    eval_stats, _, _ = model.evaluate(dataset=test_ds)
    metric = eval_stats["combined"]["loss"]
    print(f"{metric}")
    return float(metric)


def serve(model_dir, port):
    run_server(os.path.join(model_dir, "model"), host="0.0.0.0", port=port, allowed_origins=["*"])


def start_serve(model_dir, port):
    process = Process(target=serve, args=(model_dir, port))
    process.start()
    return process


def stop_serve(process):
    process.kill()
    process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd", choices=["generate_random_config", "preprocess", "train", "evaluate", "serve", "hash_config"]
    )
    parser.add_argument("--config")
    parser.add_argument("--output_dir")
    parser.add_argument("--processed_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    if args.config:
        args.config = json.loads(args.config)

    if args.cmd == "generate_random_config":
        generate_random_config()
    elif args.cmd == "hash_config":
        hash_config(args.config)
    elif args.cmd == "preprocess":
        preprocess(args.config, args.output_dir)
    elif args.cmd == "train":
        train(args.config, args.processed_dir, args.output_dir)
    elif args.cmd == "evaluate":
        evaluate(args.processed_dir, args.model_dir)
    elif args.cmd == "serve":
        serve(args.model_dir, args.port)
