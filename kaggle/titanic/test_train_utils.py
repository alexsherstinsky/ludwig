import os
import tempfile
import time

import pandas as pd
import requests
from train_utils import evaluate, generate_random_config, preprocess, start_serve, stop_serve, train

from ludwig.datasets import titanic

with tempfile.TemporaryDirectory() as tmpdir:
    processed_dir = os.path.join(tmpdir, "processed")
    model_dir = os.path.join(tmpdir, "model")

    config = generate_random_config()
    preprocess(config, processed_dir)
    print(f"listing contents of {processed_dir}: {os.listdir(processed_dir)}")

    train_loss = train(config, processed_dir, model_dir)
    eval_loss = evaluate(processed_dir, model_dir)
    print(f"listing contents of {model_dir}: {os.listdir(model_dir)}")

    # Example request:
    #
    #     test_df = titanic.load().head(5)
    #     response = requests.post(
    #        'http://localhost:8000/batch_predict',
    #        data={'dataset': test_df.to_json(orient='split')}
    #     )
    #

    p = start_serve(model_dir, 8000)
    print(f"started serve process: {p.pid}")

    stop_serve(p)
    print(f"stopped serve process: {p.pid}")
