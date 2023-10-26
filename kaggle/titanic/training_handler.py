from __future__ import annotations

import json
import os
import threading
import uuid
from queue import Queue

import pandas as pd
from flask import Flask, jsonify, request

from kaggle.titanic.model import (
    convert_dictionary_to_pandas_dataframe,
    convert_pandas_dataframe_to_batch_form_dictionary,
    TitanicModel,
)
from kaggle.titanic.train_utils import evaluate, generate_random_config, preprocess, start_serve, stop_serve, train
from ludwig.datasets import titanic


class ModelTrainingThreadedTask(threading.Thread):
    @staticmethod
    def config_and_train_seqence(queue):
        while True:
            # app.logger.info(f'Beginning ModelTrainingThreadedTask.config_and_train_seqence() for model_id={model_id}')
            task_arguments = queue.get()
            print(
                f"\n[ALEX_TEST] [ModelTrainingThreadedTask:config_and_train_seqence()] STARTING_WORK-THREAD_RUN_ARGUMENTS:\n{task_arguments} ; TYPE: {str(type(task_arguments))}"
            )
            model_id = task_arguments[0]
            tmpdir = task_arguments[1]
            processed_dir = os.path.join(tmpdir, f"processed_{model_id}")
            model_dir = os.path.join(tmpdir, f"model_{model_id}")
            config = generate_random_config()
            preprocess(config, processed_dir)
            print(f"listing contents of {processed_dir}: {os.listdir(processed_dir)}")

            train_loss = train(config, processed_dir, model_dir)
            eval_loss = evaluate(processed_dir, model_dir)

            # app.logger.info(f'Finished ModelTrainingThreadedTask.config_and_train_seqence() for model_id={model_id}')
            print(
                f"\n[ALEX_TEST] [ModelTrainingThreadedTask:config_and_train_seqence()] FINISHED_THREAD_RUN_ARGUMENTS:\n{task_arguments} ; TYPE: {str(type(task_arguments))}"
            )

    def __init__(self, queue, target=config_and_train_seqence):
        super().__init__(target=target, args=(queue,))


class TrainingHandler:
    def __init__(self):
        self.queue = Queue()

        training_threads = []
        for _ in range(4):
            training_task_thread = ModelTrainingThreadedTask(self.queue)
            training_task_thread.start()
            training_threads.append(training_task_thread)
            print(
                f"\n[ALEX_TEST] [TrainingHandler:__init__()] STARTED-THREAD:\n{training_task_thread.name} ; TYPE: {str(type(training_task_thread.name))}"
            )
