#!/bin/sh
export FLASK_APP=./index.py
conda run -n ludwig flask --debug run -h 0.0.0.0
