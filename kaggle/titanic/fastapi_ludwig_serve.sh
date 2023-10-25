#!/bin/sh
conda run -n ludwig ludwig serve --model_path=$1
