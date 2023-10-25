#!/bin/sh


# verify access
echo "root page"
curl http://localhost:5000/

# get Titanic config
echo "Titanic config"
curl localhost:5000/config

# train Titanic model
echo "Titanic training statistics"
curl localhost:5000/train
