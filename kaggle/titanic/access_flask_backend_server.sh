#!/bin/sh


# verify access
echo "root page"
curl http://localhost:5000/


# get Titanic Ludwig config
echo "Titanic config"
curl localhost:5000/config

# get Titanic full Ludwig config
echo "Titanic full config"
curl localhost:5000/full-config

# get Titanic input features
echo "Titanic input features"
curl localhost:5000/input-features

# TODO: <Alex>ALEX</Alex>
## train Titanic model
#echo "Titanic training statistics"
#curl localhost:5000/train
# TODO: <Alex>ALEX</Alex>


# run a single prediction on Titanic model
echo "Titanic inference results"
curl -X POST -H "Content-Type: application/json" -d '{"Pclass": 3, "Sex": "male", "Age": 34.5, "SibSp": 0, "Parch": 0, "Fare": 7.8292, "Embarked": "Q"}' http://localhost:5000/predict
