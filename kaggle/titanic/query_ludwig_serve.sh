#!/bin/sh


echo "running single predict"
curl http://0.0.0.0:8000/predict -X POST -F 'Pclass=3' -F 'Sex=male' -F 'Age=34.5' -F 'SibSp=0' -F 'Parch=0' -F 'Fare=7.8292' -F 'Embarked=Q'
