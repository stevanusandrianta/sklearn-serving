# Scikit-Learn Serving

Deploying machine learning model supposed to be simple, this is why this project is started where I focus on one of the popular machine learning library which is [scikit-learn](![](https://github.com/scikit-learn/scikit-learn)https://github.com/scikit-learn/scikit-learn). The goal of this project is serving a scikit-learn model artifact to REST api on docker with single command.

## Pre-requisites

- Python 3.5+
- Docker is installed, how to install docker can be found [here](https://docs.docker.com/get-docker/)

## Setup

It is recommended if you just download the docker container remotely

```bash
docker pull sandrianta/sklearn-serving:latest
```

In order to build the docker image locally:

```bash
docker build -t sandrianta/sklearn-serving:latest .
```

## Usage

### 1. Using list type as an input

Building the model and run FastAPI locally

```
python3 examples/iris_logistic_regression.py

docker run -p 8000:80 \
--mount type=bind,source="$(pwd)"/,target=/tmp/model/ \
-e MODEL_PATH=/tmp/model/iris_logistic_regression.joblib \
sandrianta/sklearn-serving:latest
```

Example to invoke the API

```
curl --location --request POST 'localhost:8000/predict/' \
--header 'Content-Type: application/json' \
--data-raw '{"data" : [[1,2,3,10]]}'
```

### 2. Using json type as input

Building the model and run FastAPI locally

```
python3 examples/iris_logistic_regression.py

docker run -p 8000:80 \
--mount type=bind,source="$(pwd)"/,target=/tmp/model/ \
-e MODEL_PATH=/tmp/model/titanic_sklearn_pipelines.joblib \
sandrianta/sklearn-serving:latest
```

Example to invoke the API

```
curl --location --request POST 'localhost:8000/predict/' \
--header 'Content-Type: application/json' \
--data-raw '{"data" : [{"PassengerId":1,"Pclass":3,"Name":"Braund, Mr. Owen Harris","Sex":"male","Age":22.0,"SibSp":1,"Parch":0,"Ticket":"A\\/5 21171","Fare":7.25,"Cabin":null,"Embarked":"S"}]}'
```