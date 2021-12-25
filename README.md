#Scikit-Learn Serving

Deploying machine learning model supposed to be simple, this is why this project is started where I focus on one of the popular machine learning library which is [scikit-learn](![](https://github.com/scikit-learn/scikit-learn)https://github.com/scikit-learn/scikit-learn). The goal of this project is serving a scikit-learn model artifact to REST api on docker with single command.

## Pre-requisites

- Python 3.5+
- Docker is installed, how to install docker can be found [here](https://docs.docker.com/get-docker/)

## Setup

In order to build the docker image locally:

```bash
docker build -t sklearn-serving:latest .
```

## Usage

```
python3 examples/iris_logistic_regression.py

docker run -p 8000:80 \
--mount type=bind,source="$(pwd)"/,target=/tmp/model/ \
-e MODEL_PATH=/tmp/model/iris_logistic_regression.joblib \
sklearn-serving:latest
```

After the FastAPI on docker is up and running, you can run this curl command to invoke the API

```
curl --location --request POST 'localhost:8000/predict/' \
--header 'Content-Type: application/json' \
--data-raw '{"data" : [[1,2,3,10]]}'
```