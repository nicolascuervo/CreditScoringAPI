# CreditScoringAPI

# How to use
After performing the data exploration and model developpment, trainning and exploration a json file should be produced and included in the root path of this project with name `models_to_deploy.json` that contains the list of dictionnaries that describe the versions of models whose structure should be comptible with mflow development as the one below:


```json
[
    {
        "aliases": [],
        "creation_timestamp": 1728301840427,
        "current_stage": "None",
        "description": "",
        "last_updated_timestamp": 1728301840427,
        "name": "lightgbm",
        "run_id": "cfda60adb0714c2e9d7a74a4e86dcf71",
        "run_link": "",
        "source": "/mnt/.../mlartifacts/962910245136909974/cfda60adb0714c2e9d7a74a4e86dcf71/artifacts/lightgbm",
        "status": "READY",
        "status_message": "",
        "tags": {},
        "user_id": "",
        "version": "2"
    }
]
```
> Please note the absolut path for the model

# MLFlow

see [MLflow Documentation](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html)

## Serving models
In terminal input:
```
python3 path/to/mlflow_api.py path/to/models_to_deploy.json port_number0
```
example:
```
python3 mlflow/mlflow_api.py models_to_deploy.json 5000
```
## Tetsing API

In terminal input:
```
python3 path/to/model_test.py path/to/models_to_deploy.json port_number0
```

example:
```
python3 mlflow/model_test.py models_to_deploy.json 5000
```
> If entries in `*source_path*/serving_input_example.json`  contain **`null`** values, replace them for **`NaN`** in order to test them

