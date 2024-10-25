# CreditScoringAPI

# How to use
After performing the data exploration and model developpment, trainning and exploration a json file should be produced and included in the root path of this project with name `models_to_deploy.json` that contains the list of dictionnaries that describe the versions of models whose structure should be comptible with mflow development as the one below:


```json
[
    {
        "model_name": "name_of_model",
        "version": "1",
        "source": "/path/to/mlflow/artifacts/373709576065196998/c8c30aa6b9924d45975c34bc44c55743/artifacts/lightgbm",
        "validation_threshold": "0.4866684226260628"
    }
]
```
> Please note the absolut path for the model

# Deploying API
On terminal go to directory `path/to/api_deployment/fastAPI/' and 
input:
```
uvicorn backend_fastapi:app --reload
```
or 

```
uvicorn backend_fastapi:app --reload
```
for automaticaly reloading API after modification.
