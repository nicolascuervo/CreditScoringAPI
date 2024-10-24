# CreditScoringAPI

This project rwas developepd in an academic context based on the scenario presented at https://www.kaggle.com/c/home-credit-default-risk/.

An API for one or more models to score and aprove/deny credit requests fot one or more clients is presented.

A dashboard for interacting with one model for one client requests is also available

> both are running localy at this stage.

# How to use

## API
After performing the data exploration and model developpment, trainning and exploration a json file should be produced and included in the root/data folder of this project with name `models_to_deploy.json` that contains the list of dictionnaries that describe the versions of models whose structure should be comptible with mflow development as the one below:


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

On terminal go to directory `path/to/api_deployment/fastAPI/' and input:

```
uvicorn backend_fastapi:app
```
or 

```
uvicorn backend_fastapi:app --reload
```
## Dashboard

Include the following files  folder project-root/data

project-root/
│
└── data/
    ├── application_test.csv
    ├── application_train.csv
    └── input_information.csv

application_test/train.csv are available at https://www.kaggle.com/c/home-credit-default-risk/data.
input_information.csv os generated in parallel CreditScoring project and is provided here
    
# Deploying API
On terminal go to directory `path/to/api_deployment/fastAPI/' and input:

```
uvicorn backend_fastapi:app
```
or 

```
uvicorn backend_fastapi:app --reload
```
for automaticaly reloading API after modification.


# Dashboard

On terminal:
```
streamlit run path/to/dashboard.py
```
> the dashboard is ment to wrok with a limited MIN_SK_ID_CURR = 100001 and MAX_SK_ID_CURR = 456255. If date boyend this values is included, 