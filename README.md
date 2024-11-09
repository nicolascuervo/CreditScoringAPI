# CreditScoringAPI

This project was developepd in an academic context based on the scenario presented at https://www.kaggle.com/c/home-credit-default-risk/.

An API for one or more models to score and aprove/deny credit requests for one or more clients is presented.

# How to use

The files `application_train.csv` and `application_test.csv` from the [kaggle data](https://www.kaggle.com/c/home-credit-default-risk/data) should be converted to an sqlite database as done in notebook from parallel project [CreditScoring](https://github.com/nicolascuervo/CreditScoring). They must be loaded in two separate tables application_test and application_train. The url or path to the database should be placed in environment variable `CREDIT_REQUESTS_DB`

## locally 
the following environment variables are to be defined in a file '.env' at root:

```
SHAP_SAMPLE_SIZE=1000
MODELS_TO_DEPLOY_JSON=path_to/models_to_deploy.json
CREDIT_REQUESTS_DB=path_to/credit_requests.db
```

After performing the data exploration and model developpment, trainning and exploration a json file should be produced and included in the root/data folder of this project with name `models_to_deploy.json` that contains the list of dictionnaries that describe the versions of models whose structure should be comptible with mflow development as the one below:

```json
[
    {
        "model_name": "name_of_model",
        "version": "1",
        "source": "/path/to/mlflow/artifacts/experiment_id/run_id/artifacts/lightgbm",
        "validation_threshold": "0.4866684226260628"
    }
]
```
> Please note the absolut path for the model 

### Deployment

On terminal go to directory `path/to/api_deployment/fastAPI/' and input:

```
uvicorn backend_fastapi:app
```
or 

```
uvicorn backend_fastapi:app --reload
```
for automaticaly reloading API after modification.


Documentation for the API can be consulted on
```url
url_api/docs
```
When the API is running 

## Cloud




The model can be deployed on the cloud as a zip file :
```
zip_model.zip
|-- MLmodel
|-- conda.yaml
|-- folder_structure.txt
|-- input_example.json
|-- model.pkl
|-- python_env.yaml
|-- registered_model_meta
|-- requirements.txt
`-- serving_input_example.json
```
And the json file can simply have a sharing file directing to the download url for the zipped model.
>

```json
[
    {
        "model_name": "name_of_model",
        "version": "1",
        "source": "htttp/cloud_service.com/download_path?id=file_id3",
        "validation_threshold": "0.4866684226260628"
    }
]
```

### Heroku Deployment

Update requirements.txt with
 ```
poetry export -f requirements.txt --output requirements.txt --without-hashes
```
and push the requirements.txt file

Also push with file `Procfile` contaning this line for the API to deploy

```
web: uvicorn fastAPI.backend_fastapi:app --host=0.0.0.0 --port=${PORT}
```

the following environment variables are to be defined:

```
SHAP_SAMPLE_SIZE=1000
MODELS_TO_DEPLOY_JSON=htttp/cloud_service.com/download_path?id=file_id1
CREDIT_REQUESTS_DB=htttp/cloud_service.com/download_path?id=file_id2
```


When the API is running swagger for the API can be consulted on
```url
url_api/docs
```


