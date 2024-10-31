# CreditScoringAPI

This project was developepd in an academic context based on the scenario presented at https://www.kaggle.com/c/home-credit-default-risk/.

An API for one or more models to score and aprove/deny credit requests for one or more clients is presented.

> It runs localy at this stage.

# How to use


## locally 
the following environment variables are to be defined in a file '.env' at root:

```
SHAP_SAMPLE_SIZE=1000
MODELS_TO_DEPLOY_JSON=path_to/models_to_deploy.json
APPLICATION_TRAIN_CSV=path_to/application_train.csv
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


## Cloud


the following environment variables are to be defined:

``plain_text
SHAP_SAMPLE_SIZE=1000
MODELS_TO_DEPLOY_JSON=htttp/cloud_service.com/download_path?id=file_id1
APPLICATION_TRAIN_CSV=htttp/cloud_service.com/download_path?id=file_id2
```


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


The file `application_train.csv` from the [kaggle data](https://www.kaggle.com/c/home-credit-default-risk/data) should also be present this folder as follows:

```
project-root/
│
└── data/
    ├── model_to_deploy.json
    ├── application_train.csv
    └── input_information.csv
```
# Deployment

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

