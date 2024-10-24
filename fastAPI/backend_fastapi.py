from fastapi import FastAPI
from pydantic import BaseModel, create_model
import mlflow
import json
from typing import Any, Optional
from projet07.custom_transformers import ColumnTransformer
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd


app = FastAPI()
json_file = '../data/models_to_deploy.json'

# Create a model dynamicaly based on te kinds of inputs expected
input_information = pd.read_csv('../data/input_information.csv', index_col=0)
field_types = input_information['Dtype'].apply(
    lambda x : eval(f'({x} | dict[int, {x}], ...)')).to_dict()

print(field_types)

ModelEntries = create_model('ModelEntries', **field_types)

with open(json_file,'r') as file:
    models_to_deploy = json.load(file)

class ScoringModel(BaseModel):    
    model: Any
    validation_threshold: float

models: dict[str, ScoringModel] = {}
for model_info in models_to_deploy:
    key = model_info['model_name'] + '_v' + model_info['version']

    models[key] = ScoringModel(
        model=mlflow.pyfunc.load_model(model_uri=model_info['source'])._model_impl.sklearn_model,
        validation_threshold=model_info['validation_threshold'],
    )


@app.post("/validate_client/{model_name_v}")
async def validate_client(model_name_v: str, input_data: ModelEntries )->dict[str,float|bool|list[float|bool]]:
    '''

    '''
    input_data = input_data.dict()
    
    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys())
    input_df = input_df.replace({None:np.nan}).T
    
    y_pred_proba = models[model_name_v].model.predict_proba(input_df)[:, 1]        
    
    validation_threshold = models[model_name_v].validation_threshold

    respond = {'default_probability' : y_pred_proba,
                'validation_threshold' : validation_threshold,
                'credit_approved' : y_pred_proba < validation_threshold,
                }
    
    return respond




