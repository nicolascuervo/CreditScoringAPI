from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import json
from typing import Any
from projet07.custom_transformers import ColumnTransformer
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd

app = FastAPI()
json_file = '../models_to_deploy.json'
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



for model_name_v in models.keys():
    @app.post("/validate_client/{model_name_v}")
    async def validate_client(input_data: dict)->dict[str,float|bool]:
        print(model_name_v)
        input_data_nan = {key:np.nan for key, value in input_data.items() if value=='' }
        input_data = input_data.copy()
        input_data.update(input_data_nan)
        
        input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys()).T
        y_pred_proba = models[model_name_v].model.predict_proba(input_df)[0, 1]
        
        validation_threshold = models[model_name_v].validation_threshold

        respond = {'default_probability' : y_pred_proba,
                   'validation_threshold' : validation_threshold,
                   'credit_approved' : y_pred_proba < validation_threshold,
                  }
        
        return respond




# # FastAPI handles JSON serialization and deserialization for us.
# # We can simply use built-in python and Pydantic types, in this case dict[int, Item].
# @app.get("/")
# def index() -> dict[str, dict[int, Item]]:
#     return {"items": items}
