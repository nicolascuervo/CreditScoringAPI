from fastapi import FastAPI
from pydantic import BaseModel, create_model
import mlflow
import json
from typing import Any
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import shap

app = FastAPI(
        title = 'Credit Scoring API',
        description= 'API for serving one or more models that retreiveve structured' \
             + 'information on credit applications and returns an score and a validation treshold ' \
             + 'over which the probability of default makes the approval of the credit unadvisable',
        version="0.1.0"
)
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

@app.get(f"/available_model_name_v")
def get_available_model_name_v()->list[str]:
    return list(models.keys())



for model_name_v in models.keys():    
    async def validate_client( input_data: ModelEntries )->dict[str,float|bool|list[float|bool]]:
        """
        Endpoint to validate a client's credit approval status based on a specific model.

        This endpoint uses a pre-trained model identified by '{model_name_v}' to predict
        the probability of default for a client based on the provided input data. If
        the predicted probability is below the model's validation threshold, the credit
        is approved.

        Parameters:
        -----------
        input_data : ModelEntries
            The client's data used for model prediction. This is expected to be an instance
            of `ModelEntries` and should contain the necessary input fields for the model.

        Returns:
        --------
        dict[str, float | bool | list[float | bool]]
            A dictionary with the following keys:
            
            - 'default_probability' (list[float]): Predicted probability(s) of default based on the input data.
            - 'validation_threshold' (float): The threshold used to decide if credit is approved.
            - 'credit_approved' (list[bool]): Approval decision(s) for each input record, where True
              indicates credit approval (i.e., predicted probability is below the threshold).

        Usage:
        ------
        Make a POST request to this endpoint with client data, such as:

            POST {model_name_v}/validate_client/

        The endpoint will return a JSON response indicating the client's default probability, the
        threshold used, and whether credit is approved.

        Notes:
        ------
        - Each model is served under a different endpoint, and the model is dynamically
          selected using the `model_name_v` in the endpoint path.
        - Missing values in `input_data` are replaced with NaN for prediction.
        """
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
    # replace model_name_v in documentation
    validate_client.__doc__ = validate_client.__doc__.format(model_name_v=model_name_v)
    #post function
    app.post(f"/{model_name_v}/validate_client")(validate_client)

    @app.post(f"/{model_name_v}/shap_values")
    def shap_values(data: ModelEntries):
        # X_treated = Pipeline(lgbm_model.steps[0:-1]).transform(X)
        # lgbm_model.steps[-1][1].predict_proba(X_treated)
        model = models[model_name_v]
        
        explainer = shap.explainers.TreeExplainer(model, X_pretreated)
        shap_values = explainer(X_pretreated.iloc[1528:1529,:])

        shap.plots.waterfall(shap_values[-1])
        shap_values.base_values