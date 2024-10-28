from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model, model_validator
import mlflow
import json
from typing import Any
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from inspect import signature
import shap

app = FastAPI(
        title = 'Credit Scoring API',
        description= 'API for serving one or more models that retreiveve structured' \
             + 'information on credit applications and returns an score and a validation treshold ' \
             + 'over which the probability of default makes the approval of the credit unadvisable',
        version="0.1.0"
)
json_file = '../data/models_to_deploy.json'

X_train = pd.read_csv('../data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])

# Create a model dynamicaly based on te kinds of inputs expected
input_information = pd.read_csv('../data/input_information.csv', index_col=0)
field_types = input_information['Dtype'].apply(
    lambda x : eval(f'({x} | dict[int, {x}], ...)')).to_dict()

ModelEntries = create_model('ModelEntries', **field_types)

with open(json_file,'r') as file:
    models_to_deploy = json.load(file)

class ScoringModel(BaseModel):    
    model: Any    
    validation_threshold: float
    X_train_treated: Any = Field(init=False)

    class Config:
        arbitrary_types_allowed = True  # Allows non-primitive types like sklearn 
        
    @model_validator(mode="before")
    def set_X_train_treated(cls, values):
        model = values.get('model')
        
        if hasattr(model, 'step'):
            steps = model.steps[:-1]
            values['X_train_treated'] = Pipeline(steps).transform(X_train)
        else:
            values['X_train_treated'] = None  # Handle case when steps are missing
        
        return values

models: dict[str, ScoringModel] = {}
for model_info in models_to_deploy:
    
    key = model_info['model_name'] + '_v' + model_info['version']
    print(f'INFO:', 'Loading model {key}...', end='\t')
    model = mlflow.pyfunc.load_model(model_uri=model_info['source'])._model_impl.sklearn_model
    models[key] = ScoringModel(
        model=model,
        validation_threshold=model_info['validation_threshold'],
    )
    print('Done')

@app.get(f"/available_model_name_version")
def get_available_model_name_version()->list[str]:
    """ Provides a list of `model_name_version` strings that indicate which versions of which models are abeing served by the API
    """
    return list(models.keys())

def credit_request_dict_to_dataframe(input_data: ModelEntries):
    input_data = input_data.dict()
    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys())
    input_df = input_df.replace({None:np.nan}).T
    return input_df

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
        input_df = credit_request_dict_to_dataframe(input_data)

        y_pred_proba = models[model_name_v].model.predict_proba(input_df)[:, 1]
        validation_threshold = models[model_name_v].validation_threshold
        print(y_pred_proba, validation_threshold)
        respond = {'default_probability' : y_pred_proba,
                    'validation_threshold' : validation_threshold,
                    'credit_approved' : y_pred_proba < validation_threshold,
                    }
        
        return respond
    
    # replace model_name_v in documentation
    validate_client.__doc__ = validate_client.__doc__.format(model_name_v=model_name_v)
    #post function
    app.post(f"/{model_name_v}/validate_client")(validate_client)

    @app.post(f"/{model_name_v}/shap_value_attributes")
    def shap_value_attributes(input_data: ModelEntries)->list[dict[str,Any]]:     
        """
        Computes SHAP values for the provided input data and extracts key attributes of the explanation for interpretation.

        This function initializes a SHAP explainer for {model_name_v}, preprocesses the input data, computes SHAP values, 
        and then extracts attributes from each SHAP explanation to return a list structured explanations to be reconstituted elsewhere.

        Parameters:
        -----------
        input_data : ModelEntries
            The input data for which SHAP values are to be computed, structured as a dictionary of model features and values.

        Returns:
        --------
        list[dict[str, Any]]
            A list of dictionaries, each containing attributes of the SHAP explanations for the corresponding input instances. 
            Each dictionary includes attributes of the SHAP explanation object with any `numpy.ndarray` values converted 
            to Python lists for compatibility.

        Notes:
        ------
        - The explainer is initialized as a TreeExplainer, suitable for tree-based models like LightGBM, XGBoost, etc.
        - The function dynamically extracts attributes defined in the `shap.Explanation` constructor, excluding 'self', 
        to ensure compatibility even if the Explanation class attributes change.
        - Attributes that are `numpy.ndarray` types are converted to lists for FastAPI serialization.
        """          
        #prepare explainer
        classifier = models[model_name_v].model.steps[-1][1]
        explainer = shap.explainers.TreeExplainer(classifier, models[model_name_v].X_train_treated)

        # prepare inputs 
        X_input = credit_request_dict_to_dataframe(input_data)        
        model_pretreatment = Pipeline(models[model_name_v].model.steps[:-1])        
        X_treated = model_pretreatment.transform(X_input)

        # shapify inputs
        shap_values = explainer(X_treated)
        

        # extract explanation attributes to transfer
        argument_list = signature(shap.Explanation.__init__).parameters.keys()

        explanation_attrs = []
        for shp_vls in shap_values:
            attrs_dict = {
                    attr: (getattr(shp_vls, attr).tolist() if isinstance(getattr(shp_vls, attr), np.ndarray) else getattr(shp_vls, attr))
                    for attr in argument_list if attr != 'self'
                }
            explanation_attrs.append(attrs_dict)

        return explanation_attrs
    
    shap_value_attributes.__doc__ = shap_value_attributes.__doc__.format(model_name_v=model_name_v)
    app.post(f"/{model_name_v}/shap_value_attributes")(shap_value_attributes)
    