import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import  create_model
from api_deployment.aux_func import get_file_from, ScoringModel, load_models
from typing import Any
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from inspect import signature
from shap import Explanation
from projet07.model_evaluation import get_feature_importance_from_model
from dotenv import load_dotenv
import sqlite3
import pickle

app = FastAPI(
        title = 'Credit Scoring API',
        description= 'API for serving one or more models that retreiveve structured' \
             + 'information on credit applications and returns an score and a validation treshold ' \
             + 'over which the probability of default makes the approval of the credit unadvisable',
        version="0.1.0"
)

load_dotenv()
# size of random credit data to whose shap values will be initialized to be served in beeswarm plot
SHAP_SAMPLE_SIZE = int(os.getenv('SHAP_SAMPLE_SIZE',-1)) 

json_file_path = os.getenv('MODELS_TO_DEPLOY_JSON')
json_file = get_file_from(json_file_path, 'models_to_deploy.json')

credit_requests_path = os.getenv('CREDIT_REQUESTS_DB')
credit_requests_db =  get_file_from(credit_requests_path, 'credit_requests.db')

input_information_csv = 'data/input_information.csv' # TODO: integrate to database to make it cleaner

# Create a model dynamicaly based on the kinds of inputs expected
input_information = pd.read_csv(input_information_csv, index_col=0)
field_types = input_information['Dtype'].apply(
    lambda x : eval(f'({x} | dict[int, {x}], ...)')).to_dict()
ModelEntries = create_model('ModelEntries', **field_types)

del(input_information)

models: dict[str, ScoringModel] = load_models(json_file, SHAP_SAMPLE_SIZE, credit_requests_db)

@app.post('/get_credit_application')
def get_credit_application(SK_ID_CURR:int) -> ModelEntries:
    query = f"""
    SELECT * FROM (
        SELECT * FROM application_test
        UNION
        SELECT * FROM application_train
        )
    WHERE SK_ID_CURR = {SK_ID_CURR}
    """
    connection_obj = sqlite3.connect(credit_requests_db)
    credit_application = pd.read_sql_query(query,
                        connection_obj, 
                        index_col='SK_ID_CURR'
                    ).replace(
                        {np.nan:None}
                    )
    return credit_application.loc[SK_ID_CURR].to_dict()

@app.get("/available_model_name_version")
def get_available_model_name_version()->list[str]:
    """ 
    Provides a list of `model_name_version` strings that indicate which versions of which models are abeing served by the API
    """
    return list(models.keys())

def credit_request_dict_to_dataframe(input_data: ModelEntries): # type: ignore
    input_data = input_data.dict()
    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys())
    input_df = input_df.replace({None:np.nan}).T
    return input_df


def validate_client( input_data: ModelEntries, model_name_v: str)->dict[str,float|bool|list[float|bool]]: # type: ignore
        
    if model_name_v not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    input_df = credit_request_dict_to_dataframe(input_data)

    y_pred_proba = models[model_name_v].model.predict_proba(input_df)[:, 1]
    validation_threshold = models[model_name_v].validation_threshold
    
    respond = {'default_probability' : y_pred_proba,
                'validation_threshold' : validation_threshold,
                'credit_approved' : y_pred_proba < validation_threshold,
                }
    
    return respond    


def create_validate_endpoint(model_name_v: str):
    async def endpoint(input_data: ModelEntries):
        return await validate_client(input_data=input_data, model_name_v=model_name_v)

    # Customize the docstring for each endpoint to reflect the specific model
    endpoint.__doc__ = f"""
    Validate client status using the '{model_name_v}' model.

    This endpoint predicts the probability of default for a client based on the
    '{model_name_v}' model. If the predicted probability is below the model's
    validation threshold, credit is approved.

    Parameters:
    -----------
    input_data : ModelEntries
        The client's data used for prediction.

    Returns:
    --------
    dict[str, float | bool | list[float | bool]]
        - 'default_probability' (list[float]): Predicted default probability.
        - 'validation_threshold' (float): Approval threshold for the model.
        - 'credit_approved' (list[bool]): Indicates if credit is approved.
    """
    return endpoint

# Register each model-specific endpoint with a unique docstring
for model_name_v in models.keys():
    app.post(f"/{model_name_v}/validate_client")(create_validate_endpoint(model_name_v))


for model_name_v in models.keys():       
    def shap_value_attributes(input_data: ModelEntries|None=None)->dict[str,Any]: # type: ignore
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
        
        if input_data is None:
            with open(models[model_name_v].shap_values_sample_path, 'rb') as file:
                shap_values = pickle.load(file)            
        else:            
            # Prepare inputs 
            X_input = credit_request_dict_to_dataframe(input_data)        
            model_pretreatment = Pipeline(models[model_name_v].model.steps[:-1])        
            X_treated = model_pretreatment.transform(X_input)
            
            # Load explainer
            with open(models[model_name_v].explainer_path, 'rb') as file:
                explainer = pickle.load(file)

            # shapify inputs
            shap_values = explainer(X_treated)       
        
        # extract explanation attributes to transfer
        argument_list = list(signature(Explanation.__init__).parameters.keys())
        argument_list.remove('self')
        
        explanation_attrs = {attr: (getattr(shap_values, attr).tolist() 
                                    if isinstance(getattr(shap_values, attr), np.ndarray) 
                                    else  getattr(shap_values, attr)
                                )
                                for attr in argument_list
                            }        
        return explanation_attrs
    shap_value_attributes.__doc__ = shap_value_attributes.__doc__.format(model_name_v=model_name_v)
    app.post(f"/{model_name_v}/shap_value_attributes")(shap_value_attributes)
    
    def get_global_feature_importance(
            cum_importance_cut:float = Query(ge=0.0, le=1.0, description='Cumulative importance cut should be between 0 and 1') 
            )->dict:
        """
        Extract feature importances from a model pipeline and return a DataFrame 
        with the most important features based on cumulative importance.

        This function assumes that the model is a pipeline where the final step 
        is a classifier with a `feature_importances_` attribute (e.g., 
        LightGBMClassifier). It calculates the normalized importance of each 
        feature and selects the features whose cumulative importance reaches a 
        specified threshold.

        Args:
            model (Pipeline): A scikit-learn pipeline object with a classifier 
                            as the last step that exposes `feature_importances_`.
                            The classifier must have a `feature_names_in_` 
                            attribute to get the feature names.
            cum_importance_cut (float): Cumulative importance threshold to select 
                                        the most important features. 
        Returns:
            df (Dataframe.to_dict()) A DataFrame with the following columns:
                - 'feature': Names of the features.
                - 'importance': Importance values from the classifier.
                - 'importance_normalized': Normalized importance values.
                - 'cum_importance_normalized': Cumulative normalized importance.

            most_important_features (list): List of feature names that together 
                                            account for the cumulative importance 
                                            up to the specified threshold.
        """
        feature_importances_domain, most_important_features = get_feature_importance_from_model(models[model_name_v].model.steps[-1][1], cum_importance_cut)
        return {'feature_importances_domain': feature_importances_domain.to_dict(),
                'most_important_features': most_important_features                
                }
    get_global_feature_importance.__doc__ = get_global_feature_importance.__doc__.format(model_name_v=model_name_v)
    app.post(f"/{model_name_v}/get_global_feature_importance")(get_global_feature_importance)
    
    
    