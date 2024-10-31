import os
from fastapi import FastAPI, Query
from pydantic import BaseModel, create_model
from api_deployment.aux_func import get_file_from
import mlflow
import json
from typing import Any
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from inspect import signature
import shap
from projet07.model_evaluation import get_feature_importance_from_model
from dotenv import load_dotenv
app = FastAPI(
        title = 'Credit Scoring API',
        description= 'API for serving one or more models that retreiveve structured' \
             + 'information on credit applications and returns an score and a validation treshold ' \
             + 'over which the probability of default makes the approval of the credit unadvisable',
        version="0.1.0"
)

load_dotenv()
# size of random credit data to whose shap values will be initialized to be served in beeswarm plot
SHAP_SAMPLE_SIZE = int(os.getenv('SHAP_SAMPLE_SIZE')) 

json_file_path = os.getenv('MODELS_TO_DEPLOY_JSON')
json_file = get_file_from(json_file_path, 'models_to_deploy.json')

application_train_path = os.getenv('APPLICATION_TRAIN_CSV')
application_train_csv =  get_file_from(application_train_path, 'application_train.csv')

input_information_csv = 'data/input_information.csv'




X_train = pd.read_csv(application_train_csv, index_col='SK_ID_CURR')
if 'TARGET' in X_train.columns:
    X_train = X_train.drop(columns=['TARGET'])

# Create a model dynamicaly based on the kinds of inputs expected
input_information = pd.read_csv(input_information_csv, index_col=0)
field_types = input_information['Dtype'].apply(
    lambda x : eval(f'({x} | dict[int, {x}], ...)')).to_dict()

ModelEntries = create_model('ModelEntries', **field_types)

with open(json_file,'r') as file:
    models_to_deploy = json.load(file)

class ScoringModel(BaseModel):    
    model: Any    
    validation_threshold: float
    X_train_treated: Any
    global_shap_values: Any 

    class Config:
        arbitrary_types_allowed = True  # Allows non-primitive types like sklearn 
       

models: dict[str, ScoringModel] = {}
for model_info in models_to_deploy:
    
    key = model_info['model_name'] + '_v' + model_info['version']
    print('INFO:', f'Loading model {key}...', end='\t')
    
    model_dir = get_file_from(model_info["source"], f'{key}.zip')
    model = mlflow.pyfunc.load_model(model_uri=model_dir)._model_impl.sklearn_model

    X_train_treated = Pipeline(model.steps[:-1]).transform(X_train)
    explainer = shap.explainers.TreeExplainer(model.steps[-1][1], X_train_treated)
    global_shap_values = explainer(X_train_treated.sample(SHAP_SAMPLE_SIZE))

    models[key] = ScoringModel(
        model=model,
        validation_threshold=model_info['validation_threshold'],
        X_train_treated=X_train_treated,
        global_shap_values=global_shap_values
    )
    print('Done')


    

@app.get("/available_model_name_version")
def get_available_model_name_version()->list[str]:
    """ Provides a list of `model_name_version` strings that indicate which versions of which models are abeing served by the API
    """
    return list(models.keys())

def credit_request_dict_to_dataframe(input_data: ModelEntries): # type: ignore
    input_data = input_data.dict()
    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys())
    input_df = input_df.replace({None:np.nan}).T
    return input_df

for model_name_v in models.keys():    
    async def validate_client( input_data: ModelEntries )->dict[str,float|bool|list[float|bool]]: # type: ignore
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
        
        respond = {'default_probability' : y_pred_proba,
                    'validation_threshold' : validation_threshold,
                    'credit_approved' : y_pred_proba < validation_threshold,
                    }
        
        return respond    
    validate_client.__doc__ = validate_client.__doc__.format(model_name_v=model_name_v)    
    app.post(f"/{model_name_v}/validate_client")(validate_client)
    
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
            shap_values = models[model_name_v].global_shap_values
            
        else:            
            # prepare inputs 
            X_input = credit_request_dict_to_dataframe(input_data)        
            model_pretreatment = Pipeline(models[model_name_v].model.steps[:-1])        
            X_treated = model_pretreatment.transform(X_input)
            #prepare explainer
            classifier = models[model_name_v].model.steps[-1][1]
            explainer = shap.explainers.TreeExplainer(classifier, models[model_name_v].X_train_treated)
            # shapify inputs
            shap_values = explainer(X_treated)       
        
        # extract explanation attributes to transfer
        argument_list = list(signature(shap.Explanation.__init__).parameters.keys())
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
    
    
    