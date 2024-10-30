
import requests
import pandas as pd
import numpy as np
import json 

FAST_API = 'http://127.0.0.1:8000'

def request_model(model_uri, request, data):
    response = requests.post( f"{model_uri}/{request}/", json=data,)
    return response


X_train = pd.read_csv('data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])

input_data = X_train.sample(100).replace({np.nan:None}).to_dict()


print('Testing validate client')

response = request_model(FAST_API, 'deployment_v1/validate_client/', input_data)

if response.status_code==200:
    prediction = json.loads(response.text)            
    print(prediction)
else:
    print(response.text)

print('Testing shap values')

response = request_model(FAST_API, 'deployment_v1/shap_value_attributes', input_data)

if response.status_code==200:
    prediction = json.loads(response.text)            
    print(prediction)    
else:
    print(response.text)


print('Global shap values')

response = request_model(FAST_API, 'deployment_v1/shap_value_attributes', None)

if response.status_code==200:
    prediction = json.loads(response.text)            
    print(prediction.keys())    
else:
    print(response.text)