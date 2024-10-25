
import requests
import pandas as pd
import numpy as np
import json 

FAST_API = 'http://127.0.0.1:8000'

def request_model(model_uri, request, data):
    response = requests.post( f"{model_uri}/{request}/", json=data,)
    return response


X_train = pd.read_csv('data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])
X_test = pd.read_csv('data/application_test.csv', index_col='SK_ID_CURR')
X_full = pd.concat([X_train, X_test], axis=0).sort_index()

input_data = X_train.sample(100).replace({np.nan:None}).to_dict()
# print(input_data)


response = request_model(FAST_API, 'validate_client/deployment_v1', input_data)

if response.status_code==200:
    prediction = json.loads(response.text)            
    print(prediction)
else:
    print(response.text)
