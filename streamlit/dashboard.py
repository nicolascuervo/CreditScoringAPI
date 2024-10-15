import pandas as pd
import streamlit as st
import requests
import ast
import numpy as np

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def load_dashboard(inputs:pd.DataFrame):
    output = {}
    for feature in inputs.index:
        
        if inputs.loc[feature,'Dtype'] =='object':
            options = ast.literal_eval(inputs.loc[feature,'categories'].replace('nan', "'NA'"))
            output[feature] = st.selectbox(
                label=inputs.loc[feature, 'Column'],
                options=options,
                # value=inputs.loc[feature, 'value'],
                help=inputs.loc[feature, 'Description']
                )
        elif inputs.loc[feature,'has_nan'] :
            output[feature] = st.text_input(
                label=inputs.loc[feature, 'Column'],                         
                value=inputs.loc[feature, 'value'],
                help=inputs.loc[feature, 'Description']
                )
        elif inputs.loc[feature,'Dtype'] == 'int64':
            output[feature] = st.number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=float(inputs.loc[feature, 'value']),
                step=1.0,
                help=inputs.loc[feature, 'Description']
                )
        else:
            output[feature] = st.number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=float(inputs.loc[feature, 'value']),
                help=inputs.loc[feature, 'Description'],
                )    
    return output  
      


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    # CORTEX_URI = 'http://0.0.0.0:8890/'
    # RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    # api_choice = st.sidebar.selectbox(
    #     'Quelle API souhaitez vous utiliser',
    #     ['MLflow', 'Cortex', 'Ray Serve'])
    inputs = pd.read_csv("streamlit/input_information.csv", index_col=0)
    
    st.title('Consult credit approval')

    output = load_dashboard(inputs)
    

    predict_btn = st.button('Prédire')
    if predict_btn:
        print(output)
        pred = None

        # if api_choice == 'MLflow':
        #     pred = request_prediction(MLFLOW_URI, data)[0] * 100000
        # elif api_choice == 'Cortex':
        #     pred = request_prediction(CORTEX_URI, data)[0] * 100000
        # elif api_choice == 'Ray Serve':
        #     pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        pred = 0.95

        st.write(
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred))


if __name__ == '__main__':
    main()
