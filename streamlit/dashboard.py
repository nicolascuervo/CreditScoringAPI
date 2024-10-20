import pandas as pd
import streamlit as st
import requests
import ast
import numpy as np
import json

def request_model(model_uri, request, data):
    response = requests.post( f"{model_uri}/{request}/", json=data,)
    return response

def load_dashboard(inputs:pd.DataFrame):
    output: dict[str: str|float|int] = {}
    
    for feature in inputs.index:
        value = inputs.loc[feature, 'value']


        if inputs.loc[feature,'Dtype'] =='object':
            if value=='nan' or pd.isna(value):
                value='NA'                
            options = ast.literal_eval(inputs.loc[feature,'categories'].replace('nan', "'NA'"))
            output[feature] = st.selectbox(
                label=inputs.loc[feature, 'Column'],
                options=options,
                index=options.index(value),
                help=inputs.loc[feature, 'Description']
                )
            if output[feature] == 'NA':
                output[feature] = ''
            
        elif inputs.loc[feature,'has_nan'] :
            if pd.isna(value):
                value='na'
            output[feature] = st.text_input(
                label=inputs.loc[feature, 'Column'],                         
                value=value,
                help=inputs.loc[feature, 'Description']
                ) 
            if output[feature]=='' or output[feature] == 'na':
                output[feature] = ''
            elif inputs.loc[feature,'Dtype'] =='float64':
                output[feature] = float(output[feature])
            elif inputs.loc[feature,'Dtype'] =='int64':
                output[feature] = int(output[feature])
            

        elif (inputs.loc[feature,'Dtype']=='int64') and (inputs.loc[feature,'n_unique']==2):            
            output[feature] = st.toggle(            
                label=inputs.loc[feature, 'Column'],
                value=int(value)==1,
                help=inputs.loc[feature, 'Description']
                )
            output[feature] = int(output[feature])
        elif (inputs.loc[feature,'Dtype']=='int64'):
            output[feature] = st.number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=int(value),
                step=1,
                help=inputs.loc[feature, 'Description'],
                )              
        else:
            output[feature] = st.number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=float(value),
                format=inputs.loc[feature, 'format'],
                help=inputs.loc[feature, 'Description'],
                )  
        
    return output  
      


def main():
    
    FAST_API = 'http://127.0.0.1:8000'
    
    inputs = pd.read_csv("streamlit/input_information.csv", index_col=0)
    
    st.title('Consult credit approval')
    output = load_dashboard(inputs)
    predict_btn = st.sidebar.button('Prédire')
    if predict_btn:
        
        response = request_model(FAST_API, 'validate_client/deployment_v1', output)
        
        if response.status_code==200:
            prediction = json.loads(response.text)
            
            if prediction['credit_approved']:
                approval_text=f"Credit approved:"
                explanation_text = f"Default probability {prediction['default_probability']:0.1%} is lower than recommended threshold {prediction['validation_threshold']:0.1%}"
            else:
                approval_text=approval_text=f"Credit denied:"
                explanation_text = f" Default probability {prediction['default_probability']:0.1%} is greater or equal to recommended threshold {prediction['validation_threshold']:0.1%}"
                
            st.sidebar.write(approval_text)
            st.sidebar.write(explanation_text)
        else:
            st.sidebar.write(response.text)



if __name__ == '__main__':
    main()
