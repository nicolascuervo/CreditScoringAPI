import pandas as pd
import streamlit as st
import requests
import ast
import numpy as np
import json
from pydantic import create_model

FAST_API = 'http://127.0.0.1:8000'
MIN_SK_ID_CURR = 100001
MAX_SK_ID_CURR = 456255

@st.cache_data
def load_input_information():
    """Load the input information CSV and cache it."""
    return pd.read_csv("data/input_information.csv", index_col=0)

input_information = load_input_information()
field_types = input_information['Dtype'].apply(lambda x : eval(f'({x}, ...)')).to_dict()
ModelEntries = create_model('ModelEntries', **field_types)

def main():

    # Streamlit tabs
    tab_names = ['Credit Approval', ' Particular decision factors', 'General decision factors']
    credit_approval_tab, local_feature_importance_tab, global_feature_importabce_tab = st.tabs(tab_names)    

    # credit ID form
    if 'sk_id_curr0' in st.session_state:
        sk_id_curr0 = st.session_state['sk_id_curr0']        
    else:
        sk_id_curr0 = np.random.randint(MIN_SK_ID_CURR, MAX_SK_ID_CURR)
        st.session_state['sk_id_curr0'] = sk_id_curr0
    credit_id_form = credit_approval_tab.form("load_credit_form")
    row_0 = credit_id_form.columns([1,0.5,3])    
    sk_id_curr = row_0[0].number_input(
        label='SK_ID_CURR', 
        min_value=MIN_SK_ID_CURR,                        
        max_value=MAX_SK_ID_CURR,
        value=sk_id_curr0,
        step=1,
        help='ID of loan',
        )
    load_application = row_0[0].form_submit_button('Load Application')

    # Loan application form
    loan_request_detail_form = credit_approval_tab.form("Loan request Details")
    row_1 = loan_request_detail_form.columns([1,1,1])        
    submit_credit = row_1[1].form_submit_button('Submit Credit Request')
    

    if load_application:        
        loan_request_data_0 = ModelEntries(**get_credit_application(sk_id_curr))
        st.session_state['loan_request_data_0'] = loan_request_data_0    

    if 'loan_request_data_0' in st.session_state:
        loan_request_data_0 = st.session_state['loan_request_data_0']
        loan_submission = load_credit_request_form(loan_request_detail_form, input_information, loan_request_data_0 )
        
    # evaluate credit
    if submit_credit:
        response = request_model(FAST_API, 'validate_client/deployment_v1', loan_submission)
        display_credit_request_response(row_0[2], response)


@st.cache_data
def load_full_application_data():
    """Load and cache the full application dataset."""
    X_train = pd.read_csv('data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])
    X_test = pd.read_csv('data/application_test.csv', index_col='SK_ID_CURR')
    X_full = pd.concat([X_train, X_test], axis=0).sort_index()    
    return X_full

def display_credit_request_response(container: st.container, response):
    if response.status_code==200:
        prediction = json.loads(response.text)            
        if prediction['credit_approved']:
            approval_text=f"Credit approved:"
            # explanation_text = f"Default probability {prediction['default_probability']:0.1%} "+\
            #                    f"is lower than recommended threshold {prediction['validation_threshold']:0.1%}"
            delta_color="normal"            
        else:
            approval_text=approval_text=f"Credit denied:"
            # explanation_text = f"Default probability {prediction['default_probability']:0.1%} "+\
            #                    f"is greater or equal to recommended threshold {prediction['validation_threshold']:0.1%}"
            
            
        delta_color="inverse"

        container.title(approval_text)        
        container.metric(label='Default probability:', 
                  value=f'{prediction['default_probability']:0.1%}', 
                  delta=f'{prediction['default_probability']-prediction['validation_threshold']:0.1%}',
                  delta_color=delta_color,)
    else:
        container.write(response.text)

def request_model(model_uri, request, data):
    response = requests.post( f"{model_uri}/{request}/", json=data,)
    return response


def get_credit_application(sk_id_curr: int)-> ModelEntries:
    X_full = load_full_application_data()
    return X_full.loc[sk_id_curr,:].replace({np.nan:None}).to_dict()
    


def load_credit_request_form(form: st.container, inputs: pd.DataFrame, credit_application_0: ModelEntries):
    form_output: dict[str: str|float|int|None] = {}
    credit_application_0 = credit_application_0.dict()
    
    columns = form.columns([1,1,1]) 
    for i, feature in enumerate(inputs.index):
        
        value = credit_application_0[feature]
        if value is None:
            value = 'NA'
        
        
        if inputs.loc[feature,'Dtype'].__contains__('str'):        
                     
            options = ast.literal_eval(inputs.loc[feature,'categories'].replace('nan', "'NA'"))
            form_output[feature] = columns[1].selectbox(
                label=inputs.loc[feature, 'Column'],
                options=options,
                index=options.index(value),
                help=inputs.loc[feature, 'Description']
                )
        elif inputs.loc[feature,'Dtype'].__contains__('None') :
            form_output[feature] = columns[0].text_input(
                label=inputs.loc[feature, 'Column'],                         
                value=value,
                help=inputs.loc[feature, 'Description']
                ) 
        elif (inputs.loc[feature,'Dtype'].__contains__('int') ) and (inputs.loc[feature,'n_unique']==2):            
            form_output[feature] = columns[2].toggle(            
                label=inputs.loc[feature, 'Column'],
                value=int(value)==1,
                help=inputs.loc[feature, 'Description']
                )    
        elif  (inputs.loc[feature,'Dtype'].__contains__('int') ) :
            form_output[feature] = columns[0].number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=int(value),
                step=1,
                help=inputs.loc[feature, 'Description'],
                )              
        else:
            form_output[feature] = columns[0].number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=float(value),
                format=inputs.loc[feature, 'format'],
                help=inputs.loc[feature, 'Description'],
                )  
        if form_output[feature] == '' or form_output[feature] == 'NA':
            form_output[feature] = None
    return form_output  


if __name__ == '__main__':
    main()
