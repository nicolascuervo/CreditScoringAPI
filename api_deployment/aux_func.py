
import requests
import zipfile
from pydantic import HttpUrl
from pathlib import Path
import gdown
import os
import sqlite3
import numpy as np
from shap.explainers import TreeExplainer
import pandas as pd
import pickle
import gc
from tqdm import tqdm
from imblearn.pipeline import Pipeline


def create_explainer(model, explainer_path):
    # load train data from database
    print('\t\tCreating explainer')
    explainer = TreeExplainer(model.steps[-1][1], feature_perturbation="tree_path_dependent")
    # Store explainer
    with open(explainer_path,'wb') as file :
        pickle.dump(explainer, file)

def transform_X_train_by_chunks_in_db(steps, connection_obj, chunk_size):
    
    # Initialize table by clearing any existing data
    connection_obj.execute('DROP TABLE IF EXISTS transcient_table;')
    total_rows = pd.read_sql_query('SELECT COUNT(*) FROM application_train', connection_obj).iloc[0, 0]
    total_chunks = (total_rows // chunk_size) + 1
    print('\t\tPreparing data for explainer deployment')
    for chunk in tqdm(pd.read_sql_query('SELECT * FROM application_train',
                                    connection_obj, 
                                    index_col='SK_ID_CURR', 
                                    chunksize=chunk_size),
                      total=total_chunks, 
                      desc="\t\tTransforming chunks"):
            
        # Replace None with NaN
        chunk = chunk.replace({None: np.nan})

        # Apply each step of the transformation pipeline to the chunk
        
        chunk = Pipeline(steps).transform(chunk)
        gc.collect()  # Clear memory after each transformation

        # Append transformed chunk to temporary table
        chunk.to_sql('transcient_table', connection_obj, if_exists='append', index=True)

        # Clear chunk from memory and run garbage collection
        del chunk
        gc.collect()


def create_shap_values(model,
                       sample_size:int, 
                       shap_values_sample_path:str, 
                       explainer_path: str,
                       database_name:str):
    print('\t\tCreating global shap_value sample')
    # load random sample from database
    connection_obj = sqlite3.connect(database_name)
    query = f"""SELECT * FROM application_train
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """    
    X = pd.read_sql_query(query,
                            connection_obj, 
                            index_col='SK_ID_CURR'
                            ).replace({None:np.nan})
    # apply pretreatments
    for step in model.steps[:-1]:
        X = step[1].transform(X)
        gc.collect()

    # load explainer
    with open(explainer_path, 'rb') as file:
        explainer = pickle.load(file)

    # calculate shap values
    global_shap_values = explainer(X)
    
    # store shap values
    with open(shap_values_sample_path,'wb') as file :
        pickle.dump(global_shap_values, file)

def validate_path_or_url(value):
    # Check if it's a valid URL
    try:
        HttpUrl(value)
        return value
    except Exception:
        pass

    # Check if it's a valid folder path
    path = Path(value)
    if path.is_dir() or path.is_file():
        return value
    raise ValueError("Input must be a valid URL or a folder path.")

def get_file_from(file_address, download_file_name=None):
    file_address = validate_path_or_url(file_address)
    
    if file_address.startswith('http'):
        # Determine destination path
        if download_file_name is None:
            raise ValueError("Please provide a download_file_name for remote files.")
        file_name, ext = download_file_name.rsplit('.', 1)
        local_path = f"data/{file_name}.{ext}"
        
        if not os.path.exists(local_path):
        # Use gdown for Google Drive links
            if "drive.google.com" in file_address :
                gdown.download(file_address, local_path, fuzzy=True)
            else:
                # Standard download for other URLs
                response = requests.get(file_address, stream=True)
                with open(local_path, "wb") as file:
                    for chunk in response.iter_content(32768):  # Chunk size for large downloads
                        file.write(chunk)
            
        # Handle zip extraction if needed
        if ext == 'zip':
            file_dir = f"data/{file_name}"
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(file_dir)
            local_path = file_dir
    else:
        local_path = file_address
    
    return local_path