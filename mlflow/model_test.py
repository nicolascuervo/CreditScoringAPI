import json
import requests
import argparse


if __name__ == "__main__":
    # Using argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Serve ML models using MLflow.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file with models to deploy")
    parser.add_argument("port0", type=int, help="First port to be open")
    args = parser.parse_args()


with open(args.json_file,'rb') as file:
    models = json.load(file)

for i, model in enumerate(models):
    print(f'Requesting model: {model['model_name']}, version={model['version']}')
    url = f'http://127.0.0.1:{args.port0+i}/invocations'
    headers = {"Content-Type": "application/json"}    
    with open(model['source']+'/serving_input_example.json','r') as file:
        input_example = file.read()
    response = requests.post(url, headers=headers, data=input_example)    
    print(response)
    print(response.text)


