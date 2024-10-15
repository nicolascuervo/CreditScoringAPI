import subprocess
import json
import argparse
import sys

def serve_model_on_port(model_path, port):
    subprocess.Popen(['mlflow', 'models', 'serve','-m', f'{model_path}', '-p', str(port)])



if __name__ == "__main__":
    # Using argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Serve ML models using MLflow.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file with models to deploy")
    parser.add_argument("port0", type=int, help="First port to be open")
    args = parser.parse_args()

with open(args.json_file,'r') as file:
    models_to_deploy = json.load(file)

for i, model in enumerate(models_to_deploy):
    serve_model_on_port(model['source'], args.port0 + i)


