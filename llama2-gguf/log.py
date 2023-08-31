import mlflow
import sys
import os
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import DataType, Schema, ColSpec, ParamSchema, ParamSpec
import argparse

parser = argparse.ArgumentParser(description="Logs a llama.cpp optimzed llama2 model as MLflow model for chat or embedding-generation tasks.")
parser.add_argument('--data_path', required=True, help='model binary, e.g. models/llama-2-7b-chat.ggmlv3.q2_K.bin')
parser.add_argument('--task', required=True, help="chat|embedding-generation")

args = parser.parse_args()
print(f"Using data {args.data_path} for task {args.task}", flush=True)

con = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9",
        {
            "pip": [
                "llama-cpp-python"
            ],
        },
    ],
}

def log_chat(dpath):
    # role: system, user or assistant
    input_schema = Schema([ColSpec(DataType.string, "role", False), ColSpec(DataType.string, "message", False)])
    output_schema = Schema([ColSpec(DataType.string, None, False)])
    params = ParamSchema([ParamSpec('max_tokens', DataType.integer, 32)])
    sign = ModelSignature(input_schema, output_schema, params)
    print(f"chat task: Model signature={sign}", flush=True)
    mlflow.pyfunc.log_model("model", loader_module='customloader.chatloader', data_path=dpath, code_path=['customloader'], conda_env=con, signature=sign)
    mlflow.log_artifact('Dockerfile', artifact_path='model')

def log_embedding(dpath):
    input_schema = Schema([ColSpec(DataType.string, "text", False)])
    output_schema = Schema([ColSpec(DataType.string, None, False)])
    params = ParamSchema([ParamSpec('max_tokens', DataType.integer, 32)])
    sign = ModelSignature(input_schema, output_schema, params)
    print(f"embedding-generation task: Model signature={sign}", flush=True)
    mlflow.pyfunc.log_model("model", loader_module='customloader.embeddingloader', data_path=dpath, code_path=['customloader'], conda_env=con, signature=sign)
    mlflow.log_artifact('Dockerfile', artifact_path='model')

with mlflow.start_run():
    if args.task == "chat":
        log_chat(args.data_path)
    elif args.task == 'embedding-generation':
        log_embedding(args.data_path)
    else:
        print(f"Error. unsupported task {args.task}")
        os._exit(255)
os._exit(0)
