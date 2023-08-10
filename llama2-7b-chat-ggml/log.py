import mlflow
import sys
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import DataType, Schema, ColSpec, ParamSchema, ParamSpec

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

# role: system, user or assistant
input_schema = Schema([ColSpec(DataType.string, "role", False), ColSpec(DataType.string, "message", False)])

# Return messages are of the following format
# assistant: [name]: content
# assistant: : The river that carves the deepest valley flows from a modest spring; the grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread.
output_schema = Schema([ColSpec(DataType.string, None, False)])

params = ParamSchema([ParamSpec('max_tokens', DataType.integer, 32)])

sign = ModelSignature(input_schema, output_schema, params)
print(sign)

with mlflow.start_run():
    print(f"Using model data {sys.argv[1]}")
    mlflow.pyfunc.log_model("model", loader_module='customloader.loader', data_path=sys.argv[1], code_path=['customloader'], conda_env=con, signature=sign)
    mlflow.log_artifact('Dockerfile', artifact_path='model')
