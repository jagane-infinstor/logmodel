import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import mlflow

model_name = 'meta-llama/Llama-2-7b-chat-hf'

print(f'Logging model {model_name}')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16)

with mlflow.start_run():
    mlflow.transformers.log_model(transformers_model=pipe, artifact_path="my_pipeline")

