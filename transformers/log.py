from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import mlflow

#model = "tiiuae/falcon-7b-instruct"
#model = "gpt2-large"
#model = 'hf-tiny-model-private/tiny-random-BartForCausalLM'
model = 'google/t5-v1_1-small'

print(f'Logging model {model}')

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

with mlflow.start_run():
    mlflow.transformers.log_model(transformers_model=pipeline, artifact_path="my_pipeline")
    mlflow.log_artifact('Dockerfile', artifact_path='my_pipeline')
