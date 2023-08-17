import mlflow
import sys
import subprocess
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Chat with a llama2 ggml model")
parser.add_argument('--model', required=True, help='chat MLflow model uri or local dir with model bits')

args = parser.parse_args()
print(f"Using data {args.model} for chat", flush=True)


deps = mlflow.pyfunc.get_model_dependencies(args.model)
print(f'model dependencies={deps}')
if deps:
    result = subprocess.run(['pip', 'install', '-r', deps], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

pfmodel = mlflow.pyfunc.load_model(args.model, suppress_warnings=False, dst_path='loaded_model')
#pfmodel = mlflow.pyfunc.load_model(args.model, suppress_warnings=False)
unwrapped_model = pfmodel.unwrap_python_model()

text = []
while True:
    line = input('> ')
    text.append(line)

    data = {'text': text}
    df = pd.DataFrame.from_dict(data)
    pred = unwrapped_model.predict(df, {'max_tokens': 256, 'verbose': True})
    #print(f"assistant> {pred['choices'][0]['text']}")
    #messages.append(pred['choices'][0]['text'])
