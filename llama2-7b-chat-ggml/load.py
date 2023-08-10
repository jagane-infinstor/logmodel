import mlflow
import sys
import subprocess
import pandas as pd

deps = mlflow.pyfunc.get_model_dependencies(sys.argv[1])
print(f'model dependencies={deps}')
if deps:
    result = subprocess.run(['pip', 'install', '-r', deps], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

#pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False, dst_path='loaded_model')
pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False)
print(f'pfmodel={pfmodel}')

unwrapped_model = pfmodel.unwrap_python_model()
print(f'unwrapped_model={unwrapped_model}')

roles = []
messages = []
roles.append('system')
messages.append('You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.')
while True:
    line = input('> ')
    roles.append('user')
    messages.append(line)

    data = {'role': roles, 'message': messages}
    df = pd.DataFrame.from_dict(data)
    pred = unwrapped_model.predict(df, {'max_tokens': 256})
    print(pred)
    roles.append('assistant')
    messages.append(pred['choices'][0]['text'])
