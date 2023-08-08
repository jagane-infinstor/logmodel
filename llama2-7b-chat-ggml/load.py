import mlflow
import sys
import subprocess
import pandas as pd

deps = mlflow.pyfunc.get_model_dependencies(sys.argv[1])
print(f'model dependencies={deps}')
if deps:
    result = subprocess.run(['pip', 'install', '-r', deps], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False, dst_path='loaded_model')
#pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False)
print(f'pfmodel={pfmodel}')

data = {'role': ['system', 'user'], 'message': ['You are a helper named Oracle', 'Explain existentialism']}
df = pd.DataFrame.from_dict(data)
pfmodel.predict(df)
