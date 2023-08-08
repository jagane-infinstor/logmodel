import mlflow
import sys
import subprocess

deps = mlflow.pyfunc.get_model_dependencies(sys.argv[1])
print(f'model dependencies={deps}')
if deps:
    result = subprocess.run(['pip', 'install', '-r', deps], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
#pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False, dst_path='loaded_model')
pfmodel = mlflow.pyfunc.load_model(sys.argv[1], suppress_warnings=False)
print(f'pfmodel={pfmodel}')
pfmodel.predict('{"columns":["message"],"data":[["system: You are a helper named Oracle"], ["user: Joe.User: Explain existentialism"]]}')
