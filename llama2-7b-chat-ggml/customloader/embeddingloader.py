import mlflow
from llama_cpp import Llama
import pandas as pd

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def __init__(self, data_path):
        self.data_path = data_path
        self.llama_cpp_model = None
        self.python_model = self

    def predict(self, model_input, params=None):
        return self.predict_plus(model_input, params)

    def predict_plus(self, model_input, params=None):
        print(f'LlamaCppModel.predict_plus. Entered. model_input={model_input}', flush=True)
        if params and 'verbose' in params:
            verbose = True
        else:
            verbose = False
        model_input.reset_index()
        if self.llama_cpp_model == None:
            if params and 'n_gpu_layers' in params:
                n_gpu_layers = int(params['n_gpu_layers'])
                print(f"LlamaCppModel.predict_plus: loading model with n_gpu_layers={n_gpu_layers}")
                self.llama_cpp_model = Llama(model_path=self.data_path, embedding=True, n_gpu_layers=n_gpu_layers)
            else:
                print(f"LlamaCppModel.predict_plus: loading model n_gpu_layers not specified")
                self.llama_cpp_model = Llama(model_path=self.data_path, embedding=True)
            print(f"predict_plus: llama_cpp_model={self.llama_cpp_model}", flush=True)
        output = []
        for index, row in model_input.iterrows():
            emb = self.llama_cpp_model.create_embedding(row['text'])
            if verbose:
                print(f"text={row['text']}, embedding={emb}", flush=True)
            output.append({'text': row['text'], 'embedding': emb['data'][0]['embedding']})
        return output

def _load_pyfunc(data_path):
    print(f"_load_pyfunc: Entered. data_path={data_path}", flush=True)
    return LlamaCppModel(data_path)
