import mlflow
from llama_cpp import Llama
import pandas as pd

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
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
        global llama_cpp_model
        output = [] # llama_cpp_model(prompt, max_tokens=max_tokens, stop=[], echo=False)
        return output

def _load_pyfunc(data_path):
    print(f"_load_pyfunc: Entered. data_path={data_path}", flush=True)
    llm = Llama(model_path=data_path)
    print(f"_load_pyfunc: llm={llm}", flush=True)
    global llama_cpp_model
    llama_cpp_model = llm
    return LlamaCppModel()
