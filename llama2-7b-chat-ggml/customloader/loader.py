import mlflow
from llama_cpp import Llama
import pandas as pd

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.python_model = self

    def predict(self, model_input, params=None):
        print(f'LlamaCppModel: predict. Entered. model_input={model_input}, params={params}', flush=True)
        return self.predict_plus(model_input, params)

    def predict_plus(self, model_input, params=None):
        print(f'LlamaCppModel.predict_plus. Entered. model_input={model_input}', flush=True)

        if params:
            max_tokens = int(params['max_tokens'])
            print(f'LlamaCppModel.predict_plus. from params max_tokens={max_tokens}, params={params}')
        else:
            max_tokens = 32
            print(f'LlamaCppModel.predict_plus. default max_tokens={max_tokens}')

        model_input.reset_index()
        user = ''
        system = ''
        for index, row in model_input.iterrows():
            print(f"  role={row['role']}, message={row['message']}")
            if row['role'] == 'system':
                system = row['message']
            if row['role'] == 'user':
                user = row['message']
        prompt = f'<s>[INST] <<SYS>>\n{system}\n\n<</SYS>>\n\n{user}[/INST]\n'
        print(f"Prompt={prompt}")

        global llama_cpp_model
        output = llama_cpp_model(prompt, max_tokens=max_tokens, stop=[], echo=True)
        print(output)
        return 0

def _load_pyfunc(data_path):
    print(f"_load_pyfunc: Entered. data_path={data_path}", flush=True)
    llm = Llama(model_path=data_path)
    print(f"_load_pyfunc: llm={llm}", flush=True)
    global llama_cpp_model
    llama_cpp_model = llm
    return LlamaCppModel()
