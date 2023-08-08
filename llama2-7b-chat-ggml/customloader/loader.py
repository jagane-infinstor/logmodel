import mlflow
from llama_cpp import Llama

class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def predict(self, model_input):
        print(f'LlamaCppModel: predict. Entered. model_input={model_input}', flush=True)
        return self.my_custom_function(model_input)

    def my_custom_function(self, model_input):
        print(f'LlamaCppModel: my_custom_function. Entered. model_input={model_input}', flush=True)
        global llama_cpp_model
        # do something with the model input
        return 0

def _load_pyfunc(data_path):
    print(f"_load_pyfunc: Entered. data_path={data_path}", flush=True)
    llm = Llama(model_path=data_path)
    print(f"_load_pyfunc: llm={llm}", flush=True)
    global llama_cpp_model
    llama_cpp_model = llm
    return LlamaCppModel()
