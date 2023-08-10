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

        default_system = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.'

        if params:
            max_tokens = int(params['max_tokens'])
            print(f'LlamaCppModel.predict_plus. from params max_tokens={max_tokens}, params={params}')
        else:
            max_tokens = 32
            print(f'LlamaCppModel.predict_plus. default max_tokens={max_tokens}')

        model_input.reset_index()
        prompt = ''
        for index, row in model_input.iterrows():
            print(f"  role={row['role']}, message={row['message']}")
            if index == 0:
                if row['role'] == 'system':
                    prompt = f"<s>[INST] <<SYS>>\n{row['message']}\n\n<</SYS>>\n\n"
                    continue
                else:
                    print(f"    Using default system message since the first message's role is {row['role']} and not system")
                    prompt = f'<s>[INST] <<SYS>>\n{default_system}\n\n<</SYS>>\n\n'
                    if row['role'] == 'user':
                        prompt = prompt + f"{row['message']} [/INST]"
                    continue
            if row['role'] == 'user':
                prompt = prompt + f"<s>[INST] {row['message']} [/INST]"
                continue
            if row['role'] == 'assistant':
                assistant = row['message']
                prompt = prompt + f' {assistant} </s>'

        prompt = '<s>[INST] '
        is_inside_elem = True
        for index, row in model_input.iterrows():
            print(f"  role={row['role']}, message={row['message']}. prompt={prompt}, is_inside_elem={is_inside_elem}")
            if not is_inside_elem:
                prompt = prompt + '<s>[INST] '
            if index == 0:
                if row['role'] == 'system':
                    prompt = f"<<SYS>>\n{row['message']}\n\n<</SYS>>\n\n"
                    continue
                else:
                    print(f"    Using default system message since the first message's role is {row['role']} and not system")
                    prompt = f'<<SYS>>\n{default_system}\n\n<</SYS>>\n\n'
                    if row['role'] == 'user':
                        prompt = prompt + f"{row['message']} [/INST]"
                    continue
            if row['role'] == 'user':
                prompt = prompt + f"{row['message']} [/INST]"
                continue
            if row['role'] == 'assistant':
                assistant = row['message']
                prompt = prompt + f' {assistant} </s>'
                is_inside_elem = False
        print(f"Final Prompt={prompt}")

        global llama_cpp_model
        output = llama_cpp_model(prompt, max_tokens=max_tokens, stop=[], echo=False)
        print(output)
        return output

def _load_pyfunc(data_path):
    print(f"_load_pyfunc: Entered. data_path={data_path}", flush=True)
    llm = Llama(model_path=data_path)
    print(f"_load_pyfunc: llm={llm}", flush=True)
    global llama_cpp_model
    llama_cpp_model = llm
    return LlamaCppModel()
