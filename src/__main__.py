from llm_sdk import Small_LLM_Model
import json
import numpy as np
from parser import parser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pathlib
import os
from functools import lru_cache
def check_json(text):
    try:
      print("test")
      j = json.loads(text)
      return (j)
    except:
        return None

mycash = dict()

class JsonFsm():
    def __init__(self):
        self.state = "START"
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.start_tokens = self.get_ids(["{"])
        self.keys_tokens = self.get_ids(['"prompt"'])


    def allowed_tokens(self):
        if self.state == "START":
            pass
        elif self.state == "EXPECT_KEY":
            pass
        elif self.state == "EXPECT_COLOM":
            pass
        elif self.state == "EXPECT_VALUE":
            pass
        elif self.state == "EXPECT_COMA_OR_END":
            pass
    def get_ids(self, tokens: list[str]):
        ids_tokens = []
        for token in tokens:
            ids_tokens.append([tok.item() for tok in self.llm.encode(token)[0]])
        return ids_tokens

def generate_json(prompt, llm, li: list):
    response = []
    text = ""
    prompt_3 = f"""{prompt["prompt"]}
Output:"""
    tensor_data = llm.encode(prompt_3)
    list_data = li.copy()
    list_data.extend(tensor_data.tolist()[0])
    for _ in range(100):
        logits = llm.get_logits_from_input_ids(list_data)
        # copy = [float('-inf') for _ in range(len(logits))]
        # for i in [9000, 15, 42]:
        #     copy[i] = logits[i]
        # next_id = copy.index(max(copy))
        np_logits = np.array(logits)
        next_id = np.argmax(np_logits).item()
        list_data.append(next_id)
        response.append(next_id)
        if mycash.get((next_id)) is None:
            mycash[(next_id)] = llm.decode([next_id])
        text += mycash[(next_id)]
        # print(text)
        # print(text)
        if text.strip().endswith("}"):
            json_response = check_json(text.strip())
            if json_response:
                break
    return (json_response)

def main():
    fun_json, functions_dif, prompts = parser()
    fsm = JsonFsm()
    parame_tokens = [list(key["parameters"].keys()) for key in fun_json]
    parame_tokens = ['"'+k+'"' for sublist in parame_tokens for k in sublist]
    tokens = ['"name"',  '"prompt"', '"parameters"']
    tokens.extend(parame_tokens)
    r = fsm.get_ids(tokens)
    print(tokens)
    print(r)
    exit(1)
    llm = Small_LLM_Model()
    output_list = []
    prompt_1 = f"""
AVAILABLE FUNCTIONS
    {functions_dif}"""
    prompt_2 = """
EXAMPLES
User: What is 5 plus 10?
Output: {"prompt": "What is 5 plus 10?", "name": "fn_add_numbers", "parameters": {"a": 5, "b": 10}}
CURRENT TASK
User: 
"""
    # print(prompt)
    # exit(1)
    prompt = prompt_1 + prompt_2 
    tensor = llm.encode(prompt)
    list_ids: list = tensor.tolist()[0]
    works = os.cpu_count()
    with ThreadPoolExecutor(max_workers=11) as worker:
        result = worker.map(lambda x: generate_json(x,llm,list_ids),prompts)
    pathlib.Path("data/output/").mkdir(parents=True,exist_ok=True)

    for res in result:
        output_list.append(res)
    # print(type(output_list))
    # exit(0)
    with open("data/output/function_calls.json", 'w') as f:
        json.dump(output_list,f,indent=2)
    # print(llm.get_path_to_tokenizer_file())
    # print(llm.get_path_to_vocab_file())
    # s = llm.decode(res)
    # print(s)


main()