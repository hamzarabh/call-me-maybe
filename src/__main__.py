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


class Node():
    def __init__(self):
        self.children = {}
        self.is_end = False

class Tree():
    def __init__(self):
        self.root = Node()
    
    def insert(self, tokens: list):
        node = self.root
        for id in tokens:
            if id not in node.children:
                node.children[id] = Node()
            node = node.children[id]
        node.is_end = True



# n = Tree()
# n.insert([1,55,1])
# n.insert([1,11,1])
# n.insert([2,11,5])

# print(n.root.children.keys())
# n.root = n.root.children[1]
# print(n.root.children.keys())

# n.root = n.root.children[11]
# print(n.root.children.keys())

class JsonFsm():
    def __init__(self, jsondif):
        self.state = "START"  # START , EXPECT_KEY, EXPECT_COLON, EXPECT_VALUE, EXPECT_COMA_END
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.keys_tokens = self.get_ids(['"prompt"'])
        self.jsondif  = jsondif
        self.e_start: Tree = Tree()
        self.e_key: Tree = Tree()
        self.e_colon: Tree = Tree()
        self.e_value: Tree = Tree()
        self.e_coma_end: Tree = Tree()


    def init_allowed_tokens(self):
        self.e_start.insert(self.get_ids("{"))
        self.e_key.insert(self.get_ids('"name"'))
            # parame_tokens = [list(key["parameters"].keys()) for key in self.jsondif]
            # parame_tokens = ['"'+k+'"' for sublist in parame_tokens for k in sublist]
            # tokens = ['"name"',  '"prompt"', '"parameters"']
            # tokens.extend(parame_tokens)
            # return (self.get_ids(tokens))
        self.e_colon.insert(self.get_ids(":"))

        self.e_value.insert(self.get_ids('"fn_greet"'))
        self.e_value.insert(self.get_ids('"fn_reverse_string"'))

        self.e_coma_end.insert(self.get_ids(","))
        self.e_coma_end.insert(self.get_ids("}"))
    def get_ids(self, tokens: str):
        ids_tokens: list = []
        ids_tokens = self.llm.encode(tokens).tolist()[0]
        return ids_tokens
    def update_state(self):
        if self.state == "START":
            self.state = "EXPECT_KEY"
        elif self.state == "EXPECT_KEY":
            self.state = "EXPECT_COLON"
        elif self.state == "EXPECT_COLON":
            self.state = "EXPECT_VALUE"
        elif self.state == "EXPECT_VALUE":
            self.state = "EXPECT_COMA_END"
        elif self.state == "EXPECT_COMA_END":
            self.state = "EXPECT_KEY"
 
    def masked_logits(self, list_data: list):
        json_response = dict()
        text = ""
        for _ in range(100):
            if self.state == "START":
                k = self.e_start
            if self.state == "EXPECT_KEY":
                k = self.e_key
            if self.state == "EXPECT_COLON":
                k = self.e_colon
            if self.state == "EXPECT_VALUE":
                k = self.e_value
            if self.state == "EXPECT_COMA_END":
                k = self.e_coma_en          
            while k.root.is_end is False:
                logits = self.llm.get_logits_from_input_ids(list_data)
                np_logits = np.array(logits)
                np_copy = np_logits.copy()
                np_copy[:] = float('-infi')
                token = k.root.children.keys()
                for tok in token:
                    np_copy[tok] = np_logits[tok]
                next_id = np.argmax(np_copy).item()
                k.root = k.root.children[next_id]
                list_data.append(next_id)

                if mycash.get((next_id)) is None:
                        mycash[(next_id)] = self.llm.decode([next_id])
                text += mycash[(next_id)]
                # print(text)
                # print(text)
            self.update_state()
            if text.strip().endswith("}"):
                json_response = check_json(text.strip())
                if json_response:
                    break
            
        
        

def generate_json(prompt, llm, li: list):
    response = []
    text = ""
    prompt_3 = f"""{prompt["prompt"]}
Output:"""
    tensor_data = llm.encode(prompt_3)
    list_data = li.copy()
    list_data.extend(tensor_data.tolist()[0])
    fsm = JsonFsm(None)
    fsm.init_allowed_tokens()
    # for _ in range(100):
    #     logits = llm.get_logits_from_input_ids(list_data)
        # copy = [float('-inf') for _ in range(len(logits))]
        # for i in [9000, 15, 42]:
        #     copy[i] = logits[i]
        # next_id = copy.index(max(copy))
        # np_logits = np.array(logits)
        # next_id = np.argmax(np_logits).item()
        # list_data.append(next_id)
        # response.append(next_id)
        # if mycash.get((next_id)) is None:
        #     mycash[(next_id)] = llm.decode([next_id])
        # text += mycash[(next_id)]
        # print(text)
        # print(text)
        # if text.strip().endswith("}"):
        #     json_response = check_json(text.strip())
        #     if json_response:
        #         break
    return fsm.masked_logits(list_data=list_data)

def main():
    fun_json, functions_dif, prompts = parser()
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