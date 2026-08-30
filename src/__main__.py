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
      j = json.loads(text)
      return (j)
    except:
        return None

mycash = dict()
llm: Small_LLM_Model = Small_LLM_Model()

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
    def __init__(self, jsons, prompt):
        self.state = "START"  # START , EXPECT_KEY, EXPECT_COLON, EXPECT_VALUE, EXPECT_COMA_END
        self.sub_state = 0
        self.keys_tokens = self.get_ids(['"prompt"'])
        self.jsondif  = jsons[0]
        self.prompt  = prompt
        self.e_start: Tree = Tree()
        self.e_key_prompt: Tree = Tree()
        self.e_key_name: Tree = Tree()
        self.e_key_param: Tree = Tree()
        self.e_param_keys: Tree = Tree()
        self.e_colon: Tree = Tree()
        self.e_value_names: Tree = Tree()
        self.e_value_promt: Tree = Tree()
        self.e_value_param: Tree = Tree() #
        self.e_coma_end: Tree = Tree()
        self.e_end: Tree = Tree()


    def init_allowed_tokens(self):
        self.e_start.insert(self.get_ids("{"))
        # self.e_key.insert(self.get_ids('"name"'))
        # self.e_key.insert(self.get_ids('"prompt"'))
        # self.e_key.insert(self.get_ids('"parameters"'))
        parame_tokens = [list(key["parameters"].keys()) for key in self.jsondif]
        parame_tokens = ['"'+k+'"' for sublist in parame_tokens for k in sublist]
        tokens = ['"prompt"', '"name"', '"parameters"']
        #tokens.extend(parame_tokens)
        for pt in parame_tokens:
            self.e_param_keys.insert(self.get_ids(pt))

        self.e_key_prompt.insert(self.get_ids(tokens[0]))
        self.e_key_name.insert(self.get_ids(tokens[1]))
        self.e_key_param.insert(self.get_ids(tokens[2]))

        self.e_colon.insert(self.get_ids(":"))
        for func in self.jsondif:
            self.e_value_names.insert(self.get_ids(f'"{func['name']}"'))
        self.e_value_promt.insert(self.get_ids(f'"{self.prompt}"'))
        self.e_value_param.insert(self.get_ids("{"))


        self.e_coma_end.insert(self.get_ids(","))
        #self.e_coma_end.insert(self.get_ids("}"))
    def get_ids(self, tokens: str):
        ids_tokens: list = []
        ids_tokens = llm.encode(tokens).tolist()[0]
        return ids_tokens
    def update_state(self):
        if self.state == "START":
            self.state = "EXPECT_KEY"
        elif self.state == "EXPECT_KEY":
            self.state = "EXPECT_COLON"
        elif self.state == "EXPECT_COLON" and self.sub_state == 3:
            self.state = "START"
        elif self.state == "EXPECT_COLON":
            self.state = "EXPECT_VALUE"
        elif self.state == "EXPECT_VALUE":
            self.state = "EXPECT_COMA_END"
        elif self.state == "EXPECT_VALUE":
                self.state = "EXPECT_COMA_END"
        elif self.state == "EXPECT_COMA_END":
            self.state = "EXPECT_KEY"

 
    def masked_logits(self, list_data: list):
        json_response = dict()
        text = ""
        for _ in range(600):
            if self.state == "START":
                k = self.e_start
            elif self.state == "EXPECT_KEY" and self.sub_state == 0:
                k = self.e_key_prompt
            elif self.state == "EXPECT_KEY" and self.sub_state == 1:
                k = self.e_key_name
            elif self.state == "EXPECT_KEY" and self.sub_state == 2:
                k = self.e_key_param
                self.sub_state += 1
            elif self.state == "EXPECT_KEY" and self.sub_state == 3:
                k = self.e_param_keys
                self.sub_state += 1
                # print("teeee:")
                # print(k.root.children.keys())
            elif self.state == "EXPECT_COLON" and self.sub_state > 3:
                k = self.e_colon
                self.sub_state = 4
            elif self.state == "EXPECT_COLON":
                k = self.e_colon
            elif self.state == "EXPECT_VALUE" and self.sub_state == 0:
                k = self.e_value_promt
                self.sub_state += 1
            elif self.state == "EXPECT_VALUE" and self.sub_state == 1:
                k = self.e_value_names
                self.sub_state += 1
            elif self.state == "EXPECT_VALUE" and self.sub_state == 2:
                k = self.e_value_param
                self.sub_state += 1
            elif self.state == "EXPECT_VALUE" and self.sub_state >= 4:
                k = Tree()
            elif self.state == "EXPECT_COMA_END":
                k = self.e_coma_end
                
            current_node = k.root
            while current_node.is_end is False:
                # print(self.sub_state)
                # print(self.state)
                logits = llm.get_logits_from_input_ids(list_data)
                np_logits = np.array(logits)
                np_copy = np_logits.copy()
                np_copy[:] = -np.inf
                token = current_node.children.keys()
                for tok in token:
                    np_copy[tok] = np_logits[tok]
                # if k:
                next_id = np.argmax(np_copy).item()

                if not current_node.children.keys():
                    next_id = np.argmax(np_logits).item()
                elif next_id not in current_node.children.keys():
                    print("Error")
                    exit(127)
                if current_node.children.keys():
                    current_node = current_node.children[next_id]
                list_data.append(next_id)

                if mycash.get((next_id)) is None:
                        mycash[(next_id)] = llm.decode([next_id])

                text += mycash[(next_id)]
                if not current_node.children.keys() and "}" in mycash[(next_id)]:
                    break
                # print(text)
                # print(text)
            self.update_state()
            if text.strip().endswith("}"):
                json_response = check_json(text.strip())
                if json_response:
                    print("break")
                    break
        print("json ===>")
        print(json_response)
        return json_response
            
        
        

def generate_json(prompt, llm, li: list, jsons):
    response = []
    text = ""
    prompt_3 = f"""{prompt["prompt"]}
Output:"""
    tensor_data = llm.encode(prompt_3)
    list_data = li.copy()
    list_data.extend(tensor_data.tolist()[0])
    fsm = JsonFsm(jsons, prompt["prompt"])
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
    output_list = []
    prompt_1 = f"""
RULES
1. Output MUST be valid JSON and nothing else.
2. Do NOT write any introduction, explanation, or markdown formatting (no ```json).
3. Do NOT add any extra fields outside: "prompt", "name", "parameters".
4. Use exact parameter names and types as defined.
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
    with ThreadPoolExecutor(max_workers=len(prompts)) as worker:
        result = worker.map(lambda x: generate_json(x,llm,list_ids, (fun_json,prompts)),prompts)
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