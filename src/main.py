from llm_sdk import Small_LLM_Model
import json
import numpy as np
from parser import parser
from lark import Lark
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pathlib
import os
from functools import lru_cache
from typing import Any
def check_json(text):
    try:
      j = json.loads(text)
      return (j)
    except:
        return None


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

# grammar = """
#  object: dict

#     dict : LBRACE [pair (COMMA pair)*] RBRACE
#     pair : ESCAPED_STRING COLON value
#     value: dict | ESCAPED_STRING | SIGNED_NUMBER

#     LBRACE: "{"
#     RBRACE: "}"
#     COLON: ":"
#     COMMA: ","
#     %import common.ESCAPED_STRING
#     %import common.SIGNED_NUMBER
#     %import common.WS
#     %ignore WS
# """
name_grammar = r"""
        start: NAME COLON ESCAPED_STRING
        NAME: "name"
        COLON: ":"

        %import common.ESCAPED_STRING
        %import common.WS
        %ignore WS
"""
prompt_grammar = r"""
        start: PROMPT COLON ESCAPED_STRING
        PROMPT: "prompt"
        COLON: ":"

        %import common.ESCAPED_STRING
        %import common.WS
        %ignore WS
"""


llm: Small_LLM_Model = Small_LLM_Model()
@lru_cache(maxsize=None)
def smart_decode(ids: tuple):
    text = llm.decode(list(ids))
    return text

@lru_cache(maxsize=None)
def smart_encode(prompt :str):
    res = llm.encode(prompt)
    return tuple(res.tolist()[0])

class JsonFsm():
    def __init__(self, jsons, prompt):
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
        self.e_coma: Tree = Tree()
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

        # print(self.e_param_keys.root.children.keys())
        # exit(0)

        self.e_key_prompt.insert(self.get_ids(tokens[0]))
        self.e_key_name.insert(self.get_ids(tokens[1]))
        self.e_key_param.insert(self.get_ids(tokens[2]))

        self.e_colon.insert(self.get_ids(":"))
        for func in self.jsondif:
            self.e_value_names.insert(self.get_ids(f'"{func['name']}"'))
        # print(self.prompt)
        self.e_value_promt.insert(self.get_ids(json.dumps(self.prompt)))
        self.e_value_param.insert(self.get_ids("{"))


        self.e_coma.insert(self.get_ids(","))
        self.e_end.insert(self.get_ids("}"))
    def get_ids(self, tokens: str):
        ids_tokens: list = []
        ids_tokens = smart_encode(tokens)
        return list(ids_tokens)

    # def get_name_tokens(self, token):
        # if token  == "1":
        #     return ("name",self.e_key_name.root)
        # lark_parser = Lark(name_grammar, start="start", parser="lalr")
        # interactive = lark_parser.parse_interactive()
        # for tok in lark_parser.lex(token):
        #     interactive.feed_token(tok)
        # token_allowed = interactive.accepts()
        # for t in token_allowed:
        #     if t == "NAME":
        #         return ("name",self.e_key_name.root)
        #     if t == "COLON":
        #         return ("name:",self.e_colon.root)
        #     if t == "ESCAPED_STRING":
        #         return (None,  self.e_value_names.root)

    def get_tokens(self, token):
        if token == 1:
             return (2,self.e_start.root)
        if token == 2:
                return (3, self.e_key_prompt.root )
        if token == 3:
                return (4,self.e_colon.root)
        if token == 4:
                return (5, self.e_value_promt.root)
        if token == 5:
             return (6,self.e_coma.root)
        if token == 6:
                return (7, self.e_key_name.root)
        if token == 7:
                return (8,self.e_colon.root)
        if token == 8:
                return (9, self.e_value_names.root)
        if token == 9:
             return (10, self.e_coma.root)
        if token == 10:
             return (11, self.e_key_param.root)
        if token == 11:
             return (12, self.e_colon.root)
        if token == 12:
            return (0, self.e_start.root)
        # if token == 13:
        #      return (14, self.e_param_keys.root)
        # if token == 14:
        #      return (0, self.e_colon.root)
            
    def gen_next_token(self,list_data, current_node: Tree):
        text: str = ""
        while current_node.is_end is False:
                logits = llm.get_logits_from_input_ids(list_data)
                np_logits = np.array(logits)
                allowed = np.fromiter(current_node.children.keys(), dtype=np.int64)
                # np_copy = np_logits.copy()
                # np_copy[:] = -np.inf
                # token = current_node.children.keys()
                # for tok in token:
                #     np_copy[tok] = np_logits[tok]
                next_id = allowed[np.argmax(np_logits[allowed])]
                list_data.append(next_id)
                text += smart_decode(tuple([next_id]))
                current_node = current_node.children[next_id]
                # print(smart_decode(tuple([next_id])))
        return (text, list_data)

    def gen_param(self,list_data, name, textJson):
        text = ""
        parameters = dict()
        # print("hhhhh",  self.jsondif)
        for obj in self.jsondif:
            if '"'+obj["name"]+'"' == name:
                parameters = obj["parameters"]
                # print(name)
        i = 1
        for i,(k, v) in enumerate(parameters.items()):
            tree_k = Tree()
            tree_k.insert(self.get_ids(f'"{k}"'))
            t, list_data = self.gen_next_token(list_data, tree_k.root)
            textJson += t
            # print(textJson)
            t, list_data = self.gen_next_token(list_data, self.e_colon.root)
            textJson += t

            if v["type"] == "string":
                    tree_cot = Tree()
                    tree_cot.insert(self.get_ids('"'))
                    t, list_data = self.gen_next_token(list_data, tree_cot.root)
                    textJson += t
                    for _ in range(100):
                        # print(textJson)
                        logits = llm.get_logits_from_input_ids(list_data)
                        np_logits = np.array(logits)
                        next_id = np.argmax(np_logits).item()
                        text = smart_decode(tuple([next_id]))
                        # print(f"({text})")
                        if '"' in text:
                            for c in text:
                                textJson += c
                                if c == '"' :
                                    break
                            if textJson[-1] == '"':
                                break
                        list_data.append(next_id)
                        textJson += text
            else:
                for _ in range(100):
                    logits = llm.get_logits_from_input_ids(list_data)
                    np_logits = np.array(logits)
                    next_id = np.argmax(np_logits).item()
                    text = smart_decode(tuple([next_id]))
                    if ',' in text or '}' in text:
                        break
                    list_data.append(next_id)
                    textJson += text
                    # print(textJson)
            # print(f"({text})")


        
            if i < len(parameters.items()) - 1 and (not textJson.rstrip().endswith(',')):
                    t, list_data = self.gen_next_token(list_data, self.e_coma.root)
                    textJson += t
        text, list_data = self.gen_next_token(list_data, self.e_end.root)
        textJson += text
        textJson = textJson.replace('" ', '"')
        textJson = textJson.replace(' "', '"')
        return textJson
                 
            
     

         
         

    def masked_logits(self, list_data):
        json = ""
        start = 1
        name = ""
        while start:
            start , tokens_tree = self.get_tokens(start)
            text, list_data = self.gen_next_token(list_data, tokens_tree)
            if start == 9:
                name = text
            json += text
        # print("-----------------------")
        json = self.gen_param(list_data, name, json)
        if not json.strip().endswith("}}"):
            text, list_data = self.gen_next_token(list_data, self.e_end.root)
            json += text
        # print(f"=> {json}")

        # print("=>",check_json(json))
        return check_json(json)






def generate_json(prompt, li: list, jsons):
    response = []
    text = ""
    prompt_3 = f"""{prompt["prompt"]}
Output:"""
    tensor_data = list(smart_encode(prompt_3))
    list_data = li.copy()
    list_data.extend(tensor_data)
    fsm = JsonFsm(jsons, prompt["prompt"])
    fsm.init_allowed_tokens()
    return fsm.masked_logits(list_data=list_data)

def main():
    fun_json, functions_dif, prompts = parser()
    output_list = []
    prompt_1 = f"""
FUNCTIONS
    {functions_dif}"""
    prompt_2 = """
EXAMPLES
User: What is 5 plus 10?
Output: {"prompt": "What is 5 plus 10?", "name": "fn_add_numbers", "parameters": {"a": 5.0, "b": 10.0}}
Very important :
Check all the definitions before doing anything.
Choose the appropriate function name for the question.
CURRENT TASK
User: 
"""
    # print(prompt_1)
    # exit(1)
    prompt = prompt_1 + prompt_2 
    tensor = list(smart_encode(prompt))
    list_ids: list = tensor
    works = os.cpu_count()
    with ThreadPoolExecutor(max_workers=len(prompts)) as worker:
        result = worker.map(lambda x: generate_json(x,list_ids, (fun_json,prompts)),prompts)
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