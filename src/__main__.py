from llm_sdk import Small_LLM_Model
import json
from json import JSONDecodeError
import numpy as np
from parser import parser
from concurrent.futures import ThreadPoolExecutor
import pathlib
import os
from functools import lru_cache
import argparse
from typing import Any

llm: Small_LLM_Model = Small_LLM_Model()


def check_json(text: str) -> Any | None:
    try:
        j = json.loads(text)
        return (j)
    except JSONDecodeError:
        return None


class Node():
    def __init__(self) -> None:
        self.children: dict = {}
        self.is_end: bool = False


class Tree():
    def __init__(self) -> None:
        self.root: Any = Node()

    def insert(self, tokens: list[int]) -> None:
        node = self.root
        for id in tokens:
            if id not in node.children:
                node.children[id] = Node()
            node = node.children[id]
        node.is_end = True


@lru_cache(maxsize=None)
def smart_decode(ids: tuple) -> str:
    text: str = llm.decode(list(ids))
    return text


@lru_cache(maxsize=None)
def smart_encode(prompt: str) -> tuple:
    res: Any = llm.encode(prompt)
    return tuple(res.tolist()[0])


class JsonFsm():
    def __init__(self, difination: dict, prompt: str) -> None:
        self.jsondif: dict = difination
        self.prompt: str = prompt
        self.e_start: Tree = Tree()
        self.e_key_prompt: Tree = Tree()
        self.e_key_name: Tree = Tree()
        self.e_key_param: Tree = Tree()
        self.e_param_keys: Tree = Tree()
        self.e_colon: Tree = Tree()
        self.e_value_names: Tree = Tree()
        self.e_value_promt: Tree = Tree()
        self.e_value_param: Tree = Tree()
        self.e_coma: Tree = Tree()
        self.e_end: Tree = Tree()

    def init_allowed_tokens(self) -> None:
        self.e_start.insert(self.get_ids("{"))
        parame_tokens = [list(key["parameters"].keys())
                         for key in self.jsondif]
        paratokens = [
            '"'+k+'"' for sublist in parame_tokens for k in sublist]
        tokens = ['"prompt"', '"name"', '"parameters"']
        for pt in paratokens:
            self.e_param_keys.insert(self.get_ids(pt))

        self.e_key_prompt.insert(self.get_ids(tokens[0]))
        self.e_key_name.insert(self.get_ids(tokens[1]))
        self.e_key_param.insert(self.get_ids(tokens[2]))

        self.e_colon.insert(self.get_ids(":"))
        for func in self.jsondif:
            name = func['name']
            self.e_value_names.insert(self.get_ids(f'"{name}"'))
        self.e_value_promt.insert(self.get_ids(json.dumps(self.prompt)))
        self.e_value_param.insert(self.get_ids("{"))

        self.e_coma.insert(self.get_ids(","))
        self.e_end.insert(self.get_ids("}"))

    def get_ids(self, tokens: str) -> list:
        ids_tokens: tuple = smart_encode(tokens)
        return list(ids_tokens)

    def get_tokens(self, state: int) -> tuple[int, Any]:
        if state == 1:
            return (2, self.e_start.root)
        if state == 2:
            return (3, self.e_key_prompt.root)
        if state == 3:
            return (4, self.e_colon.root)
        if state == 4:
            return (5, self.e_value_promt.root)
        if state == 5:
            return (6, self.e_coma.root)
        if state == 6:
            return (7, self.e_key_name.root)
        if state == 7:
            return (8, self.e_colon.root)
        if state == 8:
            return (9, self.e_value_names.root)
        if state == 9:
            return (10, self.e_coma.root)
        if state == 10:
            return (11, self.e_key_param.root)
        if state == 11:
            return (12, self.e_colon.root)
        if state == 12:
            return (0, self.e_start.root)
        else:
            return (0, None)

    def gen_next_token(self, list_data: list, current_node: Node) -> tuple:
        text: str = ""
        while current_node.is_end is False:
            logits: Any = llm.get_logits_from_input_ids(list_data)
            np_logits: Any = np.array(logits)
            allowed: Any = np.fromiter(current_node.children.keys(),
                                       dtype=np.int64)
            next_id: int = allowed[np.argmax(np_logits[allowed])]
            list_data.append(next_id)
            text += smart_decode(tuple([next_id]))
            current_node = current_node.children[next_id]
        return (text, list_data)

    def gen_param(self, list_data: list, name: str, textJson: str) -> str:
        new_token = ""
        parameters = dict()

        for obj in self.jsondif:
            if '"'+obj["name"]+'"' == name:
                parameters = obj["parameters"]
        for i, (k, v) in enumerate(parameters.items()):
            tree_k = Tree()
            tree_k.insert(self.get_ids(f'"{k}"'))
            tok, list_data = self.gen_next_token(list_data, tree_k.root)
            textJson += tok
            tok, list_data = self.gen_next_token(list_data, self.e_colon.root)
            textJson += tok

            if v["type"] == "string":
                tree_cot = Tree()
                tree_cot.insert(self.get_ids('"'))
                tok, list_data = self.gen_next_token(list_data, tree_cot.root)
                textJson += tok
                tok = ""
                for _ in range(100):
                    logits = llm.get_logits_from_input_ids(list_data)
                    np_logits = np.array(logits)
                    next_id = np.argmax(np_logits).item()
                    new_token = smart_decode(tuple([next_id]))
                    for c in new_token:
                        tok += c
                        if c == '"' and textJson[-2] != "\\":
                            break
                    if tok[-1] == '"' and not tok.rstrip().endswith('\\"'):
                        textJson += tok.strip()
                        list_data.append(next_id)
                        break
                    list_data.append(next_id)

                    textJson = textJson.replace('\" \"', '\"')
                    textJson = textJson.replace('\\" \\"', '\\"\\"')
            else:
                for _ in range(100):
                    logits = llm.get_logits_from_input_ids(list_data)
                    np_logits = np.array(logits)
                    next_id = np.argmax(np_logits).item()
                    new_token = smart_decode(tuple([next_id]))
                    if ',' in new_token or '}' in new_token:
                        break
                    list_data.append(next_id)
                    textJson += new_token

            if (i < len(parameters.items()) - 1
                    and (not textJson.rstrip().endswith(','))):
                tok, list_data = self.gen_next_token(list_data, self.e_coma.root)
                textJson += tok
        new_token, list_data = self.gen_next_token(list_data, self.e_end.root)
        textJson += new_token
        textJson = textJson.replace('\" \"', '\"')
        textJson = textJson.replace('\\" \\"', '\\"\\"')
        return textJson

    def generatore(self, list_data: list) -> dict | None:
        json: str = ""
        state: int = 1
        name: str = ""
        while state:
            state, tokens_tree = self.get_tokens(state)
            text, list_data = self.gen_next_token(list_data, tokens_tree)
            if state == 9:
                name = text
            json += text
        json = self.gen_param(list_data, name, json)
        if not json.strip().endswith("}}"):
            text, list_data = self.gen_next_token(list_data, self.e_end.root)
            json += text

        return check_json(json)


def generate_json(prompt: dict, li: list, difination: dict) -> dict | None:
    user_prompt: str = f"""{prompt["prompt"]}
Output:"""
    tensor_data = list(smart_encode(user_prompt))
    list_data = li.copy()
    list_data.extend(tensor_data)
    fsm = JsonFsm(difination, prompt["prompt"])
    fsm.init_allowed_tokens()
    json = fsm.generatore(list_data=list_data)
    if json is None:
        return (None)
    for k in difination:
        if k["name"] == json["name"]:
            for c, v in k["parameters"].items():
                if v["type"] == "number":
                    json["parameters"][c] = float(json["parameters"][c])
    return json


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument('--functions_definition', type=str,
                    default="data/input/functions_definition.json")
    pa.add_argument('--input', type=str,
                    default="data/input/function_calling_tests.json")
    pa.add_argument('--output', type=str,
                    default="data/output/function_calls.json")
    args = pa.parse_args()

    if not args.functions_definition.endswith('.json'):
        print("Error: Functions definition file not json")
        exit(1)
    if not args.input.endswith('.json'):
        print("Error: Input file not json")
        exit(1)
    if not args.output.endswith('.json'):
        print("Error: Output file not json")
        exit(1)

    fun_json, functions_dif, prompts = parser(args.functions_definition,
                                              args.input)
    output_list = []
    sys_prompt = f"""
    {functions_dif}
input:"""
    total_worker: int = 1
    list_ids: list = list(smart_encode(sys_prompt))
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 4:
        total_worker = cpu_count - 4

    with ThreadPoolExecutor(max_workers=total_worker) as worker:
        result: Any = worker.map(lambda p: generate_json(
            p, list_ids, fun_json), prompts)

    for res in result:
        output_list.append(res)
    if os.path.dirname(args.output) != "":
        pathlib.Path(os.path.dirname(args.output)).mkdir(
            parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_list, f, indent=2)

if __name__ == "__main__":
    main()
