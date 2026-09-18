from smart_llm import Smart_LLM
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
import multiprocessing as mp
from multiprocessing import Pipe
# llm: Smart_LLM = None






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


# @lru_cache(maxsize=None)
# def smart_decode(ids: tuple) -> str:
#     text: str = llm.ft_decode(list(ids))
#     return text


# @lru_cache(maxsize=None)
# def smart_encode(prompt: str) -> tuple:
#     res: Any = llm.ft_encode(prompt)
#     return tuple(res.tolist()[0])


class JsonFsm():
    def __init__(self, difination: dict, prompt: str, pipe) -> None:
        self.jsondif: dict = difination
        self.prompt: str = prompt
        self.pipe = pipe
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
        ids_tokens = self.request("encode",tokens)
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

    def request(self, type:str, data: Any):
        self.pipe.send({
            "type": type,
            "data": data
        })
        response = self.pipe.recv()
        return response

    def gen_next_token(self, list_data: list, current_node: Node) -> tuple:
        text: str = ""
        while current_node.is_end is False:
            logits: Any = self.request("logits",list_data)
            np_logits: Any = np.array(logits)
            allowed: Any = np.fromiter(current_node.children.keys(),
                                       dtype=np.int64)
            next_id: int = allowed[np.argmax(np_logits[allowed])]
            list_data.append(next_id)

            text += self.request("decode", tuple([next_id]))
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
                    logits = self.request("logits",list_data)
                    np_logits = np.array(logits)
                    next_id = np.argmax(np_logits).item()
                    new_token = self.request("decode", tuple([next_id]))
                    for c in new_token:
                        tok += c
                        if c == '"' and tok[-2] != "\\":
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
                    logits = self.request("logits",list_data)
                    np_logits = np.array(logits)
                    next_id = np.argmax(np_logits).item()
                    new_token = self.request("decode", tuple([next_id]))
                    if ',' in new_token or '}' in new_token:
                        break
                    list_data.append(next_id)
                    textJson += new_token

            if (i < len(parameters.items()) - 1
                    and (not textJson.rstrip().endswith(','))):
                tok, list_data = self.gen_next_token(list_data,
                                                     self.e_coma.root)
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
            print(json)
        json = self.gen_param(list_data, name, json)
        if not json.strip().endswith("}}"):
            text, list_data = self.gen_next_token(list_data, self.e_end.root)
            json += text
        return check_json(json)


def generate_json(prompt: dict, li: list, difination: dict, pipe) -> dict | None:
    print("teees")
    user_prompt: str = f"""{prompt["prompt"]}
Output:"""

    pipe.send({
        "type": "encode",
        "data": user_prompt
        })
    print("gen send")
    data_ids = pipe.recv()
    print("res")
    print(data_ids)
    list_data = li.copy()
    list_data.extend(data_ids)
    fsm = JsonFsm(difination, prompt["prompt"], pipe)
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
