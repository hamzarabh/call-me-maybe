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

from worker import worker_server, ft_worker
# llm: Smart_LLM = None










def main() -> None:
    total_worker = 5
    server_conns = []
    worker_conns = []

    for _ in range(total_worker):
        s_conn, w_conn = mp.Pipe()
        server_conns.append(s_conn)
        worker_conns.append(w_conn)
    s_conn , w_conn = Pipe()
    server_conns.append(s_conn)
    llm_server = mp.Process(target=worker_server, args=(server_conns,))
    llm_server.start()
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

    w_conn.send({
        "type": "encode",
        "data": sys_prompt
        })
    list_ids: list = w_conn.recv()
    print(list_ids)
    # total_worker: int = 1
    # cpu_count = os.cpu_count()
    # if cpu_count is not None and cpu_count > 4:
    #     total_worker = cpu_count - 4



    tast_args = []
    for i, p in enumerate(prompts):
        tast_args.append((p, list_ids, fun_json, worker_conns[i % total_worker]))
    with mp.Pool(processes=total_worker) as worker:
        result: Any = worker.map(ft_worker, tast_args)
    
    w_conn.send(None)
    print("jou1")
    llm_server.join()
    print("jou2")
    for res in result:
        output_list.append(res)
    if os.path.dirname(args.output) != "":
        pathlib.Path(os.path.dirname(args.output)).mkdir(
            parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_list, f, indent=2)


if __name__ == "__main__":
    main()
