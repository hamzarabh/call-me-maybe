from llm_sdk import Small_LLM_Model
import json
from parser import parser
import pathlib
from functools import lru_cache
lru_cache()
def check_json(text):
    try:
      j = json.loads(text)
      return (j)
    except:
        return None
    
    
def main():
    functions_dif, prompts = parser()
    llm = Small_LLM_Model()
    pathlib.Path("data/output/").mkdir(parents=True,exist_ok=True)
    output_list = []
    prompt_1 = f"""
AVAILABLE FUNCTIONS
    {functions_dif}"""
    prompt_2 = """
EXAMPLES
User: What is 5 plus 10?
Output: {"prompt": "What is 5 plus 10?", "name": "fn_add_numbers", "parameters": {"a": 5, "b": 10}}
CURRENT TASK
"""
    # print(prompt)
    # exit(1)
    prompt = prompt_1 + prompt_2 
    tensor = llm.encode(prompt)
    list_ids: list = tensor.tolist()[0]
    for p in prompts:
        response = []
        list_data = list_ids.copy()
        prompt_3 = f"""
User: {p["prompt"]}
Output:"""
        tensor_data = llm.encode(prompt_3)
        list_data.extend(tensor_data.tolist()[0])
        for _ in range(80):
            logits = llm.get_logits_from_input_ids(list_data)
            # copy = [float('-inf') for _ in range(len(logits))]
            # for i in [9000, 15, 42]:
            #     copy[i] = logits[i]
            # next_id = copy.index(max(copy))

            next_id = logits.index(max(logits))
            list_data.append(next_id)
            response.append(next_id)
            json_response = check_json(llm.decode(response))
            if json_response:
                break
        print(json_response)
        output_list.append(json_response)
    with open("data/output/function_calls.json", 'w') as f:
        json.dump(output_list,f,indent=2)
    # print(llm.get_path_to_tokenizer_file())
    # print(llm.get_path_to_vocab_file())
    # s = llm.decode(res)
    # print(s)


main()
