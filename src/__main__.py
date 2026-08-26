from llm_sdk import Small_LLM_Model
import json
from parser import parser
import pathlib
def check_json(text):
    try:
      j = json.loads(text)
      return (j)
    except:
        return None
    
    
def main():
    functions_dif, prompts = parser()
    llm = Small_LLM_Model()
    output_list = []
    prompt_1 = f"""[SYSTEM INSTRUCTION]
You are a strict JSON-only Function Calling processor.
Your ONLY task is to convert the user prompt into a valid JSON object.
[RULES]
1. Output MUST be valid JSON and nothing else.
2. Do NOT write any introduction, explanation, or markdown formatting (no ```json).
3. Do NOT add any extra fields outside: "prompt", "name", "parameters".
4. Use exact parameter names and types as defined.
[AVAILABLE FUNCTIONS]
    {functions_dif}"""
    prompt_2 = """
[EXAMPLES]
User: What is 5 plus 10?
Output: {"prompt": "What is 5 plus 10?", "name": "fn_add_numbers", "parameters": {"a": 5, "b": 10}}
"""
    # print(prompt)
    # exit(1)
    for p in prompts:
        response = []
        prompt_3 = f"""
[CURRENT TASK]
User: {p["prompt"]}
Output:"""
        prompt = prompt_1 + prompt_2 + prompt_3
        tensor_data = llm.encode(prompt)
        list_data = tensor_data.tolist()[0]
        pathlib.Path("data/output/").mkdir(parents=True,exist_ok=True)
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
        output_list.append(json_response)
    with open("data/output/function_calls.json", 'w') as f:
        json.dump(output_list,f,indent=2)
    # print(llm.get_path_to_tokenizer_file())
    # print(llm.get_path_to_vocab_file())
    # s = llm.decode(res)
    # print(s)


main()
