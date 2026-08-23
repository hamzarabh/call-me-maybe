from llm_sdk import Small_LLM_Model
import json

def check_json(text):
    try:
      json.loads(text)
      return (1)
    except:
        return (0)
    
    
def main():
    prombt = """[SYSTEM INSTRUCTION]
You are a strict JSON-only Function Calling processor.
Your ONLY task is to convert the user prompt into a valid JSON object.

[RULES]
1. Output MUST be valid JSON and nothing else.
2. Do NOT write any introduction, explanation, or markdown formatting (no ```json).
3. Do NOT add any extra fields outside: "prompt", "name", "parameters".
4. Use exact parameter names and types as defined.

[AVAILABLE FUNCTIONS]
- Name: fn_add_numbers
  Description: Add two numbers together.
  Parameters:
    - a (type: number)
    - b (type: number)
- Name: fn_greet
  Description: Generate a greeting message for a person by name..
  Parameters:
    - name (type: string)
- Name: fn_reverse_string
  Description: Reverse a string and return the reversed result.
  Parameters:
    - s (type: string)

[EXAMPLES]
User: What is 5 plus 10?
Output: {"prompt": "What is 5 plus 10?", "name": "fn_add_numbers", "parameters": {"a": 5, "b": 10}}

[CURRENT TASK]
User: What is the sum of 265 and 345?
Output:"""
    llm = Small_LLM_Model()
    tensor_data = llm.encode(prombt)
    list_data = tensor_data.tolist()[0]
    res = []
    for _ in range(80):
        logits = llm.get_logits_from_input_ids(list_data)
        # copy = [float('-inf') for _ in range(len(logits))]
        # for i in [9000, 15, 42]:
        #     copy[i] = logits[i]
        # next_id = copy.index(max(copy))

        next_id = logits.index(max(logits))
        list_data.append(next_id)
        res.append(next_id)
        if check_json(llm.decode(res)) == 1:
            break
        
    print(llm.decode(res))
    # print(llm.get_path_to_tokenizer_file())
    # print(llm.get_path_to_vocab_file())
    # s = llm.decode(res)
    # print(s)


main()
