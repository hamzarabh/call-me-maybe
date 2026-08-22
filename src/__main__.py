from llm_sdk import Small_LLM_Model


def main():
    llm = Small_LLM_Model()
    tensor_data = llm.encode("hello can you help me ")
    list_data = tensor_data.tolist()[0]
    for _ in range(80):
        logits = llm.get_logits_from_input_ids(list_data)
        copy = [float('-inf') for _ in range(len(logits))]
        for i in [9000, 15, 42]:
            copy[i] = logits[i]
        next_id = copy.index(max(copy))
        list_data.append(next_id)
        
    print(llm.decode(list_data))
    #print(llm.get_path_to_tokenizer_file())
    # s = llm.decode(res)
    # print(s)


main()
