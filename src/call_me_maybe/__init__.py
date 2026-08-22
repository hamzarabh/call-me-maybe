from llm_sdk import Small_LLM_Model


def main():
    llm = Small_LLM_Model()
    tensor_data = llm.encode("hello")
    list_data = tensor_data.tolist()
    res = llm.get_logits_from_input_ids(list_data[0])
    print(max(res))
    print("encode: ",list_data)
    print(res.index(max(res)))
    print(llm.get_path_to_tokenizer_file())
    # s = llm.decode(res)
    # print(s)
