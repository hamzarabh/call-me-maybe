from llm_sdk import Small_LLM_Model


def main():
    llm = Small_LLM_Model()
    tensor_data = llm.encode("hello")
    list_data = tensor_data.tolist()
    res = llm.get_logits_from_input_ids(list_data[0])
    print(res)
    # s = llm.decode(res)
    # print(s)
