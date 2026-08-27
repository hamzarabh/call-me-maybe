from llm_sdk import Small_LLM_Model

def const_decoding(llm: Small_LLM_Model, prompt: str, sys_prompt_ids: list[int]):

    res: list[int] = []

    s1 = '{\n   "prompt": "' + prompt + ',\n"name": "'
    s1_ids = llm.encode(s1)[0].tolist()
    sys_prompt_ids += s1_ids
    res += s1_ids

    func_name: list[int] = get_func_name()
    sys_prompt_ids += func_name

    s2 = '",\n"parameters": '
    s2_ids: list[int] = llm.encode(s2)[0].tolist()
    sys_prompt_ids += s2_ids
    res += s2_ids

    parametres_ids: list[int] = get_parametres()
    sys_prompt_ids += parametres_ids

    s3 = '\n}'
    s3_ids: list[int] = llm.encode(s3)[0].tolist()
    res += s2_ids

    res += s3_ids

    return llm.decode(res)

