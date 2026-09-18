from smart_llm import Smart_LLM
from multiprocessing.connection import wait
from engine import generate_json

llm = None
def worker_server(server_conns):
    # Client-Server
    global llm
    llm = Smart_LLM()

    run = True
    while run:
        res = []
        print("res1")
        ready_conns = wait(server_conns)
        print("res2")
        for conn in ready_conns:
            try:
                task = conn.recv()
                print(task)
                print("ok")
                if task is None:
                    run = False
                    break
                t_type = task.get("type")
                t_data = task.get("data")
                if t_type == "decode":
                    res = llm.ft_decode(t_data)
                elif t_type == "encode":
                    res = llm.ft_encode(t_data)
                elif t_type == "logits":
                    res = llm.get_logits_from_input_ids(t_data)
                else:
                    res = "Error: Incorrect task type!,(decode, encode, logits)!"
                conn.send(res)
                print("send")
            except Exception as e:
                conn.send(f"Error: {e}")

def ft_worker(p):
    generate_json(p[0], p[1], p[2], p[3])

