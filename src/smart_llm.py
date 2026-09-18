from llm_sdk import Small_LLM_Model
import json
from functools import lru_cache
import multiprocessing as mp

class Smart_LLM(Small_LLM_Model):
    def __init__(self):
        super().__init__()
        self.vocab = self.get_vocab()
        self.tokenizer = self.get_tokenizer()

    def get_vocab(self) -> dict:
        with open(self.get_path_to_vocab_file()) as f :
            vocab = json.load(f)
            vocab = dict(zip(vocab.values(), vocab.keys()))
        return vocab
    def get_tokenizer(self):
        with open(self.get_path_to_tokenizer_file()) as f:
            tokenizer = json.load(f)
            return tokenizer.get("model")

    @lru_cache(maxsize=None)
    def ft_decode(self,list_ids: tuple) -> str:
        text_result: str = ""
        for id in list_ids:
            text_result += self.vocab[id]
        text_result = text_result.replace('Ġ',' ')
        return text_result

    @lru_cache(maxsize=None)
    def ft_encode(self, text: str):
        ids: list = []
        text_list = list(text.replace(" ", 'Ġ'))
        for a, b in self.tokenizer.get("merges"):
            for i in range(len(text_list)-1):
                if i >= len(text_list)-1:
                    break
                if a == text_list[i] and b == text_list[i+1]:
                    text_list[i] = f"{a}{b}"
                    del text_list[i+1]
                    

        for token in text_list:
            if token in self.tokenizer.get("vocab"):
                ids.append(self.tokenizer.get("vocab")[token])
            else:
                continue
        return ids


