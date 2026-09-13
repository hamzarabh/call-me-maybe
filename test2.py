from collections import defaultdict
import json
corpus = ["hug", "pug", "pun"]

res = []


with open("data.txt") as data:
    test = data.read()

def gen_vocab_freqs(text: str):
    vocab_freqs = defaultdict(int)
    for word in text.split():
        wor = ' '.join(["Ġ"]+ list(word))
        vocab_freqs[wor] += 1
    return vocab_freqs


def get_stats(vocab_freqs):

    pair_freqs = defaultdict(int)
    for word , freq in vocab_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) -1):
            pair = (symbols[i] , symbols[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def ft_merger(pair_freqs, vocab_freq, merges_list):
    c1, c2 = max(dict(pair_freqs), key=pair_freqs.get)
    token = c1 + c2
    merges_list.append([c1,c2])
    new_vocab_freq = defaultdict()
    for word, freq in vocab_freq.items():
        new_word: str = word.replace(f"{c1} {c2}", token)
        new_vocab_freq[new_word] = freq
    with open('merge.txt', 'a') as f:
        f.write(token+'\n')
    return (new_vocab_freq, merges_list)

ids = []
# vocab_freq = gen_vocab_freqs(test)
# merges_list = []
# all_tokens: list = list()
# for i in range(1500):
#     for word in vocab_freq.keys():
#         for symbol in word.split():
#             if symbol not in all_tokens:
#                 all_tokens.append(symbol)
#     pair_freqs = get_stats(vocab_freq)
#     if not pair_freqs:
#         break
#     vocab_freq, merges_list = ft_merger(pair_freqs, vocab_freq, merges_list)

# vocab_dict = {token:idx for idx, token in enumerate(all_tokens)}
# tokenazer_data = {
#     "vocab": vocab_dict,
#     "merges":merges_list
# }
# with open('tokenizer.json', 'w', encoding='utf-8') as f:
#     json.dump(tokenazer_data, f, ensure_ascii=False, indent=4)




def ft_encode(text: str):
    ids_list = []
    text = text.replace(" ", "Ġ")
    # text = get_stats(text)
    text = list(text)
    print(text)

    # text.insert(0,"Ġ")
    # symbols_list = [list(text.keys())]
    with open('tokenizer.json') as f:
        tokenizer_file = json.load(f)
        for a, b in tokenizer_file["merges"]:
            new = f"{a}{b}"
            for i in range(len(text)-2):
                # if i >= len(text)-1:
                #     break
                if a == text[i] and b == text[i+1]:
                    text[i] = new
                    del text[i+1]
        for token in text:
            if token in tokenizer_file["vocab"]:
                ids_list.append(tokenizer_file["vocab"][token])
            else:
                continue

        return (ids_list)


print(ft_encode("hello good"))







# print(dict(pair_freqs))
# print(vocab_freq)



# vocab = [k for k, v in pair_freqs.items() if v > 1]



# def bpe():



# hug pug  pun

# h
# u
# g

# p
# u
# g

# p
# u
# n

# hu
# ug

# pu
# ug

# pu
# un
# ***