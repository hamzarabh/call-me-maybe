import transformers
from outlines import Generator, from_transformers
from pydantic import BaseModel


class Answer(BaseModel):
    name: str
    age: int


model_name = "Qwen/Qwen3-0.6B"

hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

model = from_transformers(
    hf_model,
    tokenizer,
)

generator = Generator(model, Answer)

result = generator(
    "My name is Hamza and I am 26 years old.",
    max_new_tokens=50,
)

print(result)

answer = Answer.model_validate_json(result)

print(answer)
print(answer.name)
print(answer.age)
