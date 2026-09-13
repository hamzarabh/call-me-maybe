***This project has been created as part of the 42 curriculum by hrabh***
## Description:
This project is an introduction to function calling (tool calling) in large language models (LLMs).
The main goal of the project is to guide a small language model to understand a user's prompt, select the most suitable function , and generate a valid json object containing the function name and its required parameters.
The project focuses on using constrained decoding to control the model's token generation and ensure that the generated output follows the required JSON structure and schema.
## Intructions:
This project uses 'uv' for dependency management.
### Installation
Install the project dependencies with:
```
make install
```
### Running
Run the project with:
```
make run
```
### Debugging
Run the project im debug mode with:
```
make debug
```
### Linting
Run the project's linting and type checks (flake8 , mypy) with:
```
make lint
```
## Resources:
uv => https://docs.astral.sh/uv 

FSM => https://www.youtube.com/watch?v=E45v2dD3IQU

Constrained Decoding => https://medium.com/@albersj66/part-6-implementing-constrained-decoding-for-phi-3-vision-2c72a1be6a17

Function calling => https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses

Byte-Pair Encoding (BPE) => https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/
## Algorithm:
To implement Constrained Decoding, I use a Finite State Machine (FSM) to control the token generation process.
At each satate, the FSM determines which tokens are allowed. The LLM then selects the token with the highest logit among the allowed tokens. After generating a token, the FSM transitions to the corresponding next state.
This process continues until the FSM reaches a final state, ensuring that the generated output follows the required JSON structure.

## Design Decisions

***The main design decision was to use a Finite State Machine (FSM) to implement constrained decoding.***

The FSM defines the valid JSON structure and restricts the tokens that the model can generate at each state. This prevents the model from generating tokens that would produce an invalid JSON structure.

I also used a Trie-based structure to efficiently represent the allowed token sequences for JSON keys, function names, and fixed JSON characters.

Function definitions are processed dynamically, so the system is not hardcoded for a specific set of functions. The LLM is responsible for selecting the appropriate function based on the user's prompt.
## Performance analysis:

The main performance bottleneck is the number of calls made to the language model during token generation.

For every generated token, the model produces logits, and the constrained decoder processes these logits to select an allowed token. Generating function parameters, especially strings and numbers, can therefore increase the total execution time.

The project was optimized by using tokenization and decoding caches and by processing independent prompts concurrently using a ThreadPoolExecutor.

The main performance target is to process the test dataset within the required five-minute limit while maintaining valid JSON and schema-compliant outputs.
## Challenges I Faced

One of the main challenges was correctly extracting the function parameters from the model’s output. The model can generate different sequences of tokens for text strings and numbers, making it difficult to ensure that the generated values match the expected parameter types.

Another challenge was performance. The initial implementation took more than five minutes to process the test dataset because the model had to be called repeatedly during the token-by-token generation process.

To address these issues, I used token caching, constrained token selection, and concurrent processing of independent requests, in addition to reducing the configuration in System Prompt.
## Testing strategy:
The testing strategy focuses on both functional correctness and constrained decoding.

The system is tested with different types of function definitions and prompts, including:

- Empty and simple prompts.
- Functions with multiple parameters.
- String and number parameters.
- Large numbers.
- Special characters inside strings.
- Ambiguous prompts requiring the model to select between multiple functions.
- Invalid or unexpected model outputs.

The generated output is checked to ensure that it is valid JSON and contains the required `prompt`, `name`, and `parameters` fields.

The project is also checked with `flake8` and `mypy` to ensure code quality and type correctness.
## Example usage:
The  project can be run using the fllowing command:
```bash
uv run python -m src 
    --functions_definition data/input/functions_definition.json --input data/input/functions_calling_tests.json --output data/output/function_calls.json
```
## AI usage:
***I utilized AI tools to assist with***

- Understanding Finite State Machines (FSM) and Constrained Decoding concepts.
- Brainstorming solution to optimize execution time.
- Resolving Mypy type-hinting errors.
- Help me debug.
