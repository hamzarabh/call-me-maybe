import json
from json import JSONDecodeError
from typing import  Literal, Any
from pydantic import BaseModel, ValidationError


class ParametersModel(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]

class   FunDefModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametersModel]
    returns: ParametersModel

class PromptModel(BaseModel):
    prompt: str

def parser() -> tuple[str, list] | None:
    try:
        with open("data/input/functions_definition.json") as f:
            try:
                data_loaded: Any = json.load(f)
                function: str = ""
                if isinstance(data_loaded, list) == False:
                    data: list = [data_loaded]
                else:
                    data: list = data_loaded
                for item in data:
                    parser = FunDefModel(**item)
                    function += f"name: {str(item["name"])}, description: {str(item["description"])}, parameters: {str(item["parameters"])}" + "\n"
            except (JSONDecodeError ) as e:
                print(f"json forma not valide in 'functions_definition.json': {e}")
                return None
            except ValidationError as e:
                print(f"Json schema not valide in 'functions_definition.json': {e}")
                return None

        with open("data/input/function_calling_tests.json") as f:
            try:
                data = json.load(f)
                if isinstance(data, list) == False:
                    data = [data]

                for item in data:
                    prompt = PromptModel(**item)

            except (JSONDecodeError ) as e:
                    print(f"json forma not valide in 'function_calling_tests.json': {e}")
                    return None
            except ValidationError as e:
                print(f"Json schema not valide in 'function_calling_tests.json': {e}")
                return None
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
        return None
    return (data_loaded, function,data)
parser()