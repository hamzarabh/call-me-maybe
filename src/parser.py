import json
from json import JSONDecodeError
from typing import Literal, Any
from pydantic import BaseModel, ValidationError


class ParametersModel(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "array", "object"]


class FunDefModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametersModel]
    returns: ParametersModel


class PromptModel(BaseModel):
    prompt: str


def parser(diffile: str, promfile: str) -> tuple[Any, str, list[Any]]:
    try:
        with open(diffile) as f:
            try:
                data_loaded: Any = json.load(f)
                function: str = ""
                data: list = []
                if isinstance(data_loaded, list) is False:
                    data = [data_loaded]
                else:
                    data = data_loaded
                for item in data:
                    parm: str = ""
                    FunDefModel(**item)
                    for k, v in item["parameters"].items():
                        value = v["type"]
                        parm += f'{k}:{value} '
                    name = item["name"]
                    desc = item["description"]
                    function += f"[name: {str(name)},"
                    function += f" description: {str(desc)}, "
                    function += f"parameters: {parm}]" + "\n"
            except (JSONDecodeError) as e:
                print("json forma not valide in "
                      f"{diffile}: {e}")
                exit(1)
            except ValidationError as e:
                print("Json schema not valide in"
                      f" {promfile}: {e}")
                exit(1)

        with open(promfile) as f:
            try:
                data = json.load(f)
                if isinstance(data, list) is False:
                    data = [data]

                for item in data:
                    PromptModel(**item)

            except (JSONDecodeError) as e:
                print("json forma not valide in"
                      f" {diffile}: {e}")
                exit(1)
            except ValidationError as e:
                print(f"Json schema not valide in "
                      f"{promfile}: {e}")
                exit(1)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
        exit(1)
    return (data_loaded, function, data)
