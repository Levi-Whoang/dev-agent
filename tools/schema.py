import inspect
from pydantic import create_model


def python_type_to_json_type(annotation):

    if annotation is str:
        return "string"

    if annotation is int:
        return "integer"

    if annotation is float:
        return "number"

    if annotation is bool:
        return "boolean"

    return "string"


def function_to_schema(func):

    signature = inspect.signature(func)

    properties = {}

    for name, parameter in signature.parameters.items():

        json_type = python_type_to_json_type(
            parameter.annotation
        )

        properties[name] = {
            "type": json_type
        }

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(signature.parameters.keys())
            }
        }
    }

def tools_to_schema(tools):

    return [
        function_to_schema(tool)
        for tool in tools.values()
    ]

def create_args_model(func):

    signature = inspect.signature(func)

    fields = {}

    for name, parameter in signature.parameters.items():

        fields[name] = ( parameter.annotation, ... )

    return create_model(
        f"{func.__name__}Args",**fields)