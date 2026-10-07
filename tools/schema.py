import inspect
from pydantic import create_model


def create_args_model(func):
    signature = inspect.signature(func)
    fields = {}

    for name, parameter in signature.parameters.items():

        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"工具 {func.__name__} 的参数 {name} 缺少类型注解"
            )

        if parameter.default is inspect.Parameter.empty:
            fields[name] = (
                parameter.annotation,
                ...
            )
        else:
            fields[name] = (
                parameter.annotation,
                parameter.default
            )

    return create_model(
        f"{func.__name__}Args",
        **fields
    )

def function_to_schema(func):
    args_model = create_args_model(func)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": args_model.model_json_schema()
        }
    }


def tools_to_schema(tools):
    return [
        function_to_schema(tool)
        for tool in tools.values()
    ]