from typing import Literal

def calculate(a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]):
    """进行两个数字之间的数学运算"""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        return a / b

    return "不支持的运算"