"""计算工具。

改成 async def 的原因：Agent 循环统一 `await tool(...)`，
混着同步函数会逼着执行器到处判断，接口越简单越好。
计算本身是纯 CPU 微秒级操作，不需要真的进线程池。
"""

from __future__ import annotations

from typing import Literal


async def calculate(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> dict:
    """进行两个数字之间的数学运算，返回结果或错误说明。"""
    if operation == "add":
        return {"result": a + b}
    if operation == "subtract":
        return {"result": a - b}
    if operation == "multiply":
        return {"result": a * b}
    if operation == "divide":
        if b == 0:
            # 让模型看到明确的失败原因，而不是一个 ZeroDivisionError 字符串
            return {"error": "除数不能为 0"}
        return {"result": a / b}

    return {"error": f"不支持的运算：{operation}"}
