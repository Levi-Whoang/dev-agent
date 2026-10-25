"""天气工具（演示用假数据）。

真实项目里这里应该是 `await httpx.AsyncClient().get(...)`，
写异步的价值就在这里——网络等待不该占住线程。
"""

from __future__ import annotations

_WEATHER = {
    "深圳": {"temperature": 28, "weather": "晴"},
    "北京": {"temperature": 18, "weather": "多云"},
}


async def get_weather(city: str) -> dict:
    """查询指定城市的天气（示例数据，仅支持深圳和北京）。"""
    return _WEATHER.get(
        city, {"temperature": None, "weather": "未知"}
    )
