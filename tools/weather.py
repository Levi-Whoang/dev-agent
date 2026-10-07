def get_weather(city: str):
    """告诉 LLM 这个工具是干什么的"""

    weather = {
        "深圳": {
            "temperature": 28,
            "weather": "晴"
        },
        "北京": {
            "temperature": 18,
            "weather": "多云"
        }
    }

    return weather.get(city, {
        "temperature": None,
        "weather": "未知"
    })