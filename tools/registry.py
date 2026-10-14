from tools.knowledge import knowledge_search
from tools.order import get_order
from tools.weather import get_weather
from tools.calculator import calculate


TOOLS = {
    "get_order": get_order,
    "get_weather": get_weather,
    "calculate": calculate,
    "knowledge_search": knowledge_search
}