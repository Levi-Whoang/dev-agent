from tools.knowledge import knowledge_search
from tools.order import get_order
from tools.weather import get_weather
from tools.calculator import calculate
from tools.knowledge import knowledge_search_context


TOOLS = {
    "get_order": get_order,
    "get_weather": get_weather,
    "calculate": calculate,
    "knowledge_search": knowledge_search,
    "knowledge_search_context": knowledge_search_context,

}