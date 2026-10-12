# # tools/schemas.py
#
# TOOLS_SCHEMA = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_order",
#             "description": "查询订单状态和金额",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "order_id": {
#                         "type": "string",
#                         "description": "订单ID"
#                     }
#                 },
#                 "required": ["order_id"]
#             }
#         }
#     },
#
#     {
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "查询指定城市的天气",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "city": {
#                         "type": "string",
#                         "description": "城市名称"
#                     }
#                 },
#                 "required": ["city"]
#             }
#         }
#     },
#
#     {
#         "type": "function",
#         "function": {
#             "name": "calculate",
#             "description": "进行两个数字之间的数学运算",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "a": {
#                         "type": "number"
#                     },
#                     "b": {
#                         "type": "number"
#                     },
#                     "operation": {
#                         "type": "string",
#                         "enum": [
#                             "add",
#                             "subtract",
#                             "multiply",
#                             "divide"
#                         ]
#                     }
#                 },
#                 "required": [
#                     "a",
#                     "b",
#                     "operation"
#                 ]
#             }
#         }
#     }
# ]