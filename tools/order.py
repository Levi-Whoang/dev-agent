"""订单查询工具（演示用假数据）。"""

from __future__ import annotations

_ORDERS = {
    "10086": {"status": "paid", "amount": 99.9},
    "10087": {"status": "pending", "amount": 199.0},
}


async def get_order(order_id: str) -> dict:
    """查询订单状态和金额。"""
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"订单不存在：{order_id}"}
    return {"order_id": order_id, **order}
