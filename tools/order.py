def get_order(order_id: str):
    """查询订单状态和金额"""

    orders = {
        "10086": {
            "status": "paid",
            "amount": 99.9
        },
        "10087": {
            "status": "pending",
            "amount": 199.0
        }
    }

    return orders.get(order_id)