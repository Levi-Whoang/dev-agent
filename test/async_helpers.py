"""异步模型流的离线测试替身。"""
class AsyncStream:
    # 包装分片并记录连接关闭状态。
    def __init__(self, items):
        self.items = iter(items)
        self.closed = False

    # 返回异步迭代器。
    def __aiter__(self):
        return self

    # 逐个提供模拟模型分片。
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration

    # 模拟异步连接关闭。
    async def close(self):
        self.closed = True
        close = getattr(self.items, "close", None)
        if close:
            close()

# 收集异步生成器输出以供断言。
async def collect(stream):
    return [item async for item in stream]
