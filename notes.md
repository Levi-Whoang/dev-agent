# python基础知识
## 编码规则
| 方法 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `str.encode("utf-8")` | `str` | `bytes` | 将文本编码为 UTF-8 字节 |
| `b.decode("utf-8")` | `bytes` | `str` | 将 UTF-8 字节解码为文本 |
| `bytes.fromhex(s)` | 十六进制 `str` | `bytes` | 将十六进制字符串转换为字节 |
| `b.hex()` | `bytes` | 十六进制 `str` | 将字节转换为十六进制字符串 |
| `bytearray.fromhex(s)` | 十六进制 `str` | `bytearray` | 将十六进制字符串转换为可变字节数组 |
| `bytes(b)` | 可迭代整数或字节数据等 | `bytes` | 创建不可变字节对象 |
| `bytearray(b)` | 可迭代整数或字节数据等 | `bytearray` | 创建可变字节数组 |
| `base64.b64encode(b)` | `bytes` | `bytes` | 将字节数据编码为 Base64 |
| `base64.b64decode(b)` | Base64 数据 | `bytes` | 将 Base64 数据解码为原始字节 |
| `hashlib.sha256(b).digest()` | `bytes` | `bytes` | 计算 SHA-256 摘要，返回原始字节 |
| `hashlib.sha256(b).hexdigest()` | `bytes` | 十六进制 `str` | 计算 SHA-256 摘要，返回十六进制字符串 |
| `secrets.token_hex(n)` | 字节数量 `n` | 十六进制 `str` | 生成安全随机数据并转换为十六进制字符串 |
### global
在某个函数内需要修改某个全局变量的时候需要使用global（读取不用，修改列表不用）
```python
x = 10
def test():
    global x
    x = 20         # ② 重新赋值，需要 global`
```

# knowledge.py
## log日志
logger = logging.getLogger("dev-agent")
这行代码获取一个名为 "dev-agent" 的日志记录器，并把它赋值给变量 logger。
后面就可以使用它：
logger.info("程序启动")
logger.warning("参数缺失")
logger.error("工具执行失败")

## @lru_cache(maxsize=1) 
是 Python 中的缓存装饰器，作用是：缓存函数的计算结果，避免相同参数重复执行函数。
在 Agent 项目中，它经常用来缓存模型对象，例如 Embedding 模型，避免每次调用都重新加载模型。



## BGE （BAAI General Embedding）
将文本转换为向量

## RRF
### RRF_K = 60
平滑参数\
越大，前几名之间的分数差异越小；越小，排名靠前的文档获得的优势越明显。

## assert 
是 Python 的一个断言语句，主要用于检查某个条件是否成立。\
age = 20\
assert age >= 18\
print("程序继续执行")\
反之抛出AssertionError异常

## model.encode() 
通常是 sentence-transformers 库中的一个方法\
作用是：把文本转换成向量（Embedding），让计算机能够通过数学计算比较文本的语义相似度。



# SQLAlchemy
**create_engine()** 创建 SQLAlchemy 的数据库引擎对象，负责协调连接池和数据库连接。




# 数据库游标操控  cursor：自定义游标
cursor.execute()：执行 SQL。
cursor.fetchone()：获取一条查询结果。
cursor.fetchall()：获取所有查询结果。

isinstance() 是 Python 内置函数，用来判断一个对象是否属于指定类型。
eg：isinstance(messages, str)判断message是不是str

json.loads() 用来把 JSON 格式的字符串转换成 Python 数据。





# JWT
