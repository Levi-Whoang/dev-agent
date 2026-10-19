# 数据库游标操控  
## cursor：自定义游标
cursor.execute()：执行 SQL。\
cursor.fetchone()：获取一条查询结果。\
cursor.fetchall()：获取所有查询结果。




### isinstance() 
是 Python 内置函数，用来判断一个对象是否属于指定类型。
eg：isinstance(messages, str)判断message是不是str

### json.loads() 
用来把 JSON 格式的字符串转换成 Python 数据。


# log日志
logger = logging.getLogger("dev-agent")
这行代码获取一个名为 "dev-agent" 的日志记录器，并把它赋值给变量 logger。\
后面就可以使用它：\
logger.info("程序启动")\
logger.warning("参数缺失")\
logger.error("工具执行失败")




@lru_cache(maxsize=1) 是 Python 中的缓存装饰器，作用是：缓存函数的计算结果，避免相同参数重复执行函数。
在 Agent 项目中，它经常用来缓存模型对象，例如 Embedding 模型，避免每次调用都重新加载模型。