"""跨分片过滤内部推理标签。"""
import re
class PublicTextFilter:
    # 初始化待完成标签与隐藏区域状态。
    def __init__(self):
        self.pending = ""
        self.hidden = []
    # 增量提取可公开文本，支持标签跨多个分片。
    def feed(self, text):
        result = []
        for char in text:
            if self.pending or char == "<":
                self.pending += char
                if char != ">":
                    continue
                match = re.fullmatch(r"<(/?)(think|analysis|reasoning)\b[^>]*>", self.pending, re.I)
                if match:
                    closing, name = match.groups()
                    if closing:
                        if self.hidden and self.hidden[-1] == name.lower():
                            self.hidden.pop()
                    else:
                        self.hidden.append(name.lower())
                elif not self.hidden:
                    result.append(self.pending)
                self.pending = ""
            elif not self.hidden:
                result.append(char)
        return "".join(result)
