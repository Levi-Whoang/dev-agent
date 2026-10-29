"""ReAct 循环的「策略层」：纯函数和纯数据结构，不碰模型也不碰数据库。

为什么单独抽一个文件：
循环里真正容易出 bug 的不是"调模型"那几行，而是这些判断——
什么时候算卡死、观测结果要不要截断、这一步到底有没有产出答案。
它们全是纯逻辑，抽出来就能用几十毫秒跑完上百个用例；
埋在 async 生成器里就只能靠真调模型去验，既慢又不可靠。

ReAct = Reasoning + Acting。一个循环步长这样：

    Thought        模型在正文里说明"我需要什么、打算查什么"
    Action         调用工具（结构化参数，走 function calling）
    Observation    工具结果（压缩后回灌给模型）
    ...重复，直到模型不再调工具，直接给答案

关键设计取舍：**没有退回去做纯文本 ReAct**。
论文原版要求模型输出 `Action: search({"q": "..."})` 这种文本，再用正则解析。
参数里有一个引号或者嵌套花括号，解析就崩，而且工具名拼错要到运行时才发现。
这里用 function calling 执行动作（schema 校验 + 调用 id 配对），
只把「思考」这一步显式化 —— 拿论文的好处，不接它的脆弱。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------- 终止原因 ----------
ANSWER = "answer"          # 模型给出了最终答案
MAX_STEPS = "max_steps"    # 步数预算用完
LOOP_DETECTED = "loop"     # 检测到在原地打转
ERROR = "error"            # 出错了

# 观测结果默认截断长度（字符数）。
# 一条检索结果动辄两三千字，原样回灌两次就把窗口占满了，
# 而模型的下一步决策通常只需要开头那几条证据。
OBSERVATION_LIMIT = 1200


@dataclass(slots=True)
class ReactStep:
    """一次循环步：一个思考 + 若干动作 + 对应的观测结果。

    为什么一步可以放多个动作：模型支持并行工具调用（一次请求里给多个 tool_calls），
    把它们拆成多步会让步数预算莫名其妙地消耗掉。
    """

    index: int
    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    @property
    def is_final(self) -> bool:
        """没有动作 = 模型认为可以直接回答，循环到此结束。"""
        return not self.actions

    @property
    def failed(self) -> bool:
        return any(not item.get("ok", True) for item in self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "actions": self.actions,
            "observations": self.observations,
            "answer": self.answer,
            "is_final": self.is_final,
        }


class ReactTrace:
    """整条推理轨迹，给 DONE 事件和日志用。"""

    def __init__(self) -> None:
        self.steps: list[ReactStep] = []

    def new_step(self, thought: str = "") -> ReactStep:
        step = ReactStep(index=len(self.steps) + 1, thought=thought)
        self.steps.append(step)
        return step

    @property
    def tool_calls(self) -> int:
        return sum(len(s.actions) for s in self.steps)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "tool_calls": self.tool_calls,
            "thoughts": sum(1 for s in self.steps if s.thought),
        }


def normalize_arguments(raw: str | dict | None) -> str:
    """把工具参数归一化成可比较的字符串。

    不这么做的话，`{"a":1,"b":2}` 和 `{"b":2,"a":1}` 会被当成两个不同的动作，
    卡死检测就失效了 —— 而模型重试时经常只是换了键的顺序。
    """
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    if not raw:
        return "{}"
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return str(raw)


class LoopGuard:
    """检测"在原地打转"。

    典型症状：工具报错 → 模型用一模一样的参数再试一次 → 再失败 → 再试。
    没有这个守卫，模型会把步数预算全部烧在同一个失败调用上，
    用户等了半分钟什么也没拿到。

    分三级处理，而不是第一次重复就掐断：
    - 第 1 次相同：正常，可能就是并行/重试
    - 累计到阈值：给模型一句提醒，让它换个思路
    - 再重复一次：直接终止，别浪费更多 token
    """

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = max(2, threshold)
        self._last_key: str | None = None
        self._repeat = 0
        self._warned = False

    def check(self, name: str, arguments: str | dict | None) -> str:
        """返回 ok / warn / abort。"""
        key = f"{name}::{normalize_arguments(arguments)}"

        if key == self._last_key:
            self._repeat += 1
        else:
            self._last_key = key
            self._repeat = 1
            self._warned = False

        if self._repeat >= self.threshold + 1:
            return "abort"
        if self._repeat == self.threshold:
            self._warned = True
            return "warn"
        return "ok"

    def reset(self) -> None:
        """换了新动作就清零，避免"隔几步又重复一次"被误判。"""
        self._last_key = None
        self._repeat = 0
        self._warned = False

    @property
    def stuck_hint(self) -> str:
        return (
            "你正在用完全相同的参数重复调用同一个工具。"
            "请换一个思路：改写查询词、换一个工具，或者基于已有信息直接回答。"
            "如果确实无法获取所需信息，请明确告知用户当前无法确认。"
        )


def compress_observation(result_text: str, limit: int = OBSERVATION_LIMIT) -> str:
    """压缩一条观测结果。

    截断而不是摘要：这一步不该再花一次模型调用（成本 + 延迟），
    而且检索结果的有用部分基本都在前面。
    """
    text = result_text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（工具输出过长，已截断；原文共 {len(text)} 字）"


_WS = re.compile(r"[ \t\u3000]+")


def extract_thought(content: str | None, limit: int = 800) -> str:
    """从 assistant 正文里取出"思考"。

    模型有时会先输出一段自然语言再调工具，这段就是天然的 Thought；
    有时什么都不说直接调工具 —— 那就返回空字符串，**不要伪造**。
    伪造出来的思考比没有更糟：日志和前端都会把它当成模型的真实推理。
    """
    text = (content or "").strip()
    if not text:
        return ""
    # 多行思考压成一行，前端展示时不会把一个气泡撑得很长
    text = _WS.sub(" ", text)
    text = re.sub(r"\n+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def should_stop(step: ReactStep) -> bool:
    """这一步能不能收尾。

    只有"没有动作"才叫结束。有些模型会在正文里写"我准备好了"却仍然调工具，
    那种情况必须继续跑 —— 否则用户会收到一句没有内容的空话。
    """
    return step.is_final and bool(step.answer.strip())
