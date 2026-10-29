"""ReAct 策略层的纯逻辑测试：不打网络、不碰数据库。

这些函数决定了循环"什么时候停、停得对不对"，是整套 ReAct 里最容易出
隐蔽 bug 的地方（边界条件、参数顺序、空值），所以单独拉出来密集测。
"""

from __future__ import annotations

import pytest

from agent.react import (
    OBSERVATION_LIMIT,
    LoopGuard,
    ReactStep,
    ReactTrace,
    compress_observation,
    extract_thought,
    normalize_arguments,
    should_stop,
)


# ---------- 参数归一化 ----------
def test_normalize_arguments_ignores_key_order():
    a = normalize_arguments('{"query": "报销", "top_k": 3}')
    b = normalize_arguments('{"top_k": 3, "query": "报销"}')
    assert a == b


def test_normalize_arguments_keeps_non_ascii_readable():
    # 不能转成 \u62a5\u9500，否则日志里根本看不懂模型在查什么
    assert "报销" in normalize_arguments('{"query": "报销"}')


def test_normalize_arguments_falls_back_on_broken_json():
    # 模型偶尔吐出半截 JSON，不能让归一化本身抛异常
    assert normalize_arguments('{"query": ') == '{"query": '


@pytest.mark.parametrize("empty", [None, "", {}])
def test_normalize_arguments_handles_empty(empty):
    assert normalize_arguments(empty) == "{}"


# ---------- 卡死检测 ----------
def test_loop_guard_allows_distinct_actions():
    guard = LoopGuard(threshold=2)
    assert guard.check("search", '{"q": "a"}') == "ok"
    assert guard.check("search", '{"q": "b"}') == "ok"
    assert guard.check("search", '{"q": "c"}') == "ok"


def test_loop_guard_escalates_same_action():
    guard = LoopGuard(threshold=2)
    assert guard.check("search", '{"q": "a"}') == "ok"
    assert guard.check("search", '{"q": "a"}') == "warn"
    assert guard.check("search", '{"q": "a"}') == "abort"


def test_loop_guard_detects_reordered_arguments():
    """模型重试时常常只是换了键顺序，不算换了思路。"""
    guard = LoopGuard(threshold=2)
    assert guard.check("search", '{"q": "a", "k": 1}') == "ok"
    assert guard.check("search", '{"k": 1, "q": "a"}') == "warn"


def test_loop_guard_resets_on_different_action():
    guard = LoopGuard(threshold=2)
    guard.check("search", '{"q": "a"}')
    guard.check("search", '{"q": "a"}')  # warn
    guard.check("search", '{"q": "b"}')  # 换了参数，计数清零
    assert guard.check("search", '{"q": "b"}') == "warn"


def test_loop_guard_threshold_is_never_below_two():
    """阈值 1 会导致"第二次相同调用就掐断"，正常重试也会被误杀。"""
    assert LoopGuard(threshold=1).threshold == 2


def test_loop_guard_hint_mentions_alternatives():
    assert "换一个思路" in LoopGuard().stuck_hint


# ---------- 观测压缩 ----------
def test_compress_observation_keeps_short_text():
    assert compress_observation("短结果") == "短结果"


def test_compress_observation_truncates_long_text():
    text = "证" * (OBSERVATION_LIMIT + 500)
    out = compress_observation(text)
    assert out.startswith("证" * 100)
    assert "已截断" in out
    assert str(len(text)) in out
    # 截断后长度必须是有限且明显小于原文
    assert len(out) < len(text)


def test_compress_observation_respects_custom_limit():
    assert len(compress_observation("x" * 100, limit=10)) < 100


def test_compress_observation_handles_none():
    assert compress_observation(None) == ""


# ---------- 思考提取 ----------
def test_extract_thought_returns_natural_language():
    assert extract_thought("我需要先查报销制度。") == "我需要先查报销制度。"


def test_extract_thought_does_not_fabricate():
    """空正文就是空思考 —— 不能编一句"用户想了解…"放进 trace。"""
    assert extract_thought("") == ""
    assert extract_thought(None) == ""
    assert extract_thought("   \n ") == ""


def test_extract_thought_collapses_newlines():
    assert "\n" not in extract_thought("第一行\n第二行\n\n第三行")


def test_extract_thought_truncates():
    out = extract_thought("思" * 900, limit=800)
    assert out.endswith("…")
    assert len(out) == 801


# ---------- 轨迹 ----------
def test_trace_counts_steps_and_tool_calls():
    trace = ReactTrace()
    first = trace.new_step("查一下")
    first.actions.append({"name": "search", "arguments": "{}"})
    trace.new_step()  # 直接给答案的一步

    assert [s.index for s in trace.steps] == [1, 2]
    assert trace.summary() == {"steps": 2, "tool_calls": 1, "thoughts": 1}


def test_step_is_final_when_no_actions():
    assert ReactStep(index=1, thought="直接回答").is_final
    assert not ReactStep(index=1, actions=[{"name": "x"}]).is_final


def test_step_failed_when_any_observation_failed():
    step = ReactStep(index=1)
    step.observations.append({"name": "a", "ok": True})
    assert not step.failed
    step.observations.append({"name": "b", "ok": False})
    assert step.failed


def test_should_stop_requires_non_empty_answer():
    """"我准备好了"这种空话不能算结束。"""
    assert should_stop(ReactStep(index=1, answer="答案"))
    assert not should_stop(ReactStep(index=1, answer=""))
    assert not should_stop(ReactStep(index=1, answer="   "))
