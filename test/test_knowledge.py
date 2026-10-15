from tools.knowledge import knowledge_search

def test_retrieve_job_requirements():
    """岗位问题应该检索到技术要求。"""
    results = knowledge_search("这个实习岗位需要掌握哪些技术？")

    assert results
    assert any(
        "Python" in item["content"]
        for item in results
    )


def test_retrieve_company_location():
    """询问公司位置应该检索到深圳信息。"""
    results = knowledge_search("公司在哪里？")

    assert results
    assert any(
        "深圳市南山区" in item["content"]
        for item in results
    )


def test_result_format():
    """检索结果应包含文本和相似度分数。"""
    results = knowledge_search("岗位要求是什么？")

    assert results

    for item in results:
        assert "content" in item
        assert "score" in item
        assert isinstance(item["content"], str)
        assert isinstance(item["score"], float)



def test_unrelated_question():
    """知识库没有相关信息时，应允许返回空结果。"""
    results = knowledge_search("公司老板姓什么？")

    assert results == []
