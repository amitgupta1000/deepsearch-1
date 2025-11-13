import asyncio
import pytest

from backend.src.nodes import AgentState

# Simple mocks for llm and search components
class DummyResult:
    def __init__(self, url, snippet):
        self.url = url
        self.snippet = snippet

class DummySearcher:
    async def search(self, q):
        # Return a list of DummyResult objects
        return [DummyResult(f"https://example.com/{i}", f"Snippet about {q} #{i}") for i in range(3)]

class DummyLLM:
    async def ainvoke(self, messages):
        class R:
            content = '{"is_sufficient": true, "knowledge_gap": "", "follow_up_queries": []}'
        return R()

@pytest.mark.asyncio
async def test_run_degraded_workflow(monkeypatch, tmp_path):
    # Monkeypatch components
    from backend.src import nodes
    monkeypatch.setattr(nodes, 'UnifiedSearcher', DummySearcher)
    monkeypatch.setattr(nodes, 'llm', DummyLLM())

    initial_state: AgentState = {
        "new_query": "test query",
        "search_queries": [],
        "rationale": None,
        "data": [],
        "relevant_contexts": {},
        "relevant_chunks": [],
        "proceed": True,
        "visited_urls": [],
        "iteration_count": 0,
        "report": None,
        "report_filename": "",
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "prompt_type": "general",
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None,
    }

    # Run create_queries -> evaluate_search_results (direct calls)
    state = await nodes.create_queries(initial_state)
    assert isinstance(state.get('search_queries', []), list)

    state = await nodes.evaluate_search_results(state)
    assert isinstance(state.get('data', []), list)

@pytest.mark.asyncio
async def test_investment_research_prompts():
    pass
