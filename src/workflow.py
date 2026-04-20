from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

try:
    from .agents import CrewNewsroom, SearchService
    from .rag import NewsroomMemory
    from .tasks import ImageCollection, ResearchBrief
except ImportError:
    from agents import CrewNewsroom, SearchService
    from rag import NewsroomMemory
    from tasks import ImageCollection, ResearchBrief


class NewsState(TypedDict, total=False):
    topic: str
    project_root: str
    search_queries: list[str]
    search_results: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    image_results: list[dict[str, Any]]
    search_attempts: int
    research_data: dict[str, Any]
    image_data: dict[str, Any]
    editorial_review: dict[str, Any]
    workflow_trace: list[str]
    editorial_guardrails: str


def build_news_graph(project_root: Path):
    search_service = SearchService()
    memory = NewsroomMemory(project_root / ".newsroom_memory")
    newsroom = CrewNewsroom(project_root)

    def prepare_context(state: NewsState) -> NewsState:
        topic = state["topic"]
        queries = [
            topic,
            f"{topic} latest developments",
        ]
        search_results = []
        for query in queries:
            search_results.extend(search_service.web_search(query, max_results=3))
        search_results = search_results[:6]

        raw_memory_hits = memory.retrieve(topic, limit=2)
        memory_hits = [
            {
                "topic": item.get("topic", ""),
                "memory_topic": item.get("memory_topic", ""),
                "overview": item.get("overview", ""),
                "sources": [
                    {"title": source.get("title", ""), "publisher": source.get("publisher", "")}
                    for source in item.get("sources", [])[:3]
                    if isinstance(source, dict)
                ],
            }
            for item in raw_memory_hits
        ]
        return {
            "search_queries": queries,
            "search_results": search_results,
            "memory_hits": memory_hits,
            "search_attempts": 1,
            "workflow_trace": ["Prepared search context and retrieved vector memory."],
        }

    def route_research(state: NewsState) -> str:
        if len(state.get("search_results", [])) >= 5:
            return "run_research"
        return "retry_research"

    def retry_research(state: NewsState) -> NewsState:
        topic = state["topic"]
        fallback_queries = [
            f"{topic} official website",
            f"{topic} report pdf",
        ]
        search_results = list(state.get("search_results", []))
        for query in fallback_queries:
            search_results.extend(search_service.web_search(query, max_results=2))
        return {
            "search_queries": list(state.get("search_queries", [])) + fallback_queries,
            "search_results": search_results[:8],
            "search_attempts": state.get("search_attempts", 1) + 1,
            "workflow_trace": list(state.get("workflow_trace", []))
            + ["Search results were weak, so LangGraph triggered a fallback search branch."],
        }

    def run_research(state: NewsState) -> NewsState:
        research = newsroom.run_research(
            state["topic"],
            state.get("search_results", []),
            state.get("memory_hits", []),
        )
        return {
            "research_data": research.model_dump(),
            "workflow_trace": list(state.get("workflow_trace", []))
            + ["CrewAI researcher agent produced a structured research brief."],
        }

    def run_image_editor(state: NewsState) -> NewsState:
        research = state["research_data"]
        image_results = search_service.image_search(research["image_query"], max_results=4)
        if len(image_results) < 2:
            image_results.extend(search_service.image_search(state["topic"], max_results=3))

        images = newsroom.run_image_editor(
            state["topic"],
            _model_from_dump("research", research),
            image_results[:4],
        )
        return {
            "image_results": image_results[:4],
            "image_data": images.model_dump(),
            "workflow_trace": list(state.get("workflow_trace", []))
            + ["CrewAI photo editor agent selected visuals for the story."],
        }

    def run_writer_editor(state: NewsState) -> NewsState:
        research_model = _model_from_dump("research", state["research_data"])
        image_model = _model_from_dump("images", state["image_data"])
        review = newsroom.run_writer_and_editor(
            state["topic"],
            research_model,
            image_model,
            state["editorial_guardrails"],
        )
        return {
            "editorial_review": review.model_dump(),
            "workflow_trace": list(state.get("workflow_trace", []))
            + ["CrewAI writer and editor agents produced the final reviewed article."],
        }

    def persist_memory(state: NewsState) -> NewsState:
        review = state["editorial_review"]
        research = state["research_data"]
        memory.store_report(
            topic=state["topic"],
            report=review["final_report"],
            sources=research["sources"],
            overview=research["topic_overview"],
        )
        return {
            "workflow_trace": list(state.get("workflow_trace", []))
            + ["Final report stored in Chroma memory for future retrieval."],
        }

    graph = StateGraph(NewsState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("retry_research", retry_research)
    graph.add_node("run_research", run_research)
    graph.add_node("run_image_editor", run_image_editor)
    graph.add_node("run_writer_editor", run_writer_editor)
    graph.add_node("persist_memory", persist_memory)

    graph.set_entry_point("prepare_context")
    graph.add_conditional_edges(
        "prepare_context",
        route_research,
        {
            "run_research": "run_research",
            "retry_research": "retry_research",
        },
    )
    graph.add_edge("retry_research", "run_research")
    graph.add_edge("run_research", "run_image_editor")
    graph.add_edge("run_image_editor", "run_writer_editor")
    graph.add_edge("run_writer_editor", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile()


def _model_from_dump(kind: str, payload: dict[str, Any]):
    if kind == "research":
        return ResearchBrief.model_validate(payload)

    return ImageCollection.model_validate(payload)
