from pathlib import Path

from dotenv import load_dotenv

try:
    from .workflow import build_news_graph
except ImportError:
    from workflow import build_news_graph

load_dotenv()


def run_news_crew(topic_input: str):
    project_root = Path(__file__).resolve().parent.parent
    graph = build_news_graph(project_root)

    state = graph.invoke(
        {
            "topic": topic_input,
            "project_root": str(project_root),
            "editorial_guardrails": (
                "Prefer precise claims, cite sources often, avoid unsupported hype, "
                "and make the final report read like a polished feature article."
            ),
        }
    )

    review = state["editorial_review"]
    research = state["research_data"]
    images = state["image_data"]

    return {
        "report": review["final_report"],
        "image_query": images["image_query_used"],
        "images": images["images"],
        "sources": research["sources"],
        "research_overview": research["topic_overview"],
        "recommended_angle": research["recommended_angle"],
        "findings": research["key_findings"],
        "editorial_review": {
            "verdict": review["verdict"],
            "confidence_score": review["confidence_score"],
            "issues": review["issues"],
            "citation_coverage": review["citation_coverage"],
        },
        "workflow_trace": state.get("workflow_trace", []),
        "memory_hits": state.get("memory_hits", []),
        "search_queries": state.get("search_queries", []),
    }


if __name__ == "__main__":
    topic = input("Enter a topic to research: ")
    result = run_news_crew(topic)
    print(result["report"])
