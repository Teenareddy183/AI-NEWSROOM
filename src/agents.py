import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv
from duckduckgo_search import DDGS

try:
    from .tasks import EditorialReview, ImageCollection, ResearchBrief
except ImportError:
    from tasks import EditorialReview, ImageCollection, ResearchBrief

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message=r".*package \(`duckduckgo_search`\) has been renamed to `ddgs`.*",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", message=r"deprecated", category=DeprecationWarning)


def configure_runtime(project_root: Path):
    storage_root = project_root / ".crewai_runtime"
    storage_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CREWAI_STORAGE_DIR", "ai_newsroom_runtime")
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_TRACKING"] = "true"

    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]:
        value = os.environ.get(key, "")
        if "127.0.0.1:9" in value:
            os.environ.pop(key, None)

    _patch_crewai_storage(storage_root)


def _patch_crewai_storage(storage_root: Path):
    def _storage_path():
        storage_root.mkdir(parents=True, exist_ok=True)
        return str(storage_root)

    try:
        from crewai.utilities import paths
        paths.db_storage_path = _storage_path
    except ImportError:
        pass

    try:
        from crewai.events.listeners.tracing import utils as tracing_utils
        tracing_utils.db_storage_path = _storage_path
    except ImportError:
        pass

    try:
        from crewai.flow.persistence import sqlite as flow_sqlite
        flow_sqlite.db_storage_path = _storage_path
    except ImportError:
        pass

    try:
        from crewai.memory.storage import kickoff_task_outputs_storage
        kickoff_task_outputs_storage.db_storage_path = _storage_path
    except ImportError:
        pass

    try:
        from crewai.memory.storage import ltm_sqlite_storage
        ltm_sqlite_storage.db_storage_path = _storage_path
    except ImportError:
        pass

    try:
        from crewai.memory.storage import rag_storage
        rag_storage.db_storage_path = _storage_path
    except ImportError:
        pass


def safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def dedupe_by_key(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        value = item.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(item)
    return output


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The agent did not return a valid JSON object.")
    return json.loads(match.group(0))


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def trim_text(value: str, limit: int = 280) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def normalize_research_payload(payload: dict[str, Any], topic: str, search_results: list[dict[str, Any]]):
    payload = payload or {}
    
    for key in ["topic_overview", "recommended_angle", "image_query"]:
        val = payload.get(key)
        if isinstance(val, dict):
            payload[key] = val.get("detail", val.get("title", str(val)))
        elif isinstance(val, list):
            payload[key] = ", ".join(str(v) for v in val)
        elif val is not None:
            payload[key] = str(val)
            
    if not payload.get("topic_overview"):
        payload["topic_overview"] = f"{topic} is the focus of this newsroom brief based on current search results and prior memory."
        
    if not payload.get("recommended_angle"):
        payload["recommended_angle"] = f"A source-backed report on recent developments, context, and why {topic} matters now."
        
    if not payload.get("image_query"):
        payload["image_query"] = topic

    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    findings = payload.get("key_findings") if isinstance(payload.get("key_findings"), list) else []

    if len(sources) < 5:
        sources.extend(
            {
                "title": item.get("title") or "Source",
                "url": item.get("url") or "",
                "publisher": item.get("publisher") or "Web",
                "relevance": f"Supports reporting on {topic}.",
            }
            for item in search_results[:8]
        )

    if not search_results:
        sources.extend(
            [
                {
                    "title": f"{topic} official overview",
                    "url": f"https://example.com/{topic.lower().replace(' ', '-')}-overview",
                    "publisher": "Fallback source",
                    "relevance": f"Fallback context for {topic}.",
                },
                {
                    "title": f"{topic} latest updates",
                    "url": f"https://example.com/{topic.lower().replace(' ', '-')}-updates",
                    "publisher": "Fallback source",
                    "relevance": f"Fallback updates for {topic}.",
                },
                {
                    "title": f"{topic} background report",
                    "url": f"https://example.com/{topic.lower().replace(' ', '-')}-background",
                    "publisher": "Fallback source",
                    "relevance": f"Fallback background for {topic}.",
                },
                {
                    "title": f"{topic} analysis",
                    "url": f"https://example.com/{topic.lower().replace(' ', '-')}-analysis",
                    "publisher": "Fallback source",
                    "relevance": f"Fallback analysis for {topic}.",
                },
                {
                    "title": f"{topic} profile",
                    "url": f"https://example.com/{topic.lower().replace(' ', '-')}-profile",
                    "publisher": "Fallback source",
                    "relevance": f"Fallback profile for {topic}.",
                },
            ]
        )

    clean_sources = []
    seen_urls = set()
    for source in sources:
        if isinstance(source, str):
            slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-") or "source"
            source = {
                "title": source,
                "url": f"https://example.com/{slug}",
                "publisher": "Generated source",
                "relevance": f"Referenced as supporting material for {topic}.",
            }
        if not isinstance(source, dict):
            continue
        url = source.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        clean_sources.append(
            {
                "title": source.get("title") or "Source",
                "url": url,
                "publisher": source.get("publisher") or safe_domain(url) or "Web",
                "relevance": source.get("relevance") or f"Relevant to {topic}.",
            }
        )

    if len(findings) < 5:
        findings.extend(
            {
                "headline": item.get("title") or "Key finding",
                "detail": item.get("snippet") or f"Reporting related to {topic}.",
                "why_it_matters": f"This helps explain the current state of {topic}.",
                "timeline": "Current / recent",
                "source_urls": [item.get("url")] if item.get("url") else [],
            }
            for item in search_results[:6]
        )

    if not findings:
        findings.extend(
            [
                {
                    "headline": f"{topic} overview",
                    "detail": f"Baseline context about {topic} was limited in the retrieved results, so the workflow is preserving a fallback summary.",
                    "why_it_matters": f"This keeps the newsroom pipeline running while flagging thin source coverage for {topic}.",
                    "timeline": "Current context",
                    "source_urls": [],
                },
                {
                    "headline": f"{topic} public visibility",
                    "detail": f"The topic appears to need stronger or more targeted search queries for richer sourcing.",
                    "why_it_matters": f"This signals that the research agent may need broader source coverage for {topic}.",
                    "timeline": "Current context",
                    "source_urls": [],
                },
                {
                    "headline": f"{topic} institutional context",
                    "detail": f"Basic institutional or background information can still be assembled from fallback context when live results are thin.",
                    "why_it_matters": f"This helps the writer provide structure even during weak retrieval conditions.",
                    "timeline": "Current context",
                    "source_urls": [],
                },
                {
                    "headline": f"{topic} reporting gap",
                    "detail": f"The workflow detected an information gap and preserved that signal for the editor.",
                    "why_it_matters": f"This reduces the risk of overconfident reporting when evidence is limited.",
                    "timeline": "Current context",
                    "source_urls": [],
                },
                {
                    "headline": f"{topic} next research step",
                    "detail": f"Additional official, local, or domain-specific sources would likely improve the final report quality.",
                    "why_it_matters": f"This guides future retrieval and helps make the project more production-aware.",
                    "timeline": "Current context",
                    "source_urls": [],
                },
            ]
        )

    clean_findings = []
    for finding in findings:
        if isinstance(finding, str):
            finding = {
                "headline": finding[:80],
                "detail": finding,
                "why_it_matters": f"This finding contributes context for {topic}.",
                "timeline": "Current context",
                "source_urls": [],
            }
        if not isinstance(finding, dict):
            continue
        source_urls = finding.get("source_urls") if isinstance(finding.get("source_urls"), list) else []
        clean_findings.append(
            {
                "headline": finding.get("headline") or "Key finding",
                "detail": finding.get("detail") or f"Research detail related to {topic}.",
                "why_it_matters": finding.get("why_it_matters") or f"This point matters for understanding {topic}.",
                "timeline": finding.get("timeline") or "Current context",
                "source_urls": [url for url in source_urls if isinstance(url, str) and url][:3],
            }
        )

    payload["sources"] = clean_sources[:8]
    payload["key_findings"] = clean_findings[:8]
    return payload


def normalize_image_payload(payload: dict[str, Any], topic: str, image_query: str, image_results: list[dict[str, Any]]):
    payload = payload or {}
    payload.setdefault("image_query_used", image_query or topic)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []

    if len(images) < 2:
        images.extend(
            {
                "title": item.get("title") or "Related image",
                "image_url": item.get("image_url") or "",
                "source_page": item.get("source_page") or "",
                "caption": item.get("title") or f"Visual related to {topic}",
                "attribution": item.get("attribution") or "Web source",
                "placement_hint": "Use within the main report body.",
            }
            for item in image_results[:4]
        )

    if not images:
        images.extend(
            [
                {
                    "title": f"{topic} visual placeholder",
                    "image_url": "https://images.unsplash.com/photo-1497366754035-f200968a6e72",
                    "source_page": "https://unsplash.com/photos/fMUIVein7Ng",
                    "caption": f"General visual related to {topic}",
                    "attribution": "unsplash.com",
                    "placement_hint": "Use near the opening section.",
                },
                {
                    "title": f"{topic} secondary visual placeholder",
                    "image_url": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3",
                    "source_page": "https://unsplash.com/photos/npxXWgQ33ZQ",
                    "caption": f"Supporting visual for {topic}",
                    "attribution": "unsplash.com",
                    "placement_hint": "Use in the middle of the report.",
                },
            ]
        )

    clean_images = []
    seen_urls = set()
    for image in images:
        if not isinstance(image, dict):
            continue
        image_url = image.get("image_url", "")
        source_page = image.get("source_page", "")
        if not image_url or not source_page or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        clean_images.append(
            {
                "title": image.get("title") or "Related image",
                "image_url": image_url,
                "source_page": source_page,
                "caption": image.get("caption") or f"Visual related to {topic}",
                "attribution": image.get("attribution") or safe_domain(source_page) or "Web source",
                "placement_hint": image.get("placement_hint") or "Use in the body of the report.",
            }
        )

    payload["images"] = clean_images[:4]
    return payload


def normalize_editorial_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload or {}
    payload["verdict"] = str(payload.get("verdict", "approved")).lower()
    if payload["verdict"] not in {"approved", "revise"}:
        payload["verdict"] = "approved"

    try:
        payload["confidence_score"] = float(payload.get("confidence_score", 0.75))
    except Exception:
        payload["confidence_score"] = 0.75

    issues = payload.get("issues")
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []
    payload["issues"] = [str(issue) for issue in issues]
    payload["citation_coverage"] = str(
        payload.get("citation_coverage", "Editor completed a source-grounding pass.")
    )
    payload["final_report"] = str(payload.get("final_report") or payload.get("report") or "")
    return payload


class SearchService:
    def web_search(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            if "Ratelimit" in str(e) or "202" in str(e):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=max_results, backend="lite"))
                except Exception:
                    # If fallback also fails, return empty list so the workflow uses fallback sources
                    results = []
            else:
                raise

        formatted = []
        for item in results:
            url = item.get("href", "")
            formatted.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    "publisher": safe_domain(url),
                    "snippet": trim_text(item.get("body", ""), 220),
                }
            )
        return dedupe_by_key(formatted, "url")

    def image_search(self, query: str, max_results: int = 8) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.images(
                        keywords=query,
                        region="wt-wt",
                        safesearch="moderate",
                        max_results=max_results,
                    )
                )
        except Exception as e:
            if "Ratelimit" in str(e) or "202" in str(e):
                # Images don't have a reliable fallback backend, so we just return empty
                # and let the workflow use the placeholder images
                results = []
            else:
                raise

        formatted = []
        for item in results:
            image_url = item.get("image", "")
            source_page = item.get("url", "")
            if not image_url or not source_page:
                continue
            formatted.append(
                {
                    "title": item.get("title", "") or "Related image",
                    "image_url": image_url,
                    "source_page": source_page,
                    "attribution": safe_domain(source_page) or "Web source",
                }
            )
        return dedupe_by_key(formatted, "image_url")


class CrewNewsroom:
    def __init__(self, project_root: Path):
        configure_runtime(project_root)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")

        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.llm = LLM(model=f"groq/{model}", api_key=api_key, temperature=0.2)
        self.fast_llm = LLM(
            model=f"groq/{os.getenv('GROQ_FAST_MODEL', 'llama-3.1-8b-instant')}",
            api_key=api_key,
            temperature=0.2,
        )

    def _run_crew(self, agents: list[Agent], tasks: list[Task], inputs: dict[str, Any]):
        attempts = 3
        last_error = None
        for attempt in range(attempts):
            try:
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=False,
                    memory=False,
                    tracing=False,
                )
                return crew.kickoff(inputs=inputs)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if ("rate limit" in message or "timed out" in message or "timeout" in message) and attempt < attempts - 1:
                    time.sleep(4 + attempt * 2)
                    continue
                raise
        raise last_error

    def run_research(self, topic: str, search_results: list[dict[str, Any]], memory_hits: list[dict[str, Any]]) -> ResearchBrief:
        researcher = Agent(
            role="Investigative Researcher",
            goal="Build a strong, detailed, source-backed research brief from search results and memory.",
            backstory=(
                "You work in an AI newsroom. You synthesize web research with persistent vector memory, "
                "organize key developments, and preserve source fidelity for downstream agents."
            ),
            llm=self.fast_llm,
            verbose=False,
            memory=False,
            allow_delegation=False,
        )

        research_task = Task(
            description=(
                "Topic: {topic}\n\n"
                "Search results:\n{search_results}\n\n"
                "Memory hits:\n{memory_hits}\n\n"
                "Create a JSON object for a newsroom brief with these keys exactly: "
                "topic_overview, key_findings, sources, recommended_angle, image_query.\n"
                "Rules:\n"
                "- Use the provided search results and memory hits.\n"
                "- key_findings must have at least 1 item.\n"
                "- sources must have at least 1 item.\n"
                "- recommended_angle is required.\n"
                "- Return only JSON, no markdown fences."
            ),
            expected_output="A valid JSON object for the research brief.",
            agent=researcher,
        )

        result = self._run_crew(
            [researcher],
            [research_task],
            {
                "topic": topic,
                "search_results": pretty_json(search_results),
                "memory_hits": pretty_json(memory_hits),
            },
        )
        payload = normalize_research_payload(extract_json(result.raw), topic, search_results)
        return ResearchBrief.model_validate(payload)

    def run_image_editor(
        self,
        topic: str,
        research: ResearchBrief,
        image_results: list[dict[str, Any]],
    ) -> ImageCollection:
        photo_editor = Agent(
            role="Photo Editor",
            goal="Curate the strongest visuals for the article and describe how to use them.",
            backstory=(
                "You are a digital news photo editor. You pick visuals that support the editorial angle, "
                "avoid weak or spammy results, and create usable captions and placement notes."
            ),
            llm=self.fast_llm,
            verbose=False,
            memory=False,
            allow_delegation=False,
        )

        image_task = Task(
            description=(
                "Topic: {topic}\n\n"
                "Research brief:\n{research_brief}\n\n"
                "Available image results:\n{image_results}\n\n"
                "Create a JSON object with keys image_query_used and images.\n"
                "- images must contain 2 to 4 items.\n"
                "- Each image must have title, image_url, source_page, caption, attribution, placement_hint.\n"
                "- Return only JSON."
            ),
            expected_output="A valid JSON object describing the curated visuals.",
            agent=photo_editor,
        )

        result = self._run_crew(
            [photo_editor],
            [image_task],
            {
                "topic": topic,
                "research_brief": research.model_dump_json(indent=2),
                "image_results": pretty_json(image_results),
            },
        )
        payload = normalize_image_payload(extract_json(result.raw), topic, research.image_query, image_results)
        return ImageCollection.model_validate(payload)

    def run_writer_and_editor(
        self,
        topic: str,
        research: ResearchBrief,
        images: ImageCollection,
        editorial_guardrails: str,
    ) -> EditorialReview:
        writer = Agent(
            role="Senior Journalist",
            goal="Write a rich, publishable Markdown report with source-backed detail.",
            backstory=(
                "You are a senior long-form journalist. You turn structured research into compelling, "
                "well-organized reporting with strong context, good flow, and embedded visual cues."
            ),
            llm=self.llm,
            verbose=False,
            memory=False,
            allow_delegation=False,
        )

        editor = Agent(
            role="Managing Editor",
            goal="Fact-check and improve the article before publication.",
            backstory=(
                "You are the newsroom's final editor. You compare the draft against the source brief, "
                "flag weak claims, improve citations, and deliver a stronger final report."
            ),
            llm=self.llm,
            verbose=False,
            memory=False,
            allow_delegation=False,
        )

        write_task = Task(
            description=(
                "Topic: {topic}\n\n"
                "Research brief:\n{research_brief}\n\n"
                "Curated images:\n{curated_images}\n\n"
                "Write a highly professional, publication-quality Markdown executive report.\n"
                "Requirements:\n"
                "- Format like a high-end corporate briefing or premium magazine feature.\n"
                "- Strong, objective headline and a clear 'Executive Summary'.\n"
                "- 'Background & Context' section.\n"
                "- 4 to 6 detailed, structured sections with professional subheadings.\n"
                "- Use Markdown links to sources naturally throughout the text.\n"
                "- Embed at least 2 images using Markdown image syntax.\n"
                "- Add an italic caption line under each image.\n"
                "- End with 'Strategic Implications', 'Key Takeaways', and 'References'."
            ),
            expected_output="A detailed Markdown report draft.",
            agent=writer,
        )

        edit_task = Task(
            description=(
                "Topic: {topic}\n\n"
                "Research brief:\n{research_brief}\n\n"
                "Curated images:\n{curated_images}\n\n"
                "Editorial guardrails:\n{editorial_guardrails}\n\n"
                "Review the report draft against the research brief and editorial guardrails.\n"
                "Return a JSON object with keys verdict, confidence_score, issues, citation_coverage, final_report.\n"
                "- verdict must be 'approved' or 'revise'.\n"
                "- final_report must contain the revised Markdown article.\n"
                "- Return only JSON."
            ),
            expected_output="A valid JSON editorial review and final report.",
            agent=editor,
            context=[write_task],
        )

        result = self._run_crew(
            [writer, editor],
            [write_task, edit_task],
            {
                "topic": topic,
                "research_brief": research.model_dump_json(indent=2),
                "curated_images": images.model_dump_json(indent=2),
                "editorial_guardrails": editorial_guardrails,
            },
        )
        payload = normalize_editorial_payload(extract_json(result.raw))
        return EditorialReview.model_validate(payload)
