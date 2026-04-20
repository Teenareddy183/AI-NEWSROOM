from typing import List

from pydantic import BaseModel, Field


class ResearchSource(BaseModel):
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Direct source URL")
    publisher: str = Field(..., description="Publisher or publication name")
    relevance: str = Field(..., description="Why the source matters")


class ResearchFinding(BaseModel):
    headline: str = Field(..., description="Short headline for the finding")
    detail: str = Field(..., description="Detailed explanation of the finding")
    why_it_matters: str = Field(..., description="Why the finding matters")
    timeline: str = Field(..., description="Date or timeframe")
    source_urls: List[str] = Field(..., description="Supporting source URLs")


class ResearchBrief(BaseModel):
    topic_overview: str = Field(..., description="High-level topic summary")
    key_findings: List[ResearchFinding] = Field(
        ..., min_length=1, description="Detailed findings"
    )
    sources: List[ResearchSource] = Field(
        ..., min_length=1, description="Reliable sources used in the report"
    )
    recommended_angle: str = Field(..., description="Recommended editorial angle")
    image_query: str = Field(..., description="Best query to find relevant visuals")


class CuratedImage(BaseModel):
    title: str = Field(..., description="Image title")
    image_url: str = Field(..., description="Direct image URL")
    source_page: str = Field(..., description="Source page URL")
    caption: str = Field(..., description="Caption to show in the article")
    attribution: str = Field(..., description="Photographer or source attribution")
    placement_hint: str = Field(..., description="Where to place the image in the report")


class ImageCollection(BaseModel):
    image_query_used: str = Field(..., description="Image query that was used")
    images: List[CuratedImage] = Field(
        ..., min_length=2, max_length=4, description="Curated images for the article"
    )


class EditorialReview(BaseModel):
    verdict: str = Field(..., description="approved or revise")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1")
    issues: List[str] = Field(default_factory=list, description="Editorial issues found")
    citation_coverage: str = Field(..., description="Assessment of source coverage")
    final_report: str = Field(..., description="Revised final report in Markdown")
