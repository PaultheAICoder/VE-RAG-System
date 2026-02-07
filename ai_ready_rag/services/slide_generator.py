"""Slide Generator Service - Prototype for Phase 2.

Uses prompt chaining to generate presentation slides with citations:
1. Retrieve relevant context
2. Generate presentation outline
3. Expand each slide with cited content
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_ollama import ChatOllama

from ai_ready_rag.config import Settings, get_settings
from ai_ready_rag.services.rag_service import RAGService

logger = logging.getLogger(__name__)


@dataclass
class SlideContent:
    """Single slide content."""

    slide_number: int
    title: str
    bullet_points: list[str]
    citations: list[str]  # SourceIds referenced
    speaker_notes: str = ""


@dataclass
class Presentation:
    """Generated presentation."""

    title: str
    subtitle: str
    slides: list[SlideContent]
    total_sources: int
    generation_steps: list[str] = field(default_factory=list)


# Prompt templates
OUTLINE_PROMPT = """You are creating a presentation outline based on internal company documents.

TOPIC: {topic}

CONTEXT FROM DOCUMENTS:
{context}

Generate a JSON outline for a {num_slides}-slide presentation. Include:
- A compelling title
- A subtitle
- For each slide: a title and 2-3 key points to cover

IMPORTANT: Base the outline ONLY on information in the context above.

Respond with ONLY valid JSON in this exact format:
{{
  "title": "Presentation Title",
  "subtitle": "Subtitle or tagline",
  "slides": [
    {{"slide_number": 1, "title": "Slide Title", "key_points": ["point 1", "point 2"]}},
    {{"slide_number": 2, "title": "Slide Title", "key_points": ["point 1", "point 2", "point 3"]}}
  ]
}}"""

SLIDE_EXPANSION_PROMPT = """You are expanding a presentation slide with detailed content from company documents.

SLIDE TITLE: {slide_title}
KEY POINTS TO COVER: {key_points}

CONTEXT FROM DOCUMENTS:
{context}

Expand this slide into detailed bullet points. Each bullet point MUST include a citation [SourceId: xxx].

Rules:
1. 3-5 bullet points maximum
2. Each bullet must cite a source using [SourceId: document_id:chunk_index]
3. Keep bullets concise (1-2 sentences)
4. Add brief speaker notes (what to say when presenting)

Respond with ONLY valid JSON:
{{
  "title": "{slide_title}",
  "bullet_points": [
    "First point with specific fact [SourceId: abc123:0]",
    "Second point with detail [SourceId: def456:2]"
  ],
  "speaker_notes": "Brief notes about what to emphasize when presenting this slide"
}}"""


class SlideGeneratorService:
    """Generates presentations using RAG + prompt chaining."""

    def __init__(
        self,
        rag_service: RAGService,
        settings: Settings | None = None,
    ):
        self.rag = rag_service
        self.settings = settings or get_settings()
        self.llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.3,  # Lower temp for more consistent structure
        )

    async def generate_presentation(
        self,
        topic: str,
        user_tags: list[str],
        num_slides: int = 5,
        template: Literal["executive", "detailed", "marketing"] = "executive",
    ) -> Presentation:
        """Generate a full presentation on a topic.

        Args:
            topic: The presentation topic/question
            user_tags: User's access tags for document filtering
            num_slides: Number of content slides (excluding title)
            template: Presentation style template

        Returns:
            Presentation with slides and citations
        """
        steps = []

        # Step 1: Retrieve relevant context
        logger.info(f"[SLIDES] Step 1: Retrieving context for topic: {topic}")
        steps.append("Retrieved relevant documents")

        # Use RAG to get context (but we just want the chunks, not the answer)
        context_chunks = await self._get_context(topic, user_tags)

        if not context_chunks:
            raise ValueError("No relevant documents found for this topic")

        context_text = self._format_context(context_chunks)
        logger.info(f"[SLIDES] Found {len(context_chunks)} relevant chunks")

        # Step 2: Generate outline
        logger.info("[SLIDES] Step 2: Generating presentation outline")
        steps.append("Generated presentation outline")

        outline = await self._generate_outline(topic, context_text, num_slides)
        logger.info(f"[SLIDES] Outline: {outline['title']} with {len(outline['slides'])} slides")

        # Step 3: Expand each slide
        logger.info("[SLIDES] Step 3: Expanding slides with citations")
        slides = []

        for slide_outline in outline["slides"]:
            logger.info(
                f"[SLIDES] Expanding slide {slide_outline['slide_number']}: {slide_outline['title']}"
            )
            steps.append(f"Expanded slide: {slide_outline['title']}")

            slide = await self._expand_slide(
                slide_outline["title"],
                slide_outline["key_points"],
                context_text,
                slide_outline["slide_number"],
            )
            slides.append(slide)

        # Extract unique sources
        all_citations = set()
        for slide in slides:
            all_citations.update(slide.citations)

        return Presentation(
            title=outline["title"],
            subtitle=outline["subtitle"],
            slides=slides,
            total_sources=len(all_citations),
            generation_steps=steps,
        )

    async def _get_context(
        self,
        topic: str,
        user_tags: list[str],
    ) -> list[dict]:
        """Retrieve relevant document chunks."""
        # Use vector service directly for more chunks
        vector_service = self.rag.vector_service

        # Get embedding for topic
        embedding = await vector_service.embed(topic)

        # Search with more results for presentation context
        results = await vector_service.search(
            query_embedding=embedding,
            user_tags=user_tags,
            top_k=15,  # More context for presentations
        )

        return [
            {
                "source_id": f"{r.document_id}:{r.chunk_index}",
                "document_name": r.document_name,
                "content": r.content,
                "relevance": r.relevance_score,
            }
            for r in results
        ]

    def _format_context(self, chunks: list[dict]) -> str:
        """Format chunks into context string with source IDs."""
        formatted = []
        for chunk in chunks:
            formatted.append(
                f"[SourceId: {chunk['source_id']}] (from {chunk['document_name']})\n"
                f"{chunk['content']}\n"
            )
        return "\n---\n".join(formatted)

    async def _generate_outline(
        self,
        topic: str,
        context: str,
        num_slides: int,
    ) -> dict:
        """Generate presentation outline."""
        prompt = OUTLINE_PROMPT.format(
            topic=topic,
            context=context[:8000],  # Limit context size
            num_slides=num_slides,
        )

        response = await self.llm.ainvoke(prompt)
        content = response.content

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            raise ValueError(f"Could not parse outline JSON from response: {content[:200]}")

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"[SLIDES] JSON parse error: {e}\nContent: {content}")
            raise ValueError(f"Invalid JSON in outline response: {e}") from None

    async def _expand_slide(
        self,
        title: str,
        key_points: list[str],
        context: str,
        slide_number: int,
    ) -> SlideContent:
        """Expand a single slide with citations."""
        prompt = SLIDE_EXPANSION_PROMPT.format(
            slide_title=title,
            key_points=", ".join(key_points),
            context=context[:6000],  # Limit per-slide context
        )

        response = await self.llm.ainvoke(prompt)
        content = response.content

        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            # Fallback: create slide from key points without citations
            logger.warning("[SLIDES] Could not parse slide JSON, using fallback")
            return SlideContent(
                slide_number=slide_number,
                title=title,
                bullet_points=key_points,
                citations=[],
                speaker_notes="",
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("[SLIDES] JSON parse error for slide, using fallback")
            return SlideContent(
                slide_number=slide_number,
                title=title,
                bullet_points=key_points,
                citations=[],
                speaker_notes="",
            )

        # Extract citations from bullet points
        citations = []
        citation_pattern = r"\[SourceId:\s*([a-f0-9-]+:\d+)\]"
        for bullet in data.get("bullet_points", []):
            found = re.findall(citation_pattern, bullet)
            citations.extend(found)

        return SlideContent(
            slide_number=slide_number,
            title=data.get("title", title),
            bullet_points=data.get("bullet_points", key_points),
            citations=list(set(citations)),
            speaker_notes=data.get("speaker_notes", ""),
        )

    def to_markdown(self, presentation: Presentation) -> str:
        """Convert presentation to Markdown format."""
        lines = [
            f"# {presentation.title}",
            f"*{presentation.subtitle}*",
            "",
            "---",
            "",
        ]

        for slide in presentation.slides:
            lines.append(f"## Slide {slide.slide_number}: {slide.title}")
            lines.append("")
            for bullet in slide.bullet_points:
                lines.append(f"- {bullet}")
            lines.append("")
            if slide.speaker_notes:
                lines.append(f"> **Speaker Notes:** {slide.speaker_notes}")
                lines.append("")
            lines.append("---")
            lines.append("")

        lines.append(f"*Generated from {presentation.total_sources} sources*")
        return "\n".join(lines)

    def to_json(self, presentation: Presentation) -> dict:
        """Convert presentation to JSON for API response."""
        return {
            "title": presentation.title,
            "subtitle": presentation.subtitle,
            "slides": [
                {
                    "slide_number": s.slide_number,
                    "title": s.title,
                    "bullet_points": s.bullet_points,
                    "citations": s.citations,
                    "speaker_notes": s.speaker_notes,
                }
                for s in presentation.slides
            ],
            "total_sources": presentation.total_sources,
            "generation_steps": presentation.generation_steps,
        }


# Quick test function
async def test_slide_generator():
    """Test the slide generator with a sample topic."""
    from ai_ready_rag.services.rag_service import RAGService

    settings = get_settings()
    rag = RAGService(settings)
    generator = SlideGeneratorService(rag, settings)

    presentation = await generator.generate_presentation(
        topic="Employee benefits and 401k options",
        user_tags=["hr"],
        num_slides=4,
    )

    logger.info(generator.to_markdown(presentation))
    return presentation


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_slide_generator())
