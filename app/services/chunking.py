import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import Settings

CONTEXT_PREFIX = "Context: "


@dataclass
class TextChunk:
    text: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None


def _extract_page(text: str) -> Optional[int]:
    match = re.search(r"(?:page|p\.?)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _build_breadcrumb(meta: dict) -> str:
    parts: list[str] = []
    if meta.get("chapter"):
        parts.append(f"# {meta['chapter']}")
    if meta.get("section"):
        parts.append(f"## {meta['section']}")
    if meta.get("subsection"):
        parts.append(f"### {meta['subsection']}")
    return " > ".join(parts) if parts else "Document"


def _inject_context(body: str, breadcrumb: str) -> str:
    return f"{CONTEXT_PREFIX}{breadcrumb}\n\n{body.strip()}"


def chunk_markdown(markdown: str, settings: Settings) -> List[TextChunk]:
    """Context-injected markdown splitting with header breadcrumbs and table-safe separators."""
    headers = [
        ("#", "chapter"),
        ("##", "section"),
        ("###", "subsection"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    header_splits = md_splitter.split_text(markdown)

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "|", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[TextChunk] = []
    for doc in header_splits:
        body = doc.page_content if hasattr(doc, "page_content") else str(doc)
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        breadcrumb = _build_breadcrumb(meta)
        chapter = meta.get("chapter")
        section = meta.get("section")
        subsection = meta.get("subsection")

        for piece in recursive.split_text(body):
            if not piece.strip():
                continue
            injected = _inject_context(piece, breadcrumb)
            chunks.append(
                TextChunk(
                    text=injected,
                    page=_extract_page(piece),
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                )
            )

    if not chunks:
        for piece in recursive.split_text(markdown):
            if piece.strip():
                chunks.append(
                    TextChunk(
                        text=_inject_context(piece, "Document"),
                        page=_extract_page(piece),
                    )
                )

    return chunks
