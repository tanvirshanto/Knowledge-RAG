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


def clean_text(t: str) -> str:
    return re.sub(r'\W+', '', t).lower()


def _find_page_from_doc(piece: str, cleaned_items: list[tuple[str, str, int]]) -> Optional[int]:
    clean_piece = clean_text(piece)
    if not clean_piece:
        return None

    best_page = None
    longest_match_len = 0

    # Substring matching (fast & covers majority of layout-to-markdown mappings)
    for item_text, clean_item, page in cleaned_items:
        if not clean_item:
            continue
        if clean_item in clean_piece:
            match_len = len(clean_item)
            if match_len > longest_match_len:
                longest_match_len = match_len
                best_page = page
        elif clean_piece in clean_item:
            match_len = len(clean_piece)
            if match_len > longest_match_len:
                longest_match_len = match_len
                best_page = page

    if best_page is not None:
        return best_page

    # Word overlap fallback (for complex tables or split structures)
    page_scores = {}
    words_in_piece = set(re.findall(r'\w+', piece.lower()))
    if words_in_piece:
        for item_text, clean_item, page in cleaned_items:
            words_in_item = set(re.findall(r'\w+', item_text.lower()))
            overlap = len(words_in_piece.intersection(words_in_item))
            if overlap > 0:
                page_scores[page] = page_scores.get(page, 0) + overlap
        if page_scores:
            return max(page_scores, key=page_scores.get)

    return None


def _extract_page(text: str) -> Optional[int]:
    # 1. Standard pattern
    match = re.search(r"(?:page|p\.?)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 2. TOC Pattern: comma followed by a digit at the end of a line
    match_toc = re.search(r",\s*(\d+)\s*$", text, re.MULTILINE)
    if match_toc:
        return int(match_toc.group(1))

    # 3. Index Pattern: digits followed by f, t, or b (figures, tables, boxes)
    match_index = re.search(r"\b(\d+)[ftb]\b", text, re.IGNORECASE)
    if match_index:
        return int(match_index.group(1))

    return None


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


def normalize_markdown_headers(markdown: str) -> str:
    # Ensure there's a space after #, ##, ### if missing
    # E.g. "#PART" -> "# PART", "##INTRODUCTION" -> "## INTRODUCTION"
    # Also ensure there is an empty line before any header line so MarkdownHeaderTextSplitter matches it reliably.
    lines = markdown.splitlines()
    processed_lines = []
    for line in lines:
        # Match lines like #PART or ##INTRODUCTION or ###SUBSECTION
        match = re.match(r"^(#{1,3})([a-zA-Z].*)", line)
        if match:
            line = f"{match.group(1)} {match.group(2)}"
            
        if re.match(r"^#{1,3}\s", line):
            if processed_lines and processed_lines[-1].strip() != "":
                processed_lines.append("")
        processed_lines.append(line)
    return "\n".join(processed_lines)


def chunk_markdown(markdown: str, settings: Settings, doc: Optional[any] = None) -> List[TextChunk]:
    """Context-injected markdown splitting with header breadcrumbs and table-safe separators."""
    markdown = normalize_markdown_headers(markdown)
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

    cleaned_items = []
    if doc and hasattr(doc, "iterate_items"):
        for item in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text and hasattr(item, "label"):
                text = getattr(item, "label", None)
            prov = getattr(item, "prov", None)
            if text and prov and len(prov) > 0:
                page_no = prov[0].page_no
                if page_no is not None:
                    cleaned_items.append((text, clean_text(text), page_no))

    chunks: List[TextChunk] = []
    for doc_split in header_splits:
        body = doc_split.page_content if hasattr(doc_split, "page_content") else str(doc_split)
        meta = doc_split.metadata if hasattr(doc_split, "metadata") else {}
        
        # Absolute reset points for PART and INTRODUCTION
        chapter = meta.get("chapter")
        if chapter:
            chapter_upper = chapter.upper()
            if "PART" in chapter_upper or "INTRODUCTION" in chapter_upper:
                meta["section"] = None
                meta["subsection"] = None

        breadcrumb = _build_breadcrumb(meta)
        chapter = meta.get("chapter")
        section = meta.get("section")
        subsection = meta.get("subsection")

        for piece in recursive.split_text(body):
            if not piece.strip():
                continue
            
            # Prioritize native metadata, then fallback to regex
            page = None
            if cleaned_items:
                page = _find_page_from_doc(piece, cleaned_items)
            if page is None:
                page = _extract_page(piece)

            injected = _inject_context(piece, breadcrumb)
            chunks.append(
                TextChunk(
                    text=injected,
                    page=page,
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                )
            )

    if not chunks:
        for piece in recursive.split_text(markdown):
            if piece.strip():
                page = None
                if cleaned_items:
                    page = _find_page_from_doc(piece, cleaned_items)
                if page is None:
                    page = _extract_page(piece)

                chunks.append(
                    TextChunk(
                        text=_inject_context(piece, "Document"),
                        page=page,
                    )
                )

    return chunks
