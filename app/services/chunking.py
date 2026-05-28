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


def _find_page_from_doc(piece: str, cleaned_items: list[tuple[str, str, int, int]]) -> tuple[Optional[int], Optional[int]]:
    clean_piece = clean_text(piece)
    if not clean_piece:
        return None, None

    best_page = None
    best_phys_page = None
    longest_match_len = 0

    # Substring matching (fast & covers majority of layout-to-markdown mappings)
    for item_text, clean_item, page, phys_page in cleaned_items:
        if not clean_item:
            continue
        if clean_item in clean_piece:
            match_len = len(clean_item)
            if match_len > longest_match_len:
                longest_match_len = match_len
                best_page = page
                best_phys_page = phys_page
        elif clean_piece in clean_item:
            match_len = len(clean_piece)
            if match_len > longest_match_len:
                longest_match_len = match_len
                best_page = page
                best_phys_page = phys_page

    if best_page is not None:
        return best_page, best_phys_page

    # Word overlap fallback (for complex tables or split structures)
    page_scores = {}
    phys_page_scores = {}
    words_in_piece = set(re.findall(r'\w+', piece.lower()))
    if words_in_piece:
        for item_text, clean_item, page, phys_page in cleaned_items:
            words_in_item = set(re.findall(r'\w+', item_text.lower()))
            overlap = len(words_in_piece.intersection(words_in_item))
            if overlap > 0:
                page_scores[page] = page_scores.get(page, 0) + overlap
                phys_page_scores[phys_page] = phys_page_scores.get(phys_page, 0) + overlap
        if page_scores:
            best_page = max(page_scores, key=page_scores.get)
            best_phys_page = max(phys_page_scores, key=phys_page_scores.get)
            return best_page, best_phys_page

    return None, None


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


def trim_to_sentence_boundary(text: str) -> str:
    """Trim a chunk of text back to the last sentence boundary if it ends mid-sentence."""
    text = text.strip()
    if not text:
        return text

    # Do not trim markdown tables or code blocks
    if "|" in text or "```" in text:
        return text

    # Check if it already ends with sentence-ending punctuation or brackets
    if text[-1] in {".", "!", "?", '"', "'", "”", "’", ")"}:
        return text

    # Find sentence boundaries: period/exclamation/question mark followed by space or newline, or a newline
    matches = list(re.finditer(r"(?:[.\!?]\s+|\n)", text))
    if matches:
        last_match = matches[-1]
        end_idx = last_match.end()
        trimmed = text[:end_idx].strip()
        # Keep the trimmed text only if we don't discard too much (at least 40% length and > 50 chars)
        if len(trimmed) >= len(text) * 0.4 and len(trimmed) > 50:
            return trimmed

    return text


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
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    cleaned_items = []
    if doc and hasattr(doc, "iterate_items"):
        from docling_core.types.doc.document import ContentLayer
        
        # Build logical page and chapter mappings using furniture layout elements (running headers/footers)
        page_map = {}
        chapter_map = {}
        # First pass: look for standalone digits (highest priority, e.g. "219") and running header/chapter markers
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE}):
            prov = getattr(item, "prov", None)
            if not prov or len(prov) == 0:
                continue
            phys_page = prov[0].page_no
            if phys_page is None:
                continue
            text = getattr(item, "text", None)
            if text:
                cleaned_text = text.strip()
                if re.match(r"^\d+$", cleaned_text):
                    page_map[phys_page] = int(cleaned_text)
                elif re.search(r"[a-zA-Z]", cleaned_text):
                    # Clean page numbers and formatting pipes to extract raw chapter/section name
                    cleaned_chapter = re.sub(r'\b\d+\b\s*\|\s*', '', cleaned_text)
                    cleaned_chapter = re.sub(r'\s*\|\s*\b\d+\b', '', cleaned_chapter)
                    cleaned_chapter = re.sub(r'\s+\d+$', '', cleaned_chapter)
                    cleaned_chapter = re.sub(r'^\d+\s+', '', cleaned_chapter)
                    cleaned_chapter = cleaned_chapter.strip(" |")
                    if len(cleaned_chapter) > 3:
                        chapter_map[phys_page] = cleaned_chapter

        # Second pass: fallback to trailing digits (e.g. "CHAPTER 16 ... 219" or "66 | CHRONIC ASPIRATION    967")
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE}):
            prov = getattr(item, "prov", None)
            if not prov or len(prov) == 0:
                continue
            phys_page = prov[0].page_no
            if phys_page is None:
                continue
            text = getattr(item, "text", None)
            if text:
                cleaned_text = text.strip()
                if phys_page not in page_map:
                    match = re.search(r"\b(\d+)\b$", cleaned_text)
                    if match:
                        page_map[phys_page] = int(match.group(1))
                if phys_page not in chapter_map and re.search(r"[a-zA-Z]", cleaned_text):
                    cleaned_chapter = re.sub(r'\b\d+\b\s*\|\s*', '', cleaned_text)
                    cleaned_chapter = re.sub(r'\s*\|\s*\b\d+\b', '', cleaned_chapter)
                    cleaned_chapter = re.sub(r'\s+\d+$', '', cleaned_chapter)
                    cleaned_chapter = re.sub(r'^\d+\s+', '', cleaned_chapter)
                    cleaned_chapter = cleaned_chapter.strip(" |")
                    if len(cleaned_chapter) > 3:
                        chapter_map[phys_page] = cleaned_chapter

        # Now extract text content from body layer and map physical page numbers to logical page numbers
        for item, _ in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text and hasattr(item, "label"):
                text = getattr(item, "label", None)
            prov = getattr(item, "prov", None)
            if text and prov and len(prov) > 0:
                page_no = prov[0].page_no
                if page_no is not None:
                    logical_page = page_map.get(page_no, page_no)
                    cleaned_items.append((text, clean_text(text), logical_page, page_no))

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

        section = meta.get("section")
        subsection = meta.get("subsection")

        for piece in recursive.split_text(body):
            piece = trim_to_sentence_boundary(piece)
            if not piece:
                continue
            
            # Prioritize native metadata, then fallback to regex
            page = None
            phys_page = None
            if cleaned_items:
                page, phys_page = _find_page_from_doc(piece, cleaned_items)
            if page is None:
                page = _extract_page(piece)

            # Resolve logical chapter if missing
            chunk_chapter = chapter
            if not chunk_chapter and phys_page and phys_page in chapter_map:
                chunk_chapter = chapter_map[phys_page]

            # Build chunk-specific breadcrumb
            chunk_meta = {
                "chapter": chunk_chapter,
                "section": section,
                "subsection": subsection,
            }
            chunk_breadcrumb = _build_breadcrumb(chunk_meta)

            injected = _inject_context(piece, chunk_breadcrumb)
            chunks.append(
                TextChunk(
                    text=injected,
                    page=page,
                    chapter=chunk_chapter,
                    section=section,
                    subsection=subsection,
                )
            )

    if not chunks:
        for piece in recursive.split_text(markdown):
            piece = trim_to_sentence_boundary(piece)
            if not piece:
                continue

            page = None
            phys_page = None
            if cleaned_items:
                page, phys_page = _find_page_from_doc(piece, cleaned_items)
            if page is None:
                page = _extract_page(piece)

            chunk_chapter = chapter_map.get(phys_page) if (phys_page and phys_page in chapter_map) else None
            chunk_meta = {"chapter": chunk_chapter}
            chunk_breadcrumb = _build_breadcrumb(chunk_meta)

            chunks.append(
                TextChunk(
                    text=_inject_context(piece, chunk_breadcrumb),
                    page=page,
                    chapter=chunk_chapter,
                )
            )

    return chunks
