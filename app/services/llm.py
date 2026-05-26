import logging
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.config import Settings
from utils.retry import retry_sync

logger = logging.getLogger(__name__)

MEDICAL_RAG_SYSTEM = """
You are a medical information assistant.

You must answer ONLY using the retrieved medical context provided.

Rules:
1. Use ONLY information explicitly present in the retrieved context.
2. Do NOT use prior knowledge, assumptions, or external medical information.
3. If the answer cannot be fully determined from the context, respond exactly with:
   "Information not found in the provided medical context."
4. Citations are ONLY valid if they exactly match the format (Chapter <number>, Page <number>); any other format—including document titles, breadcrumbs, section headers, metadata strings, or chunk identifiers—is strictly invalid and must not be used under any circumstances, and if chapter or page information is missing or not explicitly provided in the retrieved context, no citation should be generated.
5. Do NOT invent citations, references, source labels, page numbers, chapter names, or document names. 
6. Never generate generic citations such as "(Document)", "(Source)", or "(Textbook)". If chapter/page metadata is unavailable, do not generate a citation. If the retrieved chunk's breadcrumb contains "# CONTENTS" (e.g., "Document > # CONTENTS"), never use page numbers from that chunk as the source page for medical facts—only for navigation.
7. If multiple provided context chunks support a statement, cite all relevant references from the retrieved context.
8. Keep responses concise, accurate, and medically neutral.
9. Do not provide diagnosis, treatment recommendations, or medical advice unless explicitly stated in the context.
10. If the context contains conflicting information, clearly mention the conflict instead of choosing one answer.
11. Do not summarize beyond what is directly supported by the context.
12. If the retrieved context is incomplete, ambiguous, or unclear, state that the information is unclear based on the provided context.
13. If a retrieved chunk ends mid-sentence, look for the completion of that sentence in the next retrieved chunks before finalizing the response.
14. Do not restate or paraphrase information unless it is directly supported by the retrieved context.
"""

NOT_FOUND_ANSWER = "Information not found in the provided medical context."


def build_medical_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []

    for ctx in contexts:
        page = ctx.get("page")
        chapter = ctx.get("chapter")

        meta_parts = []

        if chapter:
            meta_parts.append(f"Chapter: {chapter}")

        if page is not None:
            meta_parts.append(f"Page: {page}")

        text = ctx.get("text", "").strip()

        if meta_parts:
            meta = " | ".join(meta_parts)
            blocks.append(f"Source [{meta}]:\n{text}")
        else:
            blocks.append(text)

    context_block = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"

    return f"""Retrieved medical context:

{context_block}

---

Question: {question}

Answer based strictly on the context above."""


class GeminiLLM:
    """Medical RAG reasoning via Google Gemini API."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("LLM initialized successfully.")

    def _prompt(self, question: str, contexts: list[dict]) -> str | None:
        if not contexts or not any(c.get("text", "").strip() for c in contexts):
            return None
        return build_medical_prompt(question, contexts)

    def stream_answer(self, question: str, contexts: list[dict]) -> Iterator[str]:
        prompt = self._prompt(question, contexts)
        if prompt is None:
            logger.info("No contexts retrieved for question: %s", question)
            yield NOT_FOUND_ANSWER
            return
        logger.info("LLM prompt built successfully.")
        stream = retry_sync(
            lambda: self._client.models.generate_content_stream(
                model=self._settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MEDICAL_RAG_SYSTEM,
                    temperature=self._settings.llm_temperature,
                ),
            ),
            "gemini.generate_content_stream",
        )
        logger.info("LLM response stream created.")
        try:
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.exception("Gemini stream interrupted mid-response")
            raise RuntimeError(f"LLM stream failed mid-response: {exc}") from exc

    def answer(self, question: str, contexts: list[dict]) -> str:
        parts = list(self.stream_answer(question, contexts))
        return "".join(parts).strip() if parts else NOT_FOUND_ANSWER
