from app.pipelines.ingestion import logger
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.config import Settings

MEDICAL_RAG_SYSTEM = """
You are a medical information assistant.

You must answer ONLY using the retrieved medical context provided.

Rules:
1. Use ONLY information explicitly present in the retrieved context.
2. Do NOT use prior knowledge, assumptions, or external medical information.
3. If the answer cannot be fully determined from the context, respond exactly with:
   "Information not found in the provided medical context."
4. Cite the actual chapter and page inline for every factual statement using this format:
   "(Chapter 3, Page 12)"
5. Do NOT invent citations, references, or page numbers.
6. If multiple provided context chunks support a statement, cite all relevant references from the retrieved context.
7. Keep responses concise, accurate, and medically neutral.
8. Do not provide diagnosis, treatment recommendations, or medical advice unless explicitly stated in the context.
9. If the context contains conflicting information, clearly mention the conflict instead of choosing one answer.
10. Do not summarize beyond what is directly supported by the context.
11. If the retrieved context is incomplete, ambiguous, or unclear, state that the information is unclear based on the provided context.
"""

NOT_FOUND_ANSWER = "Information not found in the provided medical context."


def build_medical_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        page = ctx.get("page")
        chapter = ctx.get("chapter")
        meta_parts = []
        if chapter:
            meta_parts.append(f"Chapter: {chapter}")
        if page is not None:
            meta_parts.append(f"Page: {page}")
        meta = " | ".join(meta_parts) if meta_parts else "Source metadata unavailable"
        blocks.append(f"Source [{meta}]:\n{ctx.get('text', '').strip()}")

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
        stream = self._client.models.generate_content_stream(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_RAG_SYSTEM,
                temperature=self._settings.llm_temperature,
            ),
        )
        logger.info("LLM response stream created.")
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def answer(self, question: str, contexts: list[dict]) -> str:
        parts = list(self.stream_answer(question, contexts))
        return "".join(parts).strip() if parts else NOT_FOUND_ANSWER
