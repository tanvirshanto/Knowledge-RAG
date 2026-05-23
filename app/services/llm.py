from app.pipelines.ingestion import logger
from collections.abc import Iterator

from openai import OpenAI

from app.config import Settings
MEDICAL_RAG_SYSTEM = """You are a medical information assistant. Answer ONLY using the retrieved context below.

Rules:
1. Use ONLY facts present in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond with exactly:
   "Information not found in the provided medical context."
3. Cite page/chapter when available from context metadata.
4. End every response with this disclaimer on its own line:

**Medical Disclaimer:** This response is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider."""

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
        blocks.append(f"[Context {i}] ({meta})\n{ctx.get('text', '').strip()}")

    context_block = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"

    return f"""Retrieved medical context:

{context_block}

---

Question: {question}

Answer based strictly on the context above."""


class CommandCodeLLM:
    """Medical RAG reasoning via Command Code Provider API."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.command_code_api_key,
            base_url=settings.command_code_base_url,
        )
        logger.info("LLM initialized successfully.")

    def _messages(self, question: str, contexts: list[dict]) -> list[dict[str, str]] | None:
        if not contexts or not any(c.get("text", "").strip() for c in contexts):
            return None
        return [
            {"role": "system", "content": MEDICAL_RAG_SYSTEM},
            {"role": "user", "content": build_medical_prompt(question, contexts)},
        ]

    def stream_answer(self, question: str, contexts: list[dict]) -> Iterator[str]:
        messages = self._messages(question, contexts)
        if messages is None:
            logger.info("No contexts retrieved for question: %s", question)
            yield NOT_FOUND_ANSWER
            return
        logger.info("LLM prompt built successfully.")
        stream = self._client.chat.completions.create(
            model=self._settings.command_code_model,
            temperature=self._settings.llm_temperature,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        logger.info("LLM response stream created.")
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def answer(self, question: str, contexts: list[dict]) -> str:
        parts = list(self.stream_answer(question, contexts))
        return "".join(parts).strip() if parts else NOT_FOUND_ANSWER
