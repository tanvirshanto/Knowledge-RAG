"""
Module 3: Grounded Retrieval Engine (Managed)
Generates grounded answers using Vertex AI Managed RAG.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai import rag

from google.cloud.aiplatform_v1beta1 import VertexRagServiceClient
from google.cloud.aiplatform_v1beta1.types.vertex_rag_service import (
    RetrieveContextsRequest,
    RagQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundedAnswer:
    """Structured response from grounded retrieval."""
    
    answer: str
    citations: List[dict]
    grounding_chunks: List[dict]
    grounding_supports: List[dict]
    raw_response: object


class RetrievalError(Exception):
    """Raised when retrieval fails."""
    pass


class RetrievalEngine:
    """
    Generates grounded answers using Vertex AI Managed RAG.
    """
    
    def __init__(
        self,
        project_id: str,
        location: str,
        corpus_id: str,
        similarity_top_k: int = 5,
    ):
        self.project_id = project_id
        self.location = location
        self.corpus_id = corpus_id
        self.similarity_top_k = similarity_top_k
        
        # Initialize high-level Vertex AI SDK
        vertexai.init(project=project_id, location=location)
        
        # Initialize RPC client for search-only retrieval
        self._service_client = VertexRagServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
    
    @property
    def rag_corpus_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.corpus_id}"
    
    def ask(self, question: str) -> GroundedAnswer:
        """
        MANAGED: Ask a question and get a grounded answer in one API call.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            # 1. Configure the RAG retrieval tool
            # 1. Define your Strict Medical Instruction
            strict_medical_instruction = """
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

            # 2. Configure RAG Tool
            rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus_path)],
                        rag_retrieval_config=rag.RagRetrievalConfig(top_k=self.similarity_top_k),
                    )
                )
            )
            
            # 2. Get LLM settings to select model name (fallback to default if not configured)
            from app.config import get_settings
            try:
                settings = get_settings()
                model_name = settings.gemini_model or "gemini-2.5-flash"
            except Exception:
                model_name = "gemini-2.5-flash"
            
            # 3. Create GenerativeModel with RAG tool
            model = GenerativeModel(
                model_name=model_name,
                tools=[rag_retrieval_tool],
                system_instruction=strict_medical_instruction
            )
            
            # 4. Generate grounded content
            response = model.generate_content(question)
            return self._parse_managed_response(response)
            
        except Exception as e:
            raise RetrievalError(f"Managed generation failed: {e}") from e
    
    def _parse_managed_response(self, response) -> GroundedAnswer:
        """Parses response with special handling for strict medical metadata."""
        candidate = response.candidates[0]
        answer_text = candidate.content.parts[0].text
        metadata = getattr(candidate, "grounding_metadata", None)
        
        # Rule 3 Check: Detect 'Information not found' state
        is_not_found = "Information not found in the provided medical context" in answer_text
        
        citations = []
        grounding_chunks = []
        grounding_supports = []
        
        if metadata and not is_not_found:
            # 1. Parse Grounding Chunks & Metadata (using correct nested attribute access)
            for chunk in getattr(metadata, "grounding_chunks", []):
                chunk_dict = {}
                retrieved = getattr(chunk, "retrieved_context", None)
                if retrieved:
                    chunk_dict["source_uri"] = retrieved.uri
                    chunk_dict["title"] = retrieved.title
                    inner_chunk = getattr(retrieved, "chunk", None)
                    if inner_chunk:
                        chunk_dict["text"] = inner_chunk.text
                        page_span = getattr(inner_chunk, "page_span", None)
                        if page_span:
                            first_page = getattr(page_span, "first_page", None)
                            if first_page is not None:
                                chunk_dict["page"] = first_page
                else:
                    web = getattr(chunk, "web", None)
                    if web:
                        chunk_dict["source_uri"] = web.uri
                        chunk_dict["title"] = web.title
                        chunk_dict["text"] = getattr(web, "title", "")
                
                # Extract Chapter (Regex from Title/URI per requirement)
                source_text = f"{chunk_dict.get('source_uri', '')}"
                chapter_match = re.search(r"chapter\s*(\d+)", source_text, re.IGNORECASE)
                if chapter_match:
                    chunk_dict["chapter"] = chapter_match.group(1)
                
                grounding_chunks.append(chunk_dict)
            
            # 2. Parse Supports (Indices for highlighting)
            for support in getattr(metadata, "grounding_supports", []):
                grounding_supports.append({
                    "text": support.segment.text,
                    "start_index": support.segment.start_index,
                    "end_index": support.segment.end_index,
                    "grounding_chunk_indices": list(support.grounding_chunk_indices)
                })

            # 3. Create Structured Citations (Mapping Rule #4/7)
            for support in grounding_supports:
                sources = []
                for idx in support["grounding_chunk_indices"]:
                    if idx < len(grounding_chunks):
                        c = grounding_chunks[idx]
                        # Only add citation if Chapter/Page exists (per Rule 4/6)
                        if "page" in c or "chapter" in c:
                            sources.append({
                                "uri": c["source_uri"],
                                "page": c.get("page"),
                                "chapter": c.get("chapter")
                            })
                
                if sources:
                    citations.append({
                        "start_index": support["start_index"],
                        "end_index": support["end_index"],
                        "sources": sources
                    })
        
        return GroundedAnswer(
            answer=answer_text,
            citations=citations,
            grounding_chunks=grounding_chunks,
            grounding_supports=grounding_supports,
            raw_response=response,
        )

    def retrieve_contexts(self, question: str, top_k: Optional[int] = None) -> List[dict]:
        """
        Retrieves raw medical snippets from the RAG Corpus for search or debugging.
        """
        k = top_k or self.similarity_top_k
        
        try:
            # 1. Build the Search Request
            request = RetrieveContextsRequest(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                vertex_rag_store=RetrieveContextsRequest.VertexRagStore(
                    rag_resources=[
                        RetrieveContextsRequest.VertexRagStore.RagResource(
                            rag_corpus=self.rag_corpus_path
                        )
                    ]
                ),
                query=RagQuery(
                    text=question,
                    similarity_top_k=k,
                ),
            )
            
            # 2. Execute the Search
            response = self._service_client.retrieve_contexts(request=request)
            
            # 3. Parse findings into a structured list
            contexts = []
            for ctx in response.contexts.contexts:
                # Basic context data
                context_dict = {
                    "text": ctx.text,
                    "source_uri": ctx.source_uri,
                    "relevance_score": getattr(ctx, "score", 0.0),
                }
                
                # Rule 4 & 5 Support: Extract specific Page Number from the Chunk metadata
                chunk_metadata = getattr(ctx, "chunk", None)
                if chunk_metadata:
                    page_span = getattr(chunk_metadata, "page_span", None)
                    if page_span:
                        first_page = getattr(page_span, "first_page", None)
                        if first_page is not None:
                            context_dict["page"] = first_page
                
                # Extract Chapter from the title/uri if it follows a pattern like "Chapter 1"
                import re
                source_name = getattr(ctx, "source_display_name", "") or ctx.source_uri or ""
                chapter_match = re.search(r"chapter\s*(\d+)", source_name, re.IGNORECASE)
                if chapter_match:
                    context_dict["chapter"] = chapter_match.group(1)
                
                contexts.append(context_dict)
            
            return contexts
            
        except Exception as e:
            raise RetrievalError(f"Raw context retrieval failed: {e}") from e

    @classmethod
    def from_config(cls, config) -> "RetrievalEngine":
        return cls(
            project_id=config.project_id,
            location=config.location,
            corpus_id=config.corpus_id,
            similarity_top_k=config.similarity_top_k,
        )
