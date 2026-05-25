from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.config import Settings

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        pipeline_options.ocr_options = RapidOcrOptions()
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
    return _converter


def parse_pdf_to_markdown(pdf_path: Path, settings: Settings) -> tuple[str, any]:
    """Layout-aware PDF parsing with OCR and table preservation via Docling."""
    _ = settings  # reserved for future parse options
    converter = _get_converter()
    result = converter.convert(str(pdf_path))

    if result.status != ConversionStatus.SUCCESS:
        raise ValueError(
            f"Docling failed to parse PDF (status={result.status})."
        )

    markdown = result.document.export_to_markdown()
    if not markdown or not markdown.strip():
        raise ValueError("Parsed markdown is empty.")
    return markdown.strip(), result.document
