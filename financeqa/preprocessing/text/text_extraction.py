from typing import Protocol

import fitz
import pymupdf4llm
from fitz import Page

from financeqa.constants import TextExtractionType


class PdfTextExtractor(Protocol):

    def __call__(self, page: Page) -> str: ...


class SimplePdfTextExtractor:
    def __call__(self, page: Page) -> str:
        """Extract the text directly from the PDF page

        Args:
            page: PDF page

        Returns:
            extracted text
        """
        return page.get_text()  # type: ignore


class MarkdownPdfTextExtractor:
    def __call__(self, page: Page) -> str:
        """Extract the text from the PDF page and convert it to Markdown

        Args:
            page: PDF page

        Returns:
            extracted text in Markdown format
        """
        new_doc = fitz.Document()
        new_doc.insert_page(-1, from_page=page)  # type: ignore
        new_doc.close()

        return pymupdf4llm.to_markdown(new_doc)


def build_text_extractor(text_extraction_type: str) -> PdfTextExtractor:
    """Get the text extractor based on the text extraction type

    Args:
        text_extraction_type: type of text extraction

    Returns:
        text extractor
    """
    extractors_dict = {
        TextExtractionType.PDF_MARKDOWN.value: MarkdownPdfTextExtractor,
        TextExtractionType.PDF.value: SimplePdfTextExtractor,
    }
    extractor = extractors_dict.get(text_extraction_type, None)

    if not extractor:
        raise ValueError(f"Unknown text extraction type: {text_extraction_type}")

    return extractor()
