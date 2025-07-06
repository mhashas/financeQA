from pathlib import Path

from langchain_core.documents import Document as LangChainDocument


def get_document_ids(doc_root: str) -> list[str]:
    """Get the document IDs from the document root

    Args:
        doc_root: the document root

    Returns:
        the document IDs
    """
    docs_root = Path(doc_root)
    docs_paths = list(docs_root.glob("*.pdf"))
    return [(doc_path).stem for doc_path in docs_paths]


def format_docs_for_context(docs: list[LangChainDocument]) -> str:
    """Format the documents for the context

    Args:
        docs: the documents

    Returns:
        the formatted documents
    """
    return "\n\n".join(
        doc.page_content + f"\n Reference: {doc.metadata['db_document_name']}, page {doc.metadata['page_number']}"
        for doc in docs
    )
