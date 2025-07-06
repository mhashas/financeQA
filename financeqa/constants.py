from enum import Enum


class MessageType(Enum):
    USER = "user"
    SYSTEM = "system"


class OpenAIProvider(Enum):
    OPENAI = "openai"
    OPENAI_AZURE = "openai_azure"


class TextExtractionType(Enum):
    PDF = "pdf"
    PDF_MARKDOWN = "pdf_markdown"


DB_DOC_NAME_KEY = "db_document_name"
DOC_ROOT = "./data/docs/pdf/"
COLLECTION_NAME = "financeqa-documents"
NODE_PARSER_CHUNK_SIZE = 512
NODE_PARSER_CHUNK_OVERLAP = 10
TOP_K = 9
