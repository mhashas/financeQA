import torch
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from financeqa.constants import COLLECTION_NAME
from financeqa.db.db import get_db_client


class VectorStoreClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the VectorStoreClient, applying the singleton pattern"""
        if cls._instance is None:
            cls._instance = super(VectorStoreClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize the VectorStoreClient"""
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs["device"] = "cuda"

        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        db = get_db_client()
        vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=db)

        self.client = vector_store

    def get_client(self) -> VectorStore:  # noqa: F821
        """Get the VectorStoreClient

        Returns:
            the vector store client
        """
        return self.client


def get_vs() -> VectorStore:  # noqa: F821
    """Get the vector store client

    Returns:
        vector store client
    """
    return VectorStoreClient().get_client()
