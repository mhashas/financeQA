import chromadb
from chromadb.config import Settings
from chromadb.api.client import Client

from financeqa.settings import chroma_db_settings


class ChromaDBClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the ChromaDB client, applying the singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ChromaDBClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize the ChromaDB client"""
        self.client = chromadb.HttpClient(
            host=chroma_db_settings.db_host,
            port=chroma_db_settings.db_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=chroma_db_settings.db_token.get_secret_value(),
            ),
        )

    def get_client(self) -> Client:  # noqa: F821
        """Get the ChromaDB client

        Returns:
            ChromaDB clientyouy
        """
        return self.client


def get_db_client() -> Client:  # noqa: F821
    """Get the ChromaDB client

    Returns:
        ChromaDB client
    """
    return ChromaDBClient().get_client()
