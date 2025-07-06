from dotenv import find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

env_file = find_dotenv()


class OpenAIAzureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")
    endpoint: str = "https://testblblabla.openai.azure.com/"
    api_version: str = "2024-05-01-preview"
    api_key: Optional[SecretStr] = Field(None, alias="OPENAI_AZURE_KEY")


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")
    api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")
    api_key: Optional[SecretStr] = Field(None, alias="HF_API_KEY")


class ChromaDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")
    db_host: str = Field(..., alias="CHROMA_DB_HOST")
    db_port: int = Field(..., alias="CHROMA_DB_PORT")
    db_token: SecretStr = Field(..., alias="CHROMA_DB_TOKEN")


openai_azure_settings = OpenAIAzureSettings()  # type: ignore
hf_settings = HuggingFaceSettings()  # type: ignore
openai_settings = OpenAISettings()  # type: ignore
chroma_db_settings = ChromaDBSettings()  # type: ignore
