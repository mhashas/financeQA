import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.documents import Document as LangChainDocument
from langchain_core.vectorstores import VectorStoreRetriever

from financeqa.app.routers.chat.prompt import SYSTEM_MESSAGE
from financeqa.app.schema import Message, ReferencedResponse
from financeqa.app.security import get_current_api_key
from financeqa.constants import DOC_ROOT, TOP_K, MessageType
from financeqa.db.vector_store import get_vs
from financeqa.generate.hf_inference import HFChatCompletion
from financeqa.retrieval.metadata_filtering import generate_combined_search_kwargs  # type: ignore
from financeqa.retrieval.metadata_filtering import extract_search_kwargs, generate_separated_kwargs
from financeqa.settings import hf_settings
from financeqa.utils import format_docs_for_context, get_document_ids

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

router = APIRouter()
vector_store = get_vs()
generator = HFChatCompletion(hf_settings.api_key.get_secret_value())
doc_ids = get_document_ids(
    doc_root=DOC_ROOT
)  # depending the usage of the app, this can be moved to be dynamically retrieved


async def get_context_documents(query: str) -> list[LangChainDocument]:
    """Get context documents for the query

    Args:
        query: the query

    Returns:
        the context documents
    """
    logger.info("Extracting search kwargs.")
    extracted_search_kwargs = await extract_search_kwargs(doc_ids, query, generator=generator)
    logger.debug(f"Extracted search kwargs: {extracted_search_kwargs}")

    logger.info("Generating separated kwargs for Chroma search.")
    chroma_search_kwargs = generate_separated_kwargs(extracted_search_kwargs)
    logger.debug(f"Chroma search kwargs: {chroma_search_kwargs}")

    individual_k = TOP_K // len(chroma_search_kwargs) + 1
    logger.debug(f"Calculated individual_k: {individual_k}")

    context_docs = []
    for search_kwargs in chroma_search_kwargs:
        search_kwargs["k"] = individual_k
        retriever = VectorStoreRetriever(vectorstore=vector_store, search_kwargs=search_kwargs)

        logger.info(f"Invoking retriever for search kwargs: {search_kwargs}")
        documents = retriever.invoke(query)
        # documents = retriever.similarity_search_with_score(query, k=individual_k, **search_kwargs)
        logger.debug(f"Retrieved documents: {documents}")
        context_docs.extend(documents)

    # ideally we would filter, rank and return the top k documents here
    logger.debug(f"Final documents: {documents}")
    return context_docs


@router.post(
    "/query",
    response_model=ReferencedResponse,
    dependencies=[Depends(get_current_api_key)],
)
async def query(message_input: list[Message]):
    # Log input messages
    logger.debug(f"Received input messages: {message_input}")

    # Will only use the last message for now due to time constraints
    last_message = message_input[-1].message
    logger.debug(f"Processing last message: {last_message}")

    try:
        context_docs = await get_context_documents(last_message)
        docs_string = format_docs_for_context(context_docs)
        logger.debug(f"Formatted documents string: {docs_string[:500]}")  # Log only the first 500 characters

        prompt = SYSTEM_MESSAGE.format(context=docs_string)
        messages = [
            {"role": MessageType.SYSTEM.value, "content": prompt},
            {"role": MessageType.USER.value, "content": last_message},
        ]
        logger.debug(f"Generated messages: {messages}")

        logger.info("Sending messages to generator for completions.")
        response = await generator.get_completions(
            messages, model_name="meta-llama/Meta-Llama-3-8B-Instruct", response_pydantic_type=ReferencedResponse
        )
        logger.info("Response successfully generated.")
        logger.debug(f"Generator response: {response}")

        return response

    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred") from e
