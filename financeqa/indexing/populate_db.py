import functools
from pathlib import Path
from typing import Any, Callable

import click
import fitz
import pandas as pd
import torch
from fitz import Document, Page
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from financeqa.constants import (
    COLLECTION_NAME,
    DB_DOC_NAME_KEY,
    NODE_PARSER_CHUNK_OVERLAP,
    NODE_PARSER_CHUNK_SIZE,
    TextExtractionType,
)
from financeqa.db.db import get_db_client
from financeqa.preprocessing.text.text_extraction import build_text_extractor

ticker_to_name_map = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "INTC": "Intel",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
}


def get_precomputed_feature(page: Page, df: pd.DataFrame):
    document_name = Path(page.parent.name).stem
    image_name = str(Path(document_name) / f"{page.number}")

    matching_rows = df[df["image_name"].str.startswith(image_name)]
    summary = matching_rows["summary"].str.cat(sep="\n")

    return summary


def extract_from_document(doc: Document, extractors: dict[str, Callable[[Page], Any]]) -> list[LangChainDocument]:
    """Extract information from a PDF document using the provided extractors.

    Args:
        doc: PDF document to extract from
        extractors: dictionary of extractors to use

    Returns:
        list of LangChainDocuments containing the extracted information
    """
    result = []
    doc_name = Path(doc.name).stem
    year, quarter, ticker = doc_name.split()
    company_name = ticker_to_name_map[ticker]
    metadata = {
        DB_DOC_NAME_KEY: Path(doc.name).stem,
        "year": int(year),
        "quarter": quarter,
        "ticker": ticker,
        "company": company_name,
    }

    for i, page in enumerate(doc.pages()):
        metadata["page_number"] = int(page.number)

        for key, extractor_func in extractors.items():
            metadata["info_type"] = key
            info = extractor_func(page)

            if info is not None:
                result.append(LangChainDocument(page_content=info, metadata=metadata))

    return result


def process_documents(
    input_dir: str,
    extractors: dict[str, Callable[[Page], Any]],
):
    docs_root = Path(input_dir)
    docs_paths = list(docs_root.glob("*.pdf"))
    db = get_db_client()

    print("Deleting existing collection and documents")
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception as e:
        pass

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["device"] = "cuda"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=NODE_PARSER_CHUNK_SIZE, chunk_overlap=NODE_PARSER_CHUNK_OVERLAP
    )

    for doc_path in tqdm(docs_paths):
        doc = fitz.open(doc_path)
        langchain_documents = extract_from_document(doc, extractors)

        chunked_docs = []
        for document in langchain_documents:
            chunks = text_splitter.split_text(document.page_content)
            chunked_docs.extend([LangChainDocument(page_content=chunk, metadata=document.metadata) for chunk in chunks])

        Chroma.from_documents(
            chunked_docs,
            embeddings,
            collection_name=COLLECTION_NAME,
            client=db,
            collection_metadata={"hnsw:space": "cosine"},
        )


@click.command()
@click.option("--input_dir", default="./data/docs/pdf/")
@click.option("--extract_text", type=bool, default=True)
@click.option("--extract_images", type=bool, default=True)
@click.option("--images_csv_path", type=str, default="./data/images_summaries.csv")
@click.option("--extract_tables", type=bool, default=True)
@click.option("--tables_csv_path", type=str, default="./data/tables_summaries.csv")
@click.option(
    "--text_extraction_type",
    type=click.Choice([type.value for type in TextExtractionType]),
    default=TextExtractionType.PDF.value,
)
@click.option(
    "--summarize",
    type=bool,
    default=False,
    help="Summarize the extracted text and embed alongside the document",
)
@click.option(
    "--generate_hypothetical_questions",
    type=bool,
    default=False,
    help="Generate hypothetical questions and embed alongside the document",
)
def main(
    input_dir: str,
    extract_text: bool,
    extract_images: bool,
    images_csv_path: str,
    extract_tables: bool,
    tables_csv_path: str,
    text_extraction_type: str,
    summarize: bool,
    generate_hypothetical_questions: bool,
):
    if generate_hypothetical_questions:
        raise NotImplementedError("Hypothetical question generation is not yet implemented")

    if summarize:
        raise NotImplementedError("Summarization is not yet implemented")

    extractors = {}
    if extract_text:
        text_extractor = build_text_extractor(text_extraction_type)
        extractors["text"] = text_extractor

    if extract_images:
        images_df = pd.read_csv(images_csv_path)
        extractor_func = functools.partial(get_precomputed_feature, df=images_df)
        extractors["image"] = extractor_func

    if extract_tables:
        tables_df = pd.read_csv(tables_csv_path)

        # ideally this step should be done in the preprocessing step, due to lack of time, we are doing it here
        tables_df = tables_df[~tables_df["summary"].str.startswith("Error: No table found in the image.").fillna(True)]

        extractor_func = functools.partial(get_precomputed_feature, df=tables_df)
        extractors["table"] = extractor_func

    process_documents(input_dir, extractors)


if __name__ == "__main__":
    main()
