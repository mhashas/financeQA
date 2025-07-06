import asyncio

import click
import pandas as pd
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from tqdm import tqdm

from financeqa.app.schema import ReferencedDoc
from financeqa.constants import DOC_ROOT, TOP_K
from financeqa.db.vector_store import get_vs
from financeqa.generate.hf_inference import HFChatCompletion
from financeqa.retrieval.metadata_filtering import extract_search_kwargs, generate_combined_search_kwargs
from financeqa.settings import hf_settings
from financeqa.utils import get_document_ids


def evaluate_simple_retriever(test_cases_df: pd.DataFrame, vs: VectorStore):
    retriever = VectorStoreRetriever(vectorstore=vs, search_kwargs={"k": TOP_K})

    # some simple metrics for now
    score = 0
    num_relevant_docs = 0
    for _, row in tqdm(test_cases_df.iterrows(), total=len(test_cases_df)):
        question = row["question"]
        answers = [ReferencedDoc.model_validate(answer) for answer in row["answer"]]

        retrieved_documents = retriever.invoke(question)
        processed_returned_docs = []
        for doc in retrieved_documents:
            case = ReferencedDoc(
                year=doc.metadata["year"],
                quarter=doc.metadata["quarter"],
                company=doc.metadata["ticker"],
                page=doc.metadata["page_number"],
            )
            processed_returned_docs.append(case)

        matches = [
            (obj1, obj2)
            for obj1 in answers
            for obj2 in processed_returned_docs
            if obj1.model_dump() == obj2.model_dump()
        ]
        score += 1 if len(matches) > 0 else 0
        num_relevant_docs += len(matches)

    return {"score": score, "num_relevant_docs": num_relevant_docs}


def evaluate_metadatafiltering_retriever(test_cases_df: pd.DataFrame, vs: VectorStore):
    # some simple metrics for now
    score = 0
    num_relevant_docs = 0
    generator = HFChatCompletion(hf_settings.api_key.get_secret_value())
    doc_ids = get_document_ids(
        doc_root=DOC_ROOT
    )  # depending the usage of the app, this can be moved to be dynamically retrieved

    for _, row in tqdm(test_cases_df.iterrows(), total=len(test_cases_df)):
        question = row["question"]
        answers = [ReferencedDoc.model_validate(answer) for answer in row["answer"]]

        extracted_search_kwargs = asyncio.run(extract_search_kwargs(doc_ids, question, generator=generator))
        chroma_search_kwargs = generate_combined_search_kwargs(
            extracted_search_kwargs
        )  # same as separated in this case, since we only have one ticker year quarter combination
        chroma_search_kwargs["k"] = TOP_K
        retriever = VectorStoreRetriever(vectorstore=vs, search_kwargs=chroma_search_kwargs)
        retrieved_documents = retriever.invoke(question)

        processed_returned_docs = []
        for doc in retrieved_documents:
            case = ReferencedDoc(
                year=doc.metadata["year"],
                quarter=doc.metadata["quarter"],
                company=doc.metadata["ticker"],
                page=doc.metadata["page_number"],
            )
            processed_returned_docs.append(case)

        matches = [
            (obj1, obj2)
            for obj1 in answers
            for obj2 in processed_returned_docs
            if obj1.model_dump() == obj2.model_dump()
        ]
        score += 1 if len(matches) > 0 else 0
        num_relevant_docs += len(matches)

    return {"score": score, "num_relevant_docs": num_relevant_docs}


@click.command()
@click.option(
    "--test_cases_json_path",
    type=str,
    default="financeqa/evaluation/test_cases/single_document_testcases.json",
    help="path to the json containing the testcases",
)
def main(test_cases_json_path: str):
    test_cases_df = pd.read_json(test_cases_json_path)
    vs = get_vs()

    metrics = evaluate_simple_retriever(test_cases_df, vs)
    print("Metrics for simple retriever:", metrics)

    metrics = evaluate_metadatafiltering_retriever(test_cases_df, vs)
    print("Metrics for metadata filtering retriever:", metrics)


if __name__ == "__main__":
    main()
