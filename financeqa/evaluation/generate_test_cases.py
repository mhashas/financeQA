import asyncio
import random
from pathlib import Path

import click
import fitz
import pandas as pd
from fitz import Document
from tqdm import tqdm

from financeqa.constants import MessageType
from financeqa.evaluation.data_models import RetrievalEvalCases
from financeqa.generate import HFChatCompletion
from financeqa.settings import hf_settings


def extract_random_page(doc: Document) -> tuple[int, str]:
    """Extract a random page from the given PDF document

    Args:
        doc: PDF document object

    Returns:
        Tuple containing the page number and the text content of the page
    """
    num_pages = doc.page_count
    random_page_num = random.randint(0, num_pages - 1)
    page = doc.load_page(random_page_num)
    page_text = page.get_text()

    return random_page_num, page_text


def generate_single_document_testcase(
    doc_path: Path, generator: HFChatCompletion, num_tests: int = 5
) -> list[RetrievalEvalCases]:
    """Generate test cases for the retrieval evaluation

    Args:
        doc_path: Path to the doc
        generator: Inference client to generate the completion
        num_tests: Number of test cases to generate for each document

    Returns:
        List of RetrievalEvalCase objects
    """
    doc = fitz.open(doc_path)
    doc_name = doc_path.stem
    year, quarter, company = doc_name.split()

    test_cases = []
    for _ in range(num_tests):
        page_number, page_content = extract_random_page(doc)

        prompt = f"""
                You are an AI generating questions about a specific page from a document from:
                Company: {company}
                Year: {year}
                Quarter: {quarter}
                Page number: {page_number}
                Page content: <page_content>{page_content}</page_content>

                Create 1 concise, natural-sounding questions that closely resemble real user queries. 
                Focus on covering the essential details that would help the retrieval pipeline identify and fetch 
                the most relevant document. Avoid overly generic or vague questions. Do not explicitly mention the page
                number in the question, but ensure that the question is specific to the content on the page. Make sure no
                other information than the one in the page is needed to answer the question.

                As an example, a valid response would be:
                {{
                    "cases": [
                        {{
                            "question": "What was apple's profit in Q3 2022?",
                            "answer": [
                                {{
                                    "year": 2022,
                                    "quarter": "Q3",
                                    "company": "Apple,
                                    "page_number": 19,
                                }}
                            ]
                        }}
                    ]
                }}

                Supply response in this JSON format only:
                {{
                    "cases": [
                        {{
                            "question": string,
                            "answer": [
                                {{
                                    "year": string,
                                    "quarter": string,
                                    "company": string,
                                    "page_number": string,
                                }}
                            ]
                        }}
                    ]
                }}
                """

        messages = [{"role": MessageType.SYSTEM.value, "content": prompt}]

        result = asyncio.run(
            generator.get_completions(
                messages, model_name="meta-llama/Meta-Llama-3-8B-Instruct", response_pydantic_type=RetrievalEvalCases
            )
        )
        test_cases.append(result)

    return test_cases


def save_testcases(test_cases: list[RetrievalEvalCases], output_path: str):
    """Save the test cases to a JSON file

    Args:
        test_cases: List of test cases
        output_path: Path to the output JSON file
    """
    testcases_df = pd.DataFrame([question.model_dump() for case in test_cases for question in case.cases])
    testcases_df.to_json(output_path)


@click.command()
@click.option(
    "--input_root", type=str, default="./data/docs/pdf/", help="path to the directory containing the PDF documents"
)
@click.option(
    "--output_path",
    type=str,
    default="financeqa//evaluation/test_cases/single_document_testcases.json",
    help="path to save the generated test cases",
)
def main(input_root: str, output_path: str):
    docs_root = Path(input_root)
    docs_paths = list(docs_root.glob("*.pdf"))

    generator = HFChatCompletion(api_key=hf_settings.api_key.get_secret_value())

    single_document_testcases = []
    print("Generating single document test cases")
    for doc_path in tqdm(docs_paths):
        doc_test_cases = generate_single_document_testcase(doc_path, generator)
        single_document_testcases.extend(doc_test_cases)

    save_testcases(single_document_testcases, output_path)


if __name__ == "__main__":
    main()
