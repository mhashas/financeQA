from typing import Any

from pydantic import BaseModel

from financeqa.constants import MessageType
from financeqa.generate.base_inference_client import BaseInferenceClient


class MetadataExtractor(BaseModel):
    year: int
    quarter: str
    ticker: str


class Response(BaseModel):
    list_of_docs: list[MetadataExtractor]


def generate_combined_search_kwargs(response: Response) -> dict[str, Any]:
    """Generate search kwargs from the response

    Args:
        response: the LLMM response

    Returns:
        the search kwargs
    """
    if len(response.list_of_docs) == 0:
        return {}

    filter_conditions = {"$and": []}

    fields = ["year", "quarter", "ticker"]

    conditions = {field: [] for field in fields}

    for doc in response.list_of_docs:
        for field in fields:
            value = getattr(doc, field)
            if not any(condition.get(field, {}).get("$eq") == value for condition in conditions[field]):
                conditions[field].append({field: {"$eq": value}})

    for field in fields:
        if len(conditions[field]) > 1:
            filter_conditions["$and"].append({"$or": conditions[field]})
        elif len(conditions[field]) == 1:
            filter_conditions["$and"].append(conditions[field][0])

    return {"filter": filter_conditions}


def generate_separated_kwargs(response: Response) -> list[dict[str, Any]]:
    """Generate search kwargs from the response

    Args:
        response: the LLMM response

    Returns:
        a list of dictionaries with a "filter" key, each containing filter conditions
    """
    if len(response.list_of_docs) == 0:
        return []

    filter_conditions_list = []

    fields = ["year", "quarter", "ticker"]

    for doc in response.list_of_docs:
        # Create a list of filter conditions
        conditions = []

        for field in fields:
            value = getattr(doc, field)
            conditions.append({field: {"$eq": value}})

        # Wrap the conditions in a $and
        filter_conditions_list.append({"filter": {"$and": conditions}})

    return filter_conditions_list


async def extract_search_kwargs(doc_ids: list, query: str, generator: BaseInferenceClient) -> Response:
    """
    Extract search kwargs to filter the vector store based on the user query

    Args:
        doc_ids: list of document IDs
        query: user query
        generator: inference client

    Returns:
        search kwargs to filter the vector store
    """
    extract_template = f"""
A user has questions regarding some documents. They have the following titles:
{doc_ids}
The user's query is as follows: {query}

For example:
If the query is "What is the revenue of Apple in Q3 2023? Compare it to that of Microsoft in the same timeframe," 
the response should be:

{{
    "list_of_docs": [
        {{
            "year": 2023,
            "quarter": "Q3",
            "ticker": "AAPL"
        }},
        {{
            "year": 2023,
            "quarter": "Q3",
            "ticker": "MSFT"
        }}
    ]
}}

Supply the response in this JSON format only:
{{
    "list_of_docs": [
        {{
            "year": int,
            "quarter": string,
            "ticker": string
        }}
    ]
}}

Extract the document titles that are relevant to the query.
""".strip()

    response = await generator.get_completions(
        [{"role": MessageType.SYSTEM.value, "content": extract_template}],
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        response_pydantic_type=Response,
    )

    return response
