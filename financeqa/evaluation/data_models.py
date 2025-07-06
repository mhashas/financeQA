from pydantic import BaseModel

from financeqa.app.schema import ReferencedDoc


class RetrievalEvalCase(BaseModel):
    question: str
    answer: list[ReferencedDoc]


class RetrievalEvalCases(BaseModel):
    cases: list[RetrievalEvalCase]
