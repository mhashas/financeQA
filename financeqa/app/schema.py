from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ReferencedDoc(BaseModel):
    year: int
    quarter: str
    company: str
    page: int


class ReferencedResponse(BaseModel):
    """Pair of a response and a reference"""

    response: str
    references: list[ReferencedDoc]


class Message(BaseModel):
    session_id: Optional[UUID] = None
    message: str
