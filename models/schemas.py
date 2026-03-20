from enum import Enum

from pydantic import BaseModel


class QueryMode(str, Enum):
    strict = "strict"
    creative = "creative"


class AnswerLength(str, Enum):
    extra_short = "extra_short"
    short = "short"
    medium = "medium"
    long = "long"


class LearnRequest(BaseModel):
    content: str | None = None
    url: str | None = None


class LearnResponse(BaseModel):
    status: str
    chunks_added: int
    source: str


class QueryRequest(BaseModel):
    question: str
    hint: str | None = None
    model: str | None = None
    mode: QueryMode = QueryMode.strict
    length: AnswerLength | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
