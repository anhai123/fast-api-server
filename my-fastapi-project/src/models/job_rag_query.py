from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    question: str
    answer: str = Field(..., min_length=1)

class JobQueryInput(BaseModel):
    text: str
    messages: List[Message]


class JobQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
