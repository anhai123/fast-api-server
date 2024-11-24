from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from uuid import UUID

class Job(BaseModel):
    Name: str
    ApplicationDeadline: date
    LinkCompany: str
    Description: str
    JobId: UUID
    Requirements: str
    Benefits: str
    Address: str
    HowToApply: str
