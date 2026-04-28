from pydantic import BaseModel


class ClassifierRequest(BaseModel):
    job_description: str


class MatchingRequest(BaseModel):
    job_description: str
    cv_text: str