from pydantic import BaseModel


class AppDetails(BaseModel):
    appname: str
    version: str
    email: str
    author: str


class GetXRayRequest(BaseModel):
    url: str


class Drug(BaseModel):
    drug: str


class UserPrompt(BaseModel):
    prompt: str


