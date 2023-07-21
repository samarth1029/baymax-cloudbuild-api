"""
main code for FastAPI setup
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from app.models.models import AppDetails, GetXRayRequest, Drug, UserPrompt
from app.api.api import Api

description = """
API for serving as a chatbot for healthcare assistance🚀
"""

tags_metadata = [
    {
        "name": "default",
        "description": "endpoints for details of app",
    },
    {
        "name": "X-Rays",
        "description": "endpoints for x-ray interpretation",
    },
]

app = FastAPI(
    title="Baymax ChatAI",
    description=description,
    version="0.1",
    docs_url="/docs",
)


@app.get(
    "/",
)
def root():
    return {
        "message": "baymax-chatai using Fast API in Python. Go to <IP>:8000/docs for API-explorer.",
        "errors": None,
    }


@app.get("/appinfo/", tags=["default"])
def get_app_info() -> AppDetails:
    return AppDetails(**Api().get_app_details())