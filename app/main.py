"""The web API. Start it with:  uvicorn app.main:app --reload

Then open http://localhost:8000 for the chat page, or http://localhost:8000/docs for the API docs.
"""

import shutil
from pathlib import Path

import anthropic
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config
from app.ingest import SUPPORTED, ingest_file
from app.llm import LLMSetupError
from app.rag import answer
from app.store import clean_workspace_name, list_sources

app = FastAPI(title="Docs Copilot", description="Grounded Q&A over a customer's help docs.")

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    workspace: str = config.DEFAULT_WORKSPACE


class Source(BaseModel):
    n: int
    source: str
    text: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/", include_in_schema=False)
def chat_page():
    # no-cache: browsers re-check the page each load, so UI edits show up on refresh
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"status": "ok", "provider": config.LLM_PROVIDER, "model": config.MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        return answer(req.question, req.workspace)
    except ValueError as e:  # bad workspace name
        raise HTTPException(status_code=400, detail=str(e))
    except LLMSetupError as e:  # no API key, Ollama not running, model not installed...
        raise HTTPException(status_code=500, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Claude API key is missing or invalid. Check your .env file.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Claude API. Try again in a minute.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the Claude API. Check your internet connection.")


@app.post("/ingest")
def ingest(file: UploadFile = File(...), workspace: str = Form(config.DEFAULT_WORKSPACE)):
    """Upload one .md / .txt / .pdf file and add it to a workspace."""
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(sorted(SUPPORTED))} files are supported.")
    try:
        workspace = clean_workspace_name(workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    folder = config.UPLOAD_DIR / workspace
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / name
    with saved.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    chunks = ingest_file(saved, workspace)
    return {"file": name, "workspace": workspace, "chunks": chunks}


@app.get("/workspaces/{workspace}")
def workspace_info(workspace: str):
    try:
        sources = list_sources(workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"workspace": workspace, "files": sources, "total_chunks": sum(sources.values())}
