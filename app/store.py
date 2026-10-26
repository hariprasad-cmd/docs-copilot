"""The vector database (ChromaDB).

Each customer gets its own "collection" (we call it a workspace), so one
customer's docs never show up in another customer's answers.

ChromaDB turns text into embeddings for us using a small local model
(all-MiniLM-L6-v2). It downloads once (~80 MB) the first time you ingest.
"""

import re
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from app import config


@lru_cache  # create the client once and reuse it
def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def clean_workspace_name(name: str) -> str:
    """Chroma collection names may only use letters, digits, '.', '_' and '-' (3+ chars)."""
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name.strip().lower()).strip("-.")
    if len(name) < 3:
        raise ValueError("Workspace name must have at least 3 letters or digits.")
    return name


def get_collection(workspace: str):
    return _client().get_or_create_collection(name=clean_workspace_name(workspace))


def list_sources(workspace: str) -> dict:
    """Return {file name: number of chunks} for a workspace."""
    rows = get_collection(workspace).get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in rows["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts
