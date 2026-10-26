"""All settings live here, read from the .env file (or real environment variables)."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains app/, data/, eval/
ROOT = Path(__file__).resolve().parent.parent

# Load .env into os.environ (this is also how the Anthropic SDK finds ANTHROPIC_API_KEY).
load_dotenv(ROOT / ".env")

# Which LLM answers questions: "ollama" (free, local) or "claude" (needs an API key)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")  # small enough for an 8 GB laptop
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
MODEL = CLAUDE_MODEL if LLM_PROVIDER == "claude" else OLLAMA_MODEL

TOP_K = int(os.getenv("TOP_K", "4"))  # how many chunks to retrieve per question
COMPANY_NAME = os.getenv("COMPANY_NAME", "Brightwave Internet")

CHROMA_DIR = ROOT / "storage" / "chroma"  # where the vector database is saved on disk
UPLOAD_DIR = ROOT / "data" / "uploads"    # where files uploaded through the API are kept
DEFAULT_WORKSPACE = "brightwave"          # one workspace = one customer's set of docs
