# Docs Copilot

A retrieval-augmented generation (RAG) assistant that answers customer questions from a company's
own help docs, always cites its sources, and says "I don't know" instead of guessing.

It is built the way a Forward Deployed Engineer would set it up for a new customer:

1. **Ingest** the customer's docs (.md, .txt, .pdf) into their own workspace.
2. **Serve** grounded answers through a REST API and a small chat page.
3. **Evaluate** quality with a test set *before* go-live: retrieval hit rate, groundedness,
   correctness, and whether it declines out-of-scope questions.

The sample data is a made-up internet provider, "Brightwave Internet".

## How it works

```
 customer docs ──► chunk by heading ──► embeddings ──► ChromaDB (one collection per workspace)
                                                            │
 question ──► embed ──► top-4 similar chunks ◄──────────────┘
                              │
                              ▼
        LLM: Ollama (free, local) or Claude (answer only from numbered sources, cite [n])
                              │
                              ▼
                     answer + sources (JSON)
```

| File | What it does |
|------|--------------|
| `app/ingest.py` | Reads files, splits them into chunks by heading, stores them in ChromaDB |
| `app/store.py` | Opens the ChromaDB database; one collection per workspace (customer) |
| `app/rag.py` | Finds the closest chunks and asks the LLM to answer from them |
| `app/llm.py` | The only file that talks to the LLM: Ollama (local) or Claude (API) |
| `app/main.py` | FastAPI server: `/ask`, `/ingest`, `/workspaces/{name}`, `/health` |
| `app/static/index.html` | Simple chat page |
| `eval/run_eval.py` | Runs the test questions and prints quality metrics |

## Setup

You need [uv](https://docs.astral.sh/uv/) and either:
- **Ollama** (free, default): install from https://ollama.com, then `ollama pull qwen2.5:3b`
- **Claude API key**: set `LLM_PROVIDER=claude` and `ANTHROPIC_API_KEY` in `.env`

Running locally with Ollama means no customer data leaves the machine, which matters for
privacy-sensitive customers (healthcare, finance).

```bash
cd docs-copilot
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # defaults to Ollama; edit to switch to Claude
```

## Run it

```bash
# 1. Load the sample docs (first run downloads a small embedding model, ~80 MB)
python -m app.ingest data/sample_docs

# 2. Start the server
uvicorn app.main:app --reload
```

Open http://localhost:8000 to chat, or http://localhost:8000/docs for interactive API docs.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "When is my bill due?"}'

curl -X POST localhost:8000/ingest -F "file=@my_policy.pdf" -F "workspace=acme"
```

## Evaluate it

```bash
python -m eval.run_eval
```

Prints PASS/FAIL per question and a summary (pass rate, retrieval hit rate, grounded rate,
out-of-scope decline rate, average latency). Full results are saved in `eval/results/`.

The LLM judge only answers yes/no per check (Ollama forces its reply into a JSON schema); every score is computed in Python, because
LLMs are unreliable at arithmetic.

## Ideas for next steps

- Add more test questions, then change `TOP_K` or the chunk size and compare eval results.
- Stream answers to the chat page token by token.
- Add a Dockerfile and deploy it (Render, Railway, or a small VM).
