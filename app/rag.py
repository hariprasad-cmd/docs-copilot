"""Retrieval-Augmented Generation: find the relevant chunks, then ask the LLM to answer from them."""

from app import config
from app.llm import chat
from app.store import get_collection

SYSTEM_PROMPT = """You are the customer support assistant for {company}.

Answer the customer's question using only the numbered sources you are given.
- Cite the sources you used in square brackets at the end of the sentence they support, like [1] or [2][3].
- If the sources do not contain the answer, say you don't have that information
  and suggest contacting {company} support, with no citations. Do not guess or use outside knowledge.
- Keep answers short and friendly: 1-4 sentences, or a short list for steps."""


def retrieve(question: str, workspace: str, k: int = config.TOP_K) -> list[dict]:
    """Return the k chunks most similar to the question."""
    collection = get_collection(workspace)
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[question], n_results=min(k, collection.count()))
    # Chroma returns one list per query; we sent one query, so take item [0].
    sources = []
    for n, (text, meta, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), start=1
    ):
        sources.append({"n": n, "source": meta["source"], "text": text, "distance": round(distance, 3)})
    return sources


def build_user_message(question: str, sources: list[dict]) -> str:
    blocks = [f"[{s['n']}] (from {s['source']})\n{s['text']}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nCustomer question: {question}"


def answer(question: str, workspace: str = config.DEFAULT_WORKSPACE) -> dict:
    """The full RAG step: retrieve -> prompt the LLM -> return the answer plus its sources."""
    sources = retrieve(question, workspace)
    if not sources:
        return {"answer": "No documents have been loaded for this workspace yet.", "sources": []}

    text = chat(
        system=SYSTEM_PROMPT.format(company=config.COMPANY_NAME),
        user=build_user_message(question, sources),
    )
    return {"answer": text, "sources": sources}
