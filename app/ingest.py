"""Ingestion: read documents, split them into chunks, save the chunks in the vector DB.

Run from the project root:
    python -m app.ingest data/sample_docs
    python -m app.ingest path/to/customer_docs --workspace acme
"""

import argparse
from pathlib import Path

from pypdf import PdfReader

from app import config
from app.store import get_collection

SUPPORTED = {".md", ".txt", ".pdf"}
MAX_CHUNK_CHARS = 1000  # long sections get split into pieces about this size


def read_text(path: Path) -> str:
    """Return the plain text of a .md, .txt or .pdf file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str) -> list[str]:
    """Split a document into chunks.

    Step 1: split on markdown headings ("# ", "## ", ...) so each chunk is about one topic.
    Step 2: if a section is still too long, group its paragraphs into ~1000-character pieces.
    Each piece keeps its section heading at the top, so the chunk makes sense on its own.
    """
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: list[str] = []
    for section in sections:
        # Skip empty sections and ones that are only a title line (nothing to answer from).
        if not any(line.strip() and not line.startswith("#") for line in section.splitlines()):
            continue
        if len(section) <= MAX_CHUNK_CHARS:
            chunks.append(section)
            continue

        lines = section.splitlines()
        heading = lines[0] if lines[0].startswith("#") else ""
        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        piece = ""
        for para in paragraphs:
            if piece and len(piece) + len(para) > MAX_CHUNK_CHARS:
                chunks.append(piece)
                piece = heading  # start the next piece with the heading again
            piece = f"{piece}\n\n{para}".strip()
        if piece and piece != heading:
            chunks.append(piece)
    return chunks


def ingest_file(path: Path, workspace: str) -> int:
    """Add one file to a workspace. Returns how many chunks were stored."""
    chunks = split_into_chunks(read_text(path))
    if not chunks:
        return 0

    collection = get_collection(workspace)
    # Remove old chunks of this file first, so re-ingesting an edited file doesn't leave stale text.
    collection.delete(where={"source": path.name})
    collection.add(
        ids=[f"{path.name}::{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def ingest_folder(folder: Path, workspace: str) -> dict[str, int]:
    results = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            results[path.name] = ingest_file(path, workspace)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into a workspace.")
    parser.add_argument("folder", type=Path, help="folder with .md / .txt / .pdf files")
    parser.add_argument("--workspace", default=config.DEFAULT_WORKSPACE)
    args = parser.parse_args()

    results = ingest_folder(args.folder, args.workspace)
    for name, count in results.items():
        print(f"  {name}: {count} chunks")
    print(f"Done. {sum(results.values())} chunks from {len(results)} files in workspace '{args.workspace}'.")


if __name__ == "__main__":
    main()
