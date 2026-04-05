"""
Step 1: Document Ingestion Pipeline
====================================
This script takes a folder of markdown files and turns them into a searchable
vector database. It has four stages:

    1. LOAD    — Read each .md file from the docs/ folder
    2. CHUNK   — Split each file into smaller pieces (preserving meaning)
    3. EMBED   — Convert each chunk into a vector (list of numbers)
    4. STORE   — Save all chunks + vectors into a local LanceDB database

After running this script, you'll have a `data/lancedb/` folder containing
the vector database. The retriever (Step 2) will query this database.

Usage:
    python ingest.py                    # ingest docs/ with default settings
    python ingest.py --docs my_notes/   # ingest a different folder
"""

import os
import re
import glob
import argparse

import lancedb
from sentence_transformers import SentenceTransformer


# ============================================================================
# Stage 1: LOAD — Read markdown files from disk
# ============================================================================

def load_documents(docs_dir: str) -> list[dict]:
    """
    Read all .md files from a directory and return them as a list of dicts.

    Each dict looks like:
        {"source": "architecture-decisions.md", "text": "# Architecture..."}

    We store the filename in "source" so we can trace answers back to
    the original file later.
    """
    documents = []

    # glob.glob finds all files matching the pattern
    # e.g., docs/*.md → ["docs/architecture-decisions.md", "docs/meeting-notes.md", ...]
    pattern = os.path.join(docs_dir, "*.md")
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        print(f"⚠️  No .md files found in {docs_dir}")
        return documents

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Use just the filename (not the full path) as the source label
        filename = os.path.basename(path)
        documents.append({"source": filename, "text": text})
        print(f"  📄 Loaded {filename} ({len(text)} chars)")

    print(f"\n  Total: {len(documents)} documents loaded.\n")
    return documents


# ============================================================================
# Stage 2: CHUNK — Split documents into smaller pieces
# ============================================================================

def chunk_document(doc: dict, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Split one document into chunks. Each chunk is a dict:
        {"source": "file.md", "heading": "## Some Heading", "text": "..."}

    Strategy (markdown-aware, with fallback):
    -----------------------------------------
    1. First, split on markdown headings (## ).
       Each heading creates a natural "section" of related content.

    2. If a section is still too long (> chunk_size words), split it further
       by paragraphs (blank lines).

    3. If a paragraph is STILL too long, split by sentences.

    4. Apply overlap: each chunk shares some words with the next chunk,
       so context isn't lost at boundaries.

    Why this order?
    - Headings are the strongest semantic boundary (topic change)
    - Paragraphs are the next strongest (idea change)
    - Sentences are the weakest but still preserve complete thoughts
    """
    text = doc["text"]
    source = doc["source"]
    chunks = []

    # -------------------------------------------------------------------
    # Step 2a: Split the document into sections by markdown headings
    # -------------------------------------------------------------------
    # re.split(r'(?=^## )', text, flags=re.MULTILINE)
    #
    # This regex says: "split right BEFORE any line that starts with '## '"
    # The (?=...) is a "lookahead" — it splits at that position but keeps
    # the heading in the resulting section.
    #
    # Example input:
    #   "# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"
    # Result:
    #   ["# Title\nIntro\n", "## Section A\nContent A\n", "## Section B\nContent B"]
    sections = re.split(r'(?=^## )', text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract the heading from this section (if it has one)
        # We'll store it as metadata so the answer can cite "§ Choosing a Message Broker"
        heading = ""
        first_line = section.split("\n")[0]
        if first_line.startswith("#"):
            heading = first_line.strip("# ").strip()

        # -----------------------------------------------------------
        # Step 2b: Check if this section fits in one chunk
        # -----------------------------------------------------------
        words = section.split()
        if len(words) <= chunk_size:
            # Small enough — keep as one chunk
            chunks.append({
                "source": source,
                "heading": heading,
                "text": section,
            })
        else:
            # -----------------------------------------------------------
            # Step 2c: Section too big — split by paragraphs, then by
            #          sentences if needed, with overlap
            # -----------------------------------------------------------
            sub_chunks = _split_long_text(section, chunk_size, chunk_overlap)
            for sub_text in sub_chunks:
                chunks.append({
                    "source": source,
                    "heading": heading,
                    "text": sub_text,
                })

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long text into overlapping chunks by paragraphs, then sentences.

    Example with chunk_size=10 words and overlap=3:

        Original: "A B C D E F G H I J K L M N O"  (15 words)

        Chunk 1:  "A B C D E F G H I J"            (words 0-9)
        Chunk 2:  "H I J K L M N O"                 (words 7-14, overlaps 3 words)
                   ^^^
                   overlap — these words appear in BOTH chunks

    This ensures that if a concept spans a chunk boundary, at least one
    chunk contains the full context.
    """
    # First, try splitting by paragraphs (double newline)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build chunks by accumulating paragraphs until we hit the size limit
    result_chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph would exceed the limit, save current chunk
        if current_words and (len(current_words) + len(para_words)) > chunk_size:
            result_chunks.append(" ".join(current_words))

            # Keep the last 'overlap' words for the next chunk
            # This is how overlap works — we carry forward some context
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(para_words)

    # Don't forget the last chunk
    if current_words:
        result_chunks.append(" ".join(current_words))

    return result_chunks


def chunk_all_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Apply chunking to every document. Returns a flat list of all chunks
    across all documents.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        print(f"  🔪 {doc['source']} → {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    print(f"\n  Total: {len(all_chunks)} chunks created.\n")
    return all_chunks


# ============================================================================
# Stage 3: EMBED — Convert text chunks into vectors
# ============================================================================

def embed_chunks(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """
    Convert each chunk's text into a vector (list of 384 floats).

    How embedding works:
    --------------------
    The sentence-transformers model reads the text and outputs a fixed-size
    vector that captures the "meaning" of the text. Texts with similar
    meaning get vectors that are close together in 384-dimensional space.

    Example:
        "We chose Kafka"     → [0.12, -0.34, 0.87, ...]  (384 numbers)
        "Kafka was selected"  → [0.11, -0.33, 0.85, ...]  (very close!)
        "The cat sat on mat"  → [0.92, 0.15, -0.44, ...]  (far away)

    The model "all-MiniLM-L6-v2":
    - Runs locally (no API calls, no cost, private)
    - ~80MB download on first run (cached after that)
    - Output: 384-dimensional vectors
    - Good balance of speed and quality for English text

    We embed ALL chunks in one batch call because it's much faster than
    one-by-one (the model can parallelize on your CPU/GPU).
    """
    print(f"  🤖 Loading embedding model: {model_name}")
    print(f"     (first run downloads ~80MB, cached after that)\n")

    # Load the model — this is a local neural network, not an API call
    model = SentenceTransformer(model_name)

    # Extract just the text from each chunk for batch embedding
    texts = [chunk["text"] for chunk in chunks]

    # Embed all texts at once — returns a numpy array of shape (N, 384)
    print(f"  ⚡ Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)

    # Attach the vector to each chunk dict
    for chunk, vector in zip(chunks, vectors):
        # .tolist() converts numpy array → plain Python list
        # LanceDB needs plain lists, not numpy arrays
        chunk["vector"] = vector.tolist()

    print(f"\n  ✅ Embedded {len(chunks)} chunks (vector dimension: {len(vectors[0])})\n")
    return chunks


# ============================================================================
# Stage 4: STORE — Save everything into LanceDB
# ============================================================================

def store_in_lancedb(chunks: list[dict], db_path: str = "data/lancedb") -> None:
    """
    Save all chunks (with their vectors) into a local LanceDB database.

    LanceDB stores data as files on disk — no server needed.
    After this, the db_path folder will contain the vector database.

    The table schema looks like:
    ┌────────────────────────────────┬──────────────────────────┬──────────────┬──────────────────┐
    │ text (string)                  │ source (string)          │ heading (str)│ vector (float[]) │
    ├────────────────────────────────┼──────────────────────────┼──────────────┼──────────────────┤
    │ "After the Q3 outage..."       │ architecture-decisions.md│ Breaking Up  │ [0.12, -0.34,...]│
    │ "We evaluated three options.." │ architecture-decisions.md│ Message Brok │ [0.45, 0.22,...] │
    │ "A database migration script." │ incident-q3-2025.md     │ Summary      │ [0.78, -0.11,...]│
    └────────────────────────────────┴──────────────────────────┴──────────────┴──────────────────┘
    """
    print(f"  💾 Storing {len(chunks)} chunks in LanceDB at {db_path}/")

    # Connect to (or create) the database — just a folder on disk
    db = lancedb.connect(db_path)

    # Create the table (or overwrite if it already exists)
    # "documents" is the table name — like a SQL table name
    # mode="overwrite" means re-running ingest.py replaces old data
    table = db.create_table("documents", data=chunks, mode="overwrite")

    print(f"  ✅ Stored! Table 'documents' has {table.count_rows()} rows.\n")


# ============================================================================
# Main: Wire all four stages together
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    parser.add_argument("--docs", default="docs", help="Path to documents folder")
    parser.add_argument("--db", default="data/lancedb", help="Path for LanceDB storage")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max words per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap words between chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    args = parser.parse_args()

    print("=" * 60)
    print("  Document Ingestion Pipeline")
    print("=" * 60)

    # Stage 1: Load
    print("\n📂 Stage 1: Loading documents...\n")
    documents = load_documents(args.docs)
    if not documents:
        return

    # Stage 2: Chunk
    print("🔪 Stage 2: Chunking documents...\n")
    chunks = chunk_all_documents(documents, args.chunk_size, args.chunk_overlap)

    # Stage 3: Embed
    print("🧮 Stage 3: Embedding chunks...\n")
    chunks = embed_chunks(chunks, args.model)

    # Stage 4: Store
    print("💾 Stage 4: Storing in LanceDB...\n")
    store_in_lancedb(chunks, args.db)

    # Summary
    print("=" * 60)
    print("  Done! Your vector database is ready at:", args.db)
    print("  Next step: run retriever.py to search it.")
    print("=" * 60)


if __name__ == "__main__":
    main()
