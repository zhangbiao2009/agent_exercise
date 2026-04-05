"""
Step 2: Basic RAG Pipeline (Retriever + Answer)
================================================
This script searches the vector database (created by ingest.py) and uses
an LLM to answer questions based on the retrieved context.

The flow:
    1. EMBED the question  → convert to a vector (same model as ingestion)
    2. SEARCH LanceDB      → find the top K most similar chunks
    3. BUILD a prompt       → combine the question + retrieved chunks
    4. CALL the LLM        → get an answer grounded in your documents

This is "basic" RAG — one search, one answer. No grading or reformulation.
Steps 3-5 will add the agentic loop on top of this.

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    # Ask a question
    python retriever.py "What message broker did we choose?"

    # Just search (no LLM call) — useful for debugging retrieval
    python retriever.py --search-only "What caused the Q3 outage?"
"""

import os
import sys
import argparse

import lancedb
from sentence_transformers import SentenceTransformer
import litellm


# ============================================================================
# Part 1: SEARCH — Find relevant chunks in the vector database
# ============================================================================

class Retriever:
    """
    Connects to the LanceDB vector database and searches it.

    Why a class? We need to keep two things alive between searches:
    - The LanceDB table connection (avoid reconnecting every time)
    - The embedding model (takes a few seconds to load, reuse it)
    """

    def __init__(self, db_path: str = "data/lancedb", model_name: str = "all-MiniLM-L6-v2"):
        """
        Load the database and the embedding model.

        IMPORTANT: model_name must match what was used in ingest.py!
        If you ingested with "all-MiniLM-L6-v2" but search with a different
        model, the vectors are in different "spaces" and results will be garbage.
        """
        # Connect to the LanceDB database (just opens the folder on disk)
        self.db = lancedb.connect(db_path)

        # Open the "documents" table (created by ingest.py)
        self.table = self.db.open_table("documents")

        # Load the same embedding model used during ingestion
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the vector database for chunks most relevant to the query.

        How it works:
        1. Convert the query text into a vector using the embedding model
        2. Ask LanceDB to find the top_k closest vectors in the database
        3. Return the matching chunks with their metadata

        Args:
            query:  The user's question, e.g. "What message broker did we choose?"
            top_k:  How many results to return (default 5)

        Returns:
            A list of dicts, each with keys: text, source, heading, _distance
            Sorted by relevance (closest first).

            _distance is the vector distance — lower = more relevant.
            Typical values:
              0.0 - 0.5  → very relevant
              0.5 - 1.0  → somewhat relevant
              1.0+       → probably not relevant
        """
        # Step 1: Embed the query
        # This converts "What message broker did we choose?" into
        # [0.23, -0.41, 0.67, ...] — a 384-dimensional vector
        query_vector = self.model.encode(query).tolist()

        # Step 2: Search LanceDB
        # .search() computes the distance between query_vector and every
        # chunk vector in the table, then returns the closest top_k.
        #
        # Under the hood, it uses L2 (Euclidean) distance by default.
        # For 31 chunks this is instant. For millions, LanceDB uses
        # an index (IVF-PQ) to speed it up.
        results = (
            self.table
            .search(query_vector)
            .limit(top_k)
            .to_list()
        )

        # Step 3: Convert to clean dicts (drop the raw vector to save memory)
        chunks = []
        for row in results:
            chunks.append({
                "text": row["text"],
                "source": row["source"],
                "heading": row["heading"],
                "distance": row["_distance"],
            })

        return chunks


# ============================================================================
# Part 2: BUILD PROMPT — Assemble the LLM prompt from retrieved chunks
# ============================================================================

def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for the LLM call.

    The prompt has a specific structure designed to:
    1. Tell the LLM to ONLY use the provided context (prevent hallucination)
    2. Present each chunk with its source for citation
    3. Ask the question clearly at the end

    Returns a list of message dicts in the OpenAI chat format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    This format works with any LLM provider via litellm (DeepSeek, OpenAI, etc.)
    """

    # --- System message: sets the rules ---
    # This is the most important part for preventing hallucination.
    # "ONLY based on the provided context" forces the LLM to cite, not invent.
    system_message = (
        "You are a research assistant that answers questions based on a personal knowledge base.\n"
        "\n"
        "Rules:\n"
        "1. Answer ONLY based on the provided context below.\n"
        "2. If the context does not contain enough information to answer, say:\n"
        '   "I couldn\'t find this in the documents."\n'
        "3. Cite your sources using the format: (source: filename, § section).\n"
        "4. Be concise but thorough.\n"
    )

    # --- User message: context + question ---
    # We number each chunk [1], [2], etc. so the LLM can refer to them.
    # Including the source filename and heading helps it write accurate citations.
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk["source"]
        if chunk["heading"]:
            source_label += f", § {chunk['heading']}"

        context_parts.append(
            f"[{i}] (source: {source_label})\n"
            f"{chunk['text']}"
        )

    # Join all chunks with a blank line separator
    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Context (retrieved from your documents):\n"
        f"{'=' * 40}\n"
        f"{context_block}\n"
        f"{'=' * 40}\n"
        f"\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ============================================================================
# Part 3: ASK — Put it all together: search + prompt + LLM call
# ============================================================================

def ask(query: str, retriever: Retriever, model: str = "deepseek/deepseek-chat", top_k: int = 5) -> str:
    """
    The full RAG pipeline: search the DB, build a prompt, call the LLM.

    Args:
        query:      The user's question
        retriever:  A Retriever instance (connected to the vector DB)
        model:      LiteLLM model string (e.g. "deepseek/deepseek-chat")
        top_k:      Number of chunks to retrieve

    Returns:
        The LLM's answer as a string.
    """

    # --- Step 1: Retrieve relevant chunks ---
    print(f"\n🔍 Searching for: \"{query}\"")
    chunks = retriever.search(query, top_k=top_k)

    # Show what we found (useful for debugging)
    print(f"   Found {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        # Truncate text for display (first 80 chars)
        preview = chunk["text"][:80].replace("\n", " ") + "..."
        print(f"   [{i}] dist={chunk['distance']:.3f}  {chunk['source']}")
        print(f"       {preview}\n")

    # --- Step 2: Build the prompt ---
    messages = build_prompt(query, chunks)

    # --- Step 3: Call the LLM ---
    # litellm.completion() is a universal wrapper — works with any provider.
    # It reads the API key from environment variables automatically:
    #   DEEPSEEK_API_KEY for deepseek/*
    #   OPENAI_API_KEY for gpt-*
    #   ANTHROPIC_API_KEY for claude-*
    print(f"🤖 Asking {model}...\n")
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.1,    # Low temperature = more factual, less creative
        max_tokens=1024,    # Limit response length to control cost
    )

    # Extract the answer text from the response
    answer = response.choices[0].message.content

    # Show token usage (so you can monitor cost)
    usage = response.usage
    print(f"   Tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, "
          f"total: {usage.total_tokens}\n")

    return answer


# ============================================================================
# Part 4: CLI — Run from the command line
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Search your documents and get AI-powered answers")
    parser.add_argument("question", nargs="?", default=None, help="The question to ask")
    parser.add_argument("--search-only", action="store_true",
                        help="Just search, don't call the LLM (useful for debugging retrieval)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--model", default=None,
                        help="LiteLLM model string (default: AGENT_MODEL env var or deepseek/deepseek-chat)")
    parser.add_argument("--db", default="data/lancedb", help="Path to LanceDB database")
    args = parser.parse_args()

    # If no question provided, enter interactive mode
    if args.question is None:
        print("No question provided. Usage:")
        print('  python retriever.py "What message broker did we choose?"')
        print('  python retriever.py --search-only "What caused the Q3 outage?"')
        return

    # Determine which LLM to use
    model = args.model or os.environ.get("AGENT_MODEL", "deepseek/deepseek-chat")

    # Load the retriever (embedding model + DB connection)
    print("⏳ Loading retriever...")
    retriever = Retriever(db_path=args.db)

    if args.search_only:
        # --- Search-only mode: just show what the retrieval finds ---
        # This is useful for debugging. If the right chunks don't come back,
        # no amount of LLM magic will produce a good answer.
        chunks = retriever.search(args.question, top_k=args.top_k)
        print(f"\n🔍 Top {args.top_k} results for: \"{args.question}\"\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"{'─' * 60}")
            print(f"[{i}] distance={chunk['distance']:.3f}")
            print(f"    source: {chunk['source']}, § {chunk['heading']}")
            print(f"    text:\n{chunk['text'][:300]}")
            print()
    else:
        # --- Full RAG mode: search + LLM answer ---
        answer = ask(args.question, retriever, model=model, top_k=args.top_k)
        print("=" * 60)
        print(f"📝 Answer:\n\n{answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
