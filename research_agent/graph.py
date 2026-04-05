"""
Step 4: LangGraph State Machine
================================
This script replaces the for-loop in grader.py with a LangGraph graph.
Same logic, but structured as nodes (functions) and edges (decisions).

WHY LANGGRAPH?
--------------
In grader.py, the agentic loop is a Python for-loop with if/else.
That works, but has limitations:
  - No pause/resume: if the process crashes, you start over
  - No human-in-the-loop: can't pause and ask the user mid-flow
  - No persistent memory: state is lost when the script exits
  - Hard to visualize: the flow is buried in control-flow code

LangGraph solves all of these by structuring the agent as a GRAPH:
  - Each step is a NODE (a function)
  - Each decision is a CONDITIONAL EDGE (which node to go to next)
  - State flows through the graph and is saved at each step

THE GRAPH:
----------
    ┌──────────┐
    │  START    │
    └────┬─────┘
         ▼
  ┌─────────────┐
  │  retrieve    │  Search the vector DB
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │    grade     │  LLM scores the results (1-5)
  └──────┬──────┘
         ▼
    score >= 3?     Conditional edge
    ┌────┴────┐
   YES       NO
    │         │
    ▼         ▼
┌────────┐ ┌─────────────┐
│generate│ │ reformulate  │  Rewrite the query
└───┬────┘ └──────┬──────┘
    │              │
    ▼              │ (loops back to retrieve)
┌────────┐         │
│  END   │ ◄──────┘ (if max retries hit)
└────────┘

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    python graph.py "What message broker did we choose?"
    python graph.py "How do we know when things break?"
    python graph.py "What's our mobile app roadmap?"
"""

import os
import json
import argparse
from typing import TypedDict, Annotated

import litellm

# LangGraph imports
# -----------------
# StateGraph: the builder that lets you define nodes and edges
# START: a special constant meaning "the entry point of the graph"
# END: a special constant meaning "the graph is done"
from langgraph.graph import StateGraph, START, END

# MemorySaver: stores graph state in memory (for checkpointing)
# Later we can swap this for SqliteSaver for persistence across restarts
from langgraph.checkpoint.memory import MemorySaver

# Reuse Retriever and build_prompt from Step 2 (no duplication)
from retriever import Retriever, build_prompt


# ============================================================================
# Part 1: STATE — Define what data flows through the graph
# ============================================================================

class AgentState(TypedDict):
    """
    The state that flows through every node in the graph.

    Think of this as a shared "blackboard" that each node can read and write.
    When a node returns {"chunks": [...]}, LangGraph MERGES that into the
    existing state — it doesn't replace the whole state.

    TypedDict gives us type hints so we know what fields exist.
    """
    # --- Inputs (set once at the start) ---
    query: str              # The user's original question (never changes)
    model: str              # LLM model string (e.g. "deepseek/deepseek-chat")
    top_k: int              # Number of chunks to retrieve per search
    max_retries: int        # Maximum search attempts
    grade_threshold: int    # Minimum score to accept results

    # --- Working state (updated by nodes as the graph runs) ---
    current_query: str      # The current search query (may be reformulated)
    chunks: list[dict]      # Chunks from the LAST search
    best_chunks: list[dict] # Best chunks found across ALL attempts
    best_score: int         # Highest grader score so far
    grade: dict             # Result from the last grading call
    queries_tried: list[str]  # All queries we've tried so far
    attempt: int            # Current attempt number (1, 2, 3, ...)

    # --- Output (set at the end) ---
    answer: str             # The final answer


# ============================================================================
# Part 2: NODES — Each node is a function that does ONE thing
# ============================================================================
#
# Node functions take the FULL state as input and return a PARTIAL dict
# of only the fields they want to update. LangGraph merges these updates
# into the existing state automatically.
#
# Example:
#   State before: {"query": "...", "attempt": 1, "chunks": []}
#   Node returns: {"chunks": [...], "attempt": 2}
#   State after:  {"query": "...", "attempt": 2, "chunks": [...]}
#   (query unchanged, chunks and attempt updated)

# We create the Retriever once and share it across all node calls.
# This avoids reloading the embedding model on every search.
_retriever = None

def _get_retriever(db_path: str = "data/lancedb") -> Retriever:
    """Lazy-load the retriever (embedding model + DB) once."""
    global _retriever
    if _retriever is None:
        print("⏳ Loading retriever...")
        _retriever = Retriever(db_path=db_path)
    return _retriever


def retrieve_node(state: AgentState) -> dict:
    """
    NODE: Search the vector database with the current query.

    Reads:  state["current_query"], state["top_k"], state["queries_tried"]
    Writes: state["chunks"], state["queries_tried"]

    This is the same search logic from retriever.py, just wrapped as a
    LangGraph node.
    """
    retriever = _get_retriever()
    current_query = state["current_query"]
    attempt = state["attempt"]

    print(f"\n{'━' * 60}")
    print(f"  📍 NODE: retrieve (attempt {attempt}/{state['max_retries']})")
    print(f"{'━' * 60}")
    print(f"\n🔍 Searching for: \"{current_query}\"")

    # Search the vector DB
    chunks = retriever.search(current_query, top_k=state["top_k"])

    # Show what we found
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["text"][:80].replace("\n", " ") + "..."
        print(f"   [{i}] dist={chunk['distance']:.3f}  {chunk['source']}")
        print(f"       {preview}")

    # Return ONLY the fields we want to update
    # LangGraph merges this into the existing state
    return {
        "chunks": chunks,
        "queries_tried": state["queries_tried"] + [current_query],
    }


def grade_node(state: AgentState) -> dict:
    """
    NODE: Ask the LLM to grade whether the retrieved chunks answer the question.

    Reads:  state["query"], state["chunks"], state["model"], state["queries_tried"]
    Writes: state["grade"], state["best_chunks"], state["best_score"]

    Same grading logic from grader.py, wrapped as a LangGraph node.
    """
    print(f"\n  📍 NODE: grade")

    query = state["query"]       # Always grade against the ORIGINAL question
    chunks = state["chunks"]
    model = state["model"]
    queries_tried = state["queries_tried"]

    # --- Build the grading prompt ---
    system_message = (
        "You are a relevance grader. Your job is to evaluate whether retrieved "
        "document chunks contain enough information to answer a user's question.\n"
        "\n"
        "Respond with ONLY a JSON object (no markdown, no explanation outside JSON):\n"
        "{\n"
        '  "score": <1-5>,\n'
        '  "reasoning": "<one sentence explaining your score>",\n'
        '  "suggestion": "<a better search query if score < 3, otherwise empty string>"\n'
        "}\n"
        "\n"
        "Scoring guide:\n"
        "  5 = Chunks directly and completely answer the question\n"
        "  4 = Chunks mostly answer the question, minor gaps\n"
        "  3 = Chunks partially answer, but enough for a reasonable response\n"
        "  2 = Chunks are tangentially related but don't answer the question\n"
        "  1 = Chunks are completely irrelevant to the question\n"
        "\n"
        'For "suggestion": if score < 3, suggest a DIFFERENT search query that\n'
        "might find better results. Use different words, be more specific, or\n"
        "approach the topic from a different angle.\n"
        "\n"
        "IMPORTANT: Look at the retrieved chunks for CLUES about what vocabulary\n"
        "the documents use. If you see related terms in the chunks (e.g., a chunk\n"
        "mentions 'Prometheus' or 'alerting'), use those exact terms in your\n"
        "suggestion. Match the document's language, not the user's."
    )

    # Format chunks for the grader
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        if chunk["heading"]:
            source += f", § {chunk['heading']}"
        chunks_text += f"[{i}] (source: {source})\n{chunk['text']}\n\n"

    # Tell the grader what we already tried (so it doesn't repeat bad queries)
    tried_text = ""
    if queries_tried:
        tried_text = (
            f"\nPrevious queries that didn't work well:\n"
            + "\n".join(f'  - "{q}"' for q in queries_tried)
            + "\nDo NOT suggest anything similar to these.\n"
        )

    user_message = (
        f"Question: {query}\n\n"
        f"Retrieved chunks:\n"
        f"{chunks_text}"
        f"{tried_text}"
        f"Grade these chunks for relevance to the question."
    )

    # Call the LLM for grading
    print(f"📊 Grading results...")
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=200,
    )

    raw_content = response.choices[0].message.content.strip()

    # Parse JSON response (strip markdown fences if present)
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`").strip()
        if raw_content.startswith("json"):
            raw_content = raw_content[4:].strip()

    try:
        grade = json.loads(raw_content)
    except json.JSONDecodeError:
        print(f"   ⚠️  Grader returned invalid JSON, assuming score=3")
        grade = {"score": 3, "reasoning": "Could not parse", "suggestion": ""}

    grade.setdefault("score", 3)
    grade.setdefault("reasoning", "")
    grade.setdefault("suggestion", "")

    score = grade["score"]
    print(f"   Score: {score}/5")
    print(f"   Reasoning: {grade['reasoning']}")

    # Track the best results across all attempts
    best_chunks = state["best_chunks"]
    best_score = state["best_score"]
    if score > best_score:
        best_chunks = chunks
        best_score = score

    # Show token usage
    usage = response.usage
    print(f"   Grader tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, total: {usage.total_tokens}")

    return {
        "grade": grade,
        "best_chunks": best_chunks,
        "best_score": best_score,
    }


def reformulate_node(state: AgentState) -> dict:
    """
    NODE: Use the grader's suggestion as the new search query.

    Reads:  state["grade"], state["attempt"]
    Writes: state["current_query"], state["attempt"]

    This node only runs when the grader scored low AND we have retries left.
    It takes the grader's suggestion and sets it as the next search query,
    then increments the attempt counter.
    """
    suggestion = state["grade"].get("suggestion", "")
    attempt = state["attempt"]

    print(f"\n  📍 NODE: reformulate")
    print(f"🔄 Reformulating query...")
    print(f"   Old: \"{state['current_query']}\"")
    print(f"   New: \"{suggestion}\"")

    return {
        "current_query": suggestion if suggestion else state["current_query"],
        "attempt": attempt + 1,
    }


def generate_node(state: AgentState) -> dict:
    """
    NODE: Generate the final answer from the best chunks found.

    Reads:  state["query"], state["best_chunks"], state["best_score"], state["model"]
    Writes: state["answer"]

    This node runs when either:
    - The grader scored high enough (good results found)
    - Max retries reached (we answer with whatever we have)
    """
    query = state["query"]
    best_chunks = state["best_chunks"]
    best_score = state["best_score"]
    model = state["model"]

    print(f"\n  📍 NODE: generate")

    # If the best score is very low, don't try to answer — be honest
    if best_score < 2:
        answer = (
            f"I couldn't find relevant information in the documents to answer "
            f"your question: \"{query}\"\n\n"
            f"Queries tried: {state['queries_tried']}\n"
            f"Best relevance score: {best_score}/5"
        )
        print(f"📝 No relevant results found (best score: {best_score}/5)")
        return {"answer": answer}

    # Build prompt and call the LLM for the final answer
    print(f"🤖 Generating answer from best results (score {best_score}/5)...\n")
    messages = build_prompt(query, best_chunks)

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    usage = response.usage
    print(f"   Answer tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, total: {usage.total_tokens}")

    return {"answer": answer}


# ============================================================================
# Part 3: EDGES — Conditional routing (the decision points)
# ============================================================================

def should_retry_or_answer(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: After grading, decide what to do next.

    This function is called after the "grade" node. It looks at the state
    and returns a STRING that tells LangGraph which node to go to next.

    Returns one of:
        "generate"    → score is good enough, go produce the answer
        "reformulate" → score is bad, try a different query
        "generate"    → max retries reached, answer with what we have

    In LangGraph, conditional edges are the "if/else" of the graph.
    Instead of writing:
        if score >= threshold:
            generate()
        else:
            reformulate()

    You write this function, and LangGraph handles the routing.
    """
    score = state["grade"]["score"]
    threshold = state["grade_threshold"]
    attempt = state["attempt"]
    max_retries = state["max_retries"]

    if score >= threshold:
        # ✅ Good enough — proceed to answer
        print(f"\n✅ Score {score} >= threshold {threshold}. → generate")
        return "generate"
    elif attempt >= max_retries:
        # ❌ Out of retries — answer with whatever we have
        print(f"\n❌ Max retries reached ({attempt}/{max_retries}). → generate")
        return "generate"
    else:
        # 🔄 Bad results, retries left — try a different query
        print(f"\n🔄 Score {score} < {threshold}, attempt {attempt}/{max_retries}. → reformulate")
        return "reformulate"


# ============================================================================
# Part 4: BUILD THE GRAPH — Wire nodes and edges together
# ============================================================================

def build_graph():
    """
    Construct the LangGraph state machine.

    This is where we define:
    - What nodes exist (the functions above)
    - How they connect (edges)
    - Where to start
    - Where decisions happen (conditional edges)

    The result is a compiled "graph" object that you can call with .invoke().
    """

    # --- Step 1: Create a StateGraph builder ---
    # Pass the state schema (AgentState) so LangGraph knows what fields exist
    builder = StateGraph(AgentState)

    # --- Step 2: Add nodes ---
    # Each node has a name (string) and a function.
    # The function receives the full state and returns partial updates.
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("reformulate", reformulate_node)
    builder.add_node("generate", generate_node)

    # --- Step 3: Add edges ---
    #
    # SIMPLE EDGES: "always go from A to B"
    #   add_edge(A, B) means: after node A finishes, always run node B
    #
    # CONDITIONAL EDGES: "decide where to go based on state"
    #   add_conditional_edges(A, decision_func, {result: next_node})
    #   means: after node A, call decision_func(state).
    #   The return value maps to the next node via the dict.

    # START → retrieve (the first thing we do is search)
    builder.add_edge(START, "retrieve")

    # retrieve → grade (after searching, always grade the results)
    builder.add_edge("retrieve", "grade")

    # grade → ??? (this is the decision point!)
    # should_retry_or_answer() returns "generate" or "reformulate"
    builder.add_conditional_edges(
        "grade",                    # After this node...
        should_retry_or_answer,     # ...call this function to decide...
        {
            "generate": "generate",       # if it returns "generate" → go to generate
            "reformulate": "reformulate", # if it returns "reformulate" → go to reformulate
        }
    )

    # reformulate → retrieve (after reformulating, search again — this creates the LOOP)
    builder.add_edge("reformulate", "retrieve")

    # generate → END (after generating the answer, we're done)
    builder.add_edge("generate", END)

    # --- Step 4: Compile the graph ---
    # MemorySaver stores checkpoints in memory. This means:
    # - Each node's output is saved after it runs
    # - If you call invoke() with the same thread_id, it can resume
    # - For now it's in-memory (lost on restart). Step 6 will use SqliteSaver.
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# ============================================================================
# Part 5: CLI — Run from the command line
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG with LangGraph state machine"
    )
    parser.add_argument("question", nargs="?", default=None, help="The question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-retries", type=int, default=3, help="Max search attempts")
    parser.add_argument("--threshold", type=int, default=3, help="Min grader score (1-5)")
    parser.add_argument("--model", default=None, help="LiteLLM model string")
    parser.add_argument("--db", default="data/lancedb", help="Path to LanceDB database")
    parser.add_argument("--thread", default="default",
                        help="Thread ID for checkpointing (same ID = same conversation)")
    args = parser.parse_args()

    if args.question is None:
        print("Usage:")
        print('  python graph.py "What message broker did we choose?"')
        print('  python graph.py "How do we know when things break?"')
        print('  python graph.py --thread research-1 "What caused the Q3 outage?"')
        return

    model = args.model or os.environ.get("AGENT_MODEL", "deepseek/deepseek-chat")

    # --- Build the graph ---
    graph = build_graph()

    # --- Prepare the initial state ---
    # These are the inputs that kick off the graph.
    # Each field maps to a key in AgentState.
    initial_state = {
        "query": args.question,
        "model": model,
        "top_k": args.top_k,
        "max_retries": args.max_retries,
        "grade_threshold": args.threshold,
        # Working state starts empty/at zero
        "current_query": args.question,   # First search uses the original question
        "chunks": [],
        "best_chunks": [],
        "best_score": 0,
        "grade": {},
        "queries_tried": [],
        "attempt": 1,
        "answer": "",
    }

    # --- Run the graph ---
    # config contains the thread_id for checkpointing.
    # Same thread_id = LangGraph can resume from where it left off.
    # Different thread_id = fresh start.
    config = {"configurable": {"thread_id": args.thread}}

    print(f"\n{'=' * 60}")
    print(f"  LangGraph Agentic RAG")
    print(f"  Thread: {args.thread}")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    # invoke() runs the graph from START to END, following edges.
    # It returns the FINAL state after all nodes have run.
    final_state = graph.invoke(initial_state, config)

    # --- Print the answer ---
    print(f"\n{'=' * 60}")
    print(f"📝 Answer:\n\n{final_state['answer']}")
    print(f"{'=' * 60}")

    # --- Show a summary of what happened ---
    print(f"\nSummary:")
    print(f"  Queries tried: {final_state['queries_tried']}")
    print(f"  Best score: {final_state['best_score']}/5")
    print(f"  Attempts used: {final_state['attempt']}/{args.max_retries}")


if __name__ == "__main__":
    main()
