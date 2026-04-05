"""
Step 4 + 5: LangGraph State Machine with Human-in-the-Loop
===========================================================
This script replaces the for-loop in grader.py with a LangGraph graph.
Same logic, but structured as nodes (functions) and edges (decisions).

Step 5 adds HUMAN-IN-THE-LOOP: when the grader scores low, the graph
PAUSES and asks you what to search next. You can accept the suggestion,
type your own query, or quit.

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

HOW HUMAN-IN-THE-LOOP WORKS:
----------------------------
LangGraph has a built-in function called interrupt().

1. When a node calls interrupt(payload), LangGraph:
   - Saves the current state to the checkpointer
   - Stops execution and returns to the caller
   - The payload is sent back so the caller can show it to the user

2. The caller (main loop) sees the graph is paused:
   - graph.get_state(config).next is non-empty (there's more to do)
   - The interrupt payload tells us what the grader suggested

3. The caller collects user input and RESUMES:
   - graph.invoke(Command(resume=user_input), config)
   - Back inside the node, interrupt() RETURNS user_input
   - The node continues as if interrupt() was just an input() call

Think of interrupt() as a "networked input()" — it pauses the
program, sends a message out, and waits for a response back.

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
│generate│ │ reformulate  │  ⏸ PAUSE here if --human
└───┬────┘ └──────┬──────┘
    │              │
    ▼              │ (loops back to retrieve)
┌────────┐         │
│  END   │ ◄──────┘ (if max retries hit)
└────────┘

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    # Fully automatic (same as before)
    python graph.py "What message broker did we choose?"

    # Interactive — pauses and asks you when results are bad
    python graph.py --human "How do we know when things break?"
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

# interrupt: pauses the graph and sends a value back to the caller.
#   When resumed with Command(resume=X), interrupt() returns X.
# Command: used to resume a paused graph. Command(resume=value) sends
#   the value back to the interrupt() call that paused the graph.
from langgraph.types import interrupt, Command

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
    human_mode: bool        # If True, pause at reformulate for human input

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

    Reads:  state["grade"], state["attempt"], state["human_mode"]
    Writes: state["current_query"], state["attempt"]

    This node only runs when the grader scored low AND we have retries left.

    WITHOUT --human: takes the grader's suggestion automatically.
    WITH --human:    calls interrupt() to PAUSE the graph and ask the user.

    HOW interrupt() WORKS:
    ----------------------
    interrupt(payload) does three things:
      1. Saves the current graph state to the checkpointer
      2. Sends `payload` back to the caller (so they can show it to the user)
      3. PAUSES execution — the node function stops here

    Later, when the caller does graph.invoke(Command(resume=X), config):
      4. Execution RESUMES right here
      5. interrupt() RETURNS the value X
      6. The rest of the function continues normally

    It's like Python's input(), except it works across processes:
    the graph could be paused on a server and resumed from a web UI.
    """
    suggestion = state["grade"].get("suggestion", "")
    attempt = state["attempt"]

    print(f"\n  📍 NODE: reformulate")

    if state.get("human_mode", False):
        # ===== HUMAN-IN-THE-LOOP =====
        # interrupt() pauses here and sends context to the caller.
        # The caller shows this to the user and collects their input.
        # When resumed, interrupt() returns whatever the user typed.
        human_input = interrupt({
            "suggestion": suggestion,
            "score": state["grade"]["score"],
            "reasoning": state["grade"]["reasoning"],
            "queries_tried": state["queries_tried"],
            "attempt": attempt,
            "max_retries": state["max_retries"],
        })

        # Special case: if the user wants to quit, skip remaining retries.
        # We set attempt = max_retries so should_retry_or_answer() routes
        # to "generate" after the next retrieve→grade cycle.
        if human_input == "__QUIT__":
            print(f"⏹ User quit — will generate answer after one more search.")
            return {
                "current_query": suggestion if suggestion else state["current_query"],
                "attempt": state["max_retries"],  # Force exit after next grade
            }

        # human_input is whatever the user passed via Command(resume=...)
        # Empty string means "accept the suggestion"
        new_query = human_input if human_input else suggestion
        print(f"🔄 Human chose: \"{new_query}\"")
    else:
        # ===== FULLY AUTOMATIC =====
        new_query = suggestion if suggestion else state["current_query"]
        print(f"🔄 Reformulating query...")
        print(f"   Old: \"{state['current_query']}\"")
        print(f"   New: \"{new_query}\"")

    return {
        "current_query": new_query,
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
    parser.add_argument("--human", action="store_true",
                        help="Enable human-in-the-loop: pause and ask before reformulating")
    args = parser.parse_args()

    if args.question is None:
        print("Usage:")
        print('  python graph.py "What message broker did we choose?"')
        print('  python graph.py "How do we know when things break?"')
        print('  python graph.py --human "How do we know when things break?"')
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
        "human_mode": args.human,
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
    print(f"  Human-in-the-loop: {'ON' if args.human else 'OFF'}")
    print(f"{'=' * 60}")

    # --- Run the graph ---
    # invoke() runs the graph from START until it either:
    #   a) Reaches END → returns the final state
    #   b) Hits an interrupt() → returns partial state + __interrupt__ key
    final_state = graph.invoke(initial_state, config)

    # --- Human-in-the-loop resume loop ---
    # After invoke(), check if the graph is PAUSED (hit an interrupt).
    # If so, show the user what the grader suggested and ask for input.
    # Then resume with Command(resume=user_input).
    #
    # graph.get_state(config).next tells us if there are more nodes to run.
    # If it's empty → the graph reached END (we're done).
    # If it's non-empty → the graph is paused at an interrupt.
    while True:
        snapshot = graph.get_state(config)

        if not snapshot.next:
            # Graph reached END — no more nodes to run
            break

        # --- The graph is paused at an interrupt ---
        # The interrupt payload is in snapshot.tasks[0].interrupts[0].value
        # This is the dict we passed to interrupt() in reformulate_node.
        interrupt_data = snapshot.tasks[0].interrupts[0].value

        suggestion = interrupt_data["suggestion"]
        score = interrupt_data["score"]
        reasoning = interrupt_data["reasoning"]
        queries_tried = interrupt_data["queries_tried"]
        attempt = interrupt_data["attempt"]
        max_retries = interrupt_data["max_retries"]

        # Show context to the human
        print(f"\n{'─' * 60}")
        print(f"  🤚 HUMAN-IN-THE-LOOP (attempt {attempt}/{max_retries})")
        print(f"{'─' * 60}")
        print(f"  Score: {score}/5 — {reasoning}")
        print(f"  Queries tried so far: {queries_tried}")
        print(f"  Grader suggests: \"{suggestion}\"")
        print(f"")
        print(f"  Options:")
        print(f"    [Enter]     Accept the suggestion")
        print(f"    [type]      Type your own search query")
        print(f"    [q]         Quit and answer with best results so far")
        print(f"{'─' * 60}")

        user_input = input("  Your choice: ").strip()

        if user_input.lower() == "q":
            # Send __QUIT__ to the reformulate node. It will set attempt
            # to max_retries, so after one more retrieve→grade cycle,
            # should_retry_or_answer routes to generate.
            print("  ⏹ Stopping — will generate answer with best results...")
            final_state = graph.invoke(Command(resume="__QUIT__"), config)
            continue

        # Resume the graph with the user's input (or empty = accept suggestion)
        # Command(resume=X) sends X back to the interrupt() call,
        # which returns X, and the reformulate_node continues.
        resume_value = user_input if user_input else suggestion
        final_state = graph.invoke(Command(resume=resume_value), config)

    # Get the final state (from checkpointer — always up to date)
    final_state = graph.get_state(config).values

    # --- Print the answer ---
    print(f"\n{'=' * 60}")
    print(f"📝 Answer:\n\n{final_state.get('answer', '(no answer generated)')}")
    print(f"{'=' * 60}")

    # --- Show a summary of what happened ---
    print(f"\nSummary:")
    print(f"  Queries tried: {final_state['queries_tried']}")
    print(f"  Best score: {final_state['best_score']}/5")
    print(f"  Attempts used: {final_state['attempt']}/{args.max_retries}")


if __name__ == "__main__":
    main()
