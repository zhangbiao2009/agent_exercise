"""
Step 3: Agentic RAG — Grader + Reformulation Loop
==================================================
This script adds a "quality check" on top of the basic RAG from Step 2.

Instead of blindly trusting whatever the vector search returns, we now:
    1. SEARCH   — same as Step 2
    2. GRADE    — ask the LLM: "Are these results good enough to answer?"
    3. DECIDE   — if good → answer. If bad → reformulate the query and retry.
    4. ANSWER   — generate the final response (or say "not found")

This is what makes it "agentic" — the agent evaluates its own results
and decides whether to try again with a different strategy.

The loop:
    ┌─→ [Search] → [Grade] → score >= 3? → YES → [Answer]
    │                │
    │                NO (score < 3)
    │                │
    └── [Reformulate query using grader's suggestion]
        (max 3 attempts, then give up honestly)

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    # Agentic RAG (with grading + retry loop)
    python grader.py "What caused the Q3 outage?"

    # Compare with basic RAG (no grading)
    python retriever.py "What caused the Q3 outage?"
"""

import os
import json
import argparse

import litellm

# Reuse the Retriever class and build_prompt from Step 2
# No need to rewrite — we import and build on top of it
from retriever import Retriever, build_prompt


# ============================================================================
# Part 1: GRADE — Evaluate whether retrieved chunks answer the question
# ============================================================================

def grade_results(query: str, chunks: list[dict], model: str, queries_tried: list[str] = None) -> dict:
    """
    Ask the LLM to judge: "Do these chunks actually answer the user's question?"

    This is a SEPARATE LLM call from the final answer. It's cheap (~300 tokens)
    and acts as a quality gate.

    Args:
        query:          The user's original question
        chunks:         The retrieved chunks from vector search
        model:          LiteLLM model string
        queries_tried:  List of queries already attempted (so the grader
                        avoids suggesting similar ones)

    Returns:
        A dict with three fields:
        {
            "score": 1-5,         # 1 = irrelevant, 5 = perfect match
            "reasoning": "...",   # why the grader gave this score
            "suggestion": "..."   # a better query to try (if score < 3)
        }
    """

    # --- Build the grading prompt ---
    # The system message tells the LLM its role: you're a judge, not an answerer.
    # We ask for JSON output so we can parse the score programmatically.
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

    # Format the chunks for the grader to review
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        if chunk["heading"]:
            source += f", § {chunk['heading']}"
        chunks_text += f"[{i}] (source: {source})\n{chunk['text']}\n\n"

    # If we've already tried other queries, tell the grader so it avoids
    # suggesting similar things. This prevents the loop from getting stuck
    # retrying slight variations of the same bad query.
    tried_text = ""
    if queries_tried:
        tried_text = (
            f"\nPrevious queries that didn't work well:\n"
            + "\n".join(f"  - \"{q}\"" for q in queries_tried)
            + "\nDo NOT suggest anything similar to these.\n"
        )

    user_message = (
        f"Question: {query}\n\n"
        f"Retrieved chunks:\n"
        f"{chunks_text}"
        f"{tried_text}"
        f"Grade these chunks for relevance to the question."
    )

    # --- Call the LLM ---
    # temperature=0 for consistent, deterministic grading
    # max_tokens=200 because the response is just a small JSON object
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

    # --- Parse the JSON response ---
    # LLMs sometimes wrap JSON in ```json ... ```, so we strip that
    if raw_content.startswith("```"):
        # Remove markdown code fences: ```json\n{...}\n```
        raw_content = raw_content.strip("`").strip()
        if raw_content.startswith("json"):
            raw_content = raw_content[4:].strip()

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        # If the LLM didn't return valid JSON, assume a middle score
        # and let the pipeline continue. This is a safety fallback.
        print(f"   ⚠️  Grader returned invalid JSON, assuming score=3")
        print(f"   Raw response: {raw_content[:200]}")
        result = {"score": 3, "reasoning": "Could not parse grader response", "suggestion": ""}

    # Ensure all expected fields exist (defensive coding)
    result.setdefault("score", 3)
    result.setdefault("reasoning", "")
    result.setdefault("suggestion", "")

    # Show token usage for the grading call
    usage = response.usage
    print(f"   Grader tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, "
          f"total: {usage.total_tokens}")

    return result


# ============================================================================
# Part 2: AGENTIC ASK — The retry loop with grading
# ============================================================================

def agentic_ask(
    query: str,
    retriever: Retriever,
    model: str = "deepseek/deepseek-chat",
    top_k: int = 5,
    max_retries: int = 3,
    grade_threshold: int = 3,
) -> str:
    """
    The full agentic RAG pipeline with grading and query reformulation.

    Unlike the basic `ask()` from Step 2, this function:
    1. Searches the vector DB
    2. Grades the results (are they good enough?)
    3. If not good enough, reformulates the query and tries again
    4. After max_retries, gives up honestly

    Args:
        query:            The user's original question
        retriever:        A Retriever instance (connected to the vector DB)
        model:            LiteLLM model string
        top_k:            Number of chunks to retrieve per search
        max_retries:      Maximum number of search attempts (default 3)
        grade_threshold:  Minimum score to accept results (default 3)

    Returns:
        The LLM's answer as a string, or a "not found" message.
    """
    # Keep track of all queries we've tried (the original + reformulations)
    # and the best chunks we've found across all attempts.
    queries_tried = []
    best_chunks = None
    best_score = 0

    current_query = query

    for attempt in range(1, max_retries + 1):
        print(f"\n{'━' * 60}")
        print(f"  Attempt {attempt}/{max_retries}")
        print(f"{'━' * 60}")

        # --- Step 1: Search ---
        print(f"\n🔍 Searching for: \"{current_query}\"")
        chunks = retriever.search(current_query, top_k=top_k)
        queries_tried.append(current_query)

        # Show retrieved chunks
        for i, chunk in enumerate(chunks, 1):
            preview = chunk["text"][:80].replace("\n", " ") + "..."
            print(f"   [{i}] dist={chunk['distance']:.3f}  {chunk['source']}")
            print(f"       {preview}")

        # --- Step 2: Grade ---
        print(f"\n📊 Grading results...")
        grade = grade_results(query, chunks, model, queries_tried=queries_tried)
        #                     ^^^^^ Note: we grade against the ORIGINAL query,
        #                     not the reformulated one. The user's actual question
        #                     is what matters, not our rephrased search terms.

        score = grade["score"]
        reasoning = grade["reasoning"]
        suggestion = grade["suggestion"]

        print(f"   Score: {score}/5")
        print(f"   Reasoning: {reasoning}")

        # --- Keep track of the best results so far ---
        # Even if we retry, we might want to use earlier (better) results
        if score > best_score:
            best_score = score
            best_chunks = chunks

        # --- Step 3: Decide — good enough or retry? ---
        if score >= grade_threshold:
            # ✅ Results are good enough — proceed to answer
            print(f"\n✅ Score {score} >= threshold {grade_threshold}. Generating answer...\n")
            break
        else:
            # ❌ Results are not good enough
            if attempt < max_retries:
                # We have retries left — reformulate and try again
                if suggestion:
                    print(f"\n🔄 Score {score} < {grade_threshold}. Reformulating...")
                    print(f"   Grader suggests: \"{suggestion}\"")
                    current_query = suggestion
                else:
                    # Grader didn't suggest anything — we're stuck
                    print(f"\n🔄 Score {score} < {grade_threshold}. No suggestion, keeping query.")
            else:
                # No retries left — give up
                print(f"\n❌ Max retries reached. Best score was {best_score}/5.")

    # --- Step 4: Generate the final answer ---
    # Use the best chunks we found across all attempts
    if best_score < 2:
        # Score 1 = totally irrelevant. Don't even try to answer.
        # This prevents the LLM from stretching irrelevant context into a
        # plausible-sounding but wrong answer.
        answer = (
            "I couldn't find relevant information in the documents to answer "
            f"your question: \"{query}\"\n\n"
            f"Queries tried: {queries_tried}\n"
            f"Best relevance score: {best_score}/5"
        )
        print(f"\n📝 No relevant results found.\n")
        return answer

    # Build the prompt using the best chunks and call the LLM for the answer
    print(f"🤖 Generating answer from best results (score {best_score}/5)...\n")
    messages = build_prompt(query, best_chunks)

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # Show token usage for the answer generation
    usage = response.usage
    print(f"   Answer tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, "
          f"total: {usage.total_tokens}\n")

    return answer


# ============================================================================
# Part 3: CLI — Run from the command line
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG: search with grading and query reformulation"
    )
    parser.add_argument("question", nargs="?", default=None, help="The question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-retries", type=int, default=3, help="Max search attempts")
    parser.add_argument("--threshold", type=int, default=3, help="Min grader score to accept (1-5)")
    parser.add_argument("--model", default=None, help="LiteLLM model string")
    parser.add_argument("--db", default="data/lancedb", help="Path to LanceDB database")
    args = parser.parse_args()

    if args.question is None:
        print("Usage:")
        print('  python grader.py "What message broker did we choose?"')
        print('  python grader.py "What\'s our mobile app roadmap?"')
        print('  python grader.py --threshold 4 "What caused the Q3 outage?"')
        return

    model = args.model or os.environ.get("AGENT_MODEL", "deepseek/deepseek-chat")

    print("⏳ Loading retriever...")
    retriever = Retriever(db_path=args.db)

    answer = agentic_ask(
        query=args.question,
        retriever=retriever,
        model=model,
        top_k=args.top_k,
        max_retries=args.max_retries,
        grade_threshold=args.threshold,
    )

    print("=" * 60)
    print(f"📝 Answer:\n\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ============================================================================
# TEST CASES & CONCLUSIONS
# ============================================================================
#
# Run these commands to see the grader in action (activate venv first):
#   source .venv/bin/activate
#   export DEEPSEEK_API_KEY="your-key-here"
#
# --------------------------------------------------------------------------
# Test 1: Easy question — answer exists with matching wording
# --------------------------------------------------------------------------
#   python grader.py "What message broker did we choose?"
#
#   Result: Score 5/5 on attempt 1. No reformulation needed.
#   Answer: "Apache Kafka" (source: architecture-decisions.md)
#   Tokens: ~1900 total (grader ~1019 + answer ~890)
#
#   Conclusion: When the query wording closely matches the document text,
#   retrieval works on the first try. The grader confirms and passes through.
#
# --------------------------------------------------------------------------
# Test 2: No answer exists — info is not in the documents
# --------------------------------------------------------------------------
#   python grader.py "What's our mobile app roadmap?"
#
#   Result: Score 2 → 2 → 1 across 3 attempts. Reformulations tried:
#     Attempt 1: original query → score 2
#       Suggestion: "mobile app feature roadmap 2025 release schedule"
#     Attempt 2: reformulated → score 2
#       Suggestion: "mobile app development timeline..."
#     Attempt 3: reformulated again → score 1
#     Gave up honestly: "I couldn't find this in the documents."
#   Tokens: ~3360 total (3 grader calls + 1 answer call)
#
#   Conclusion: When information genuinely doesn't exist, the grader
#   correctly scores low on every attempt. The agent gives up honestly
#   instead of hallucinating. The retry cost (~3360 tokens) is the price
#   of honesty — worth it vs. a confident wrong answer.
#
# --------------------------------------------------------------------------
# Test 3: Vague question — answer exists but wording doesn't match
# --------------------------------------------------------------------------
#   python grader.py "How do we know when things break?"
#
#   Result: Score 2 → 4 (improved with reformulation!)
#     Attempt 1: "How do we know when things break?"
#       Top result: Circuit Breaker Pattern (dist 1.400) — about failure
#       *handling*, not failure *detection*. Monitoring Stack was #4 (dist 1.715).
#       Score: 2/5
#       Suggestion: "monitoring alerting detection system failures indicators"
#     Attempt 2: reformulated query
#       Top result: Monitoring Stack (dist 1.076) — jumped from #4 to #1!
#       Score: 4/5 ✅
#   Answer: Prometheus, Grafana, Loki, PagerDuty for P1/P2, Slack for P3/P4.
#
#   Conclusion: THIS is the key reformulation win. The vague word "break"
#   matched Circuit Breaker (wrong topic). The grader recognized the mismatch
#   and suggested precise terms like "monitoring" and "alerting." The
#   Monitoring Stack chunk distance dropped from 1.715 → 1.076.
#   Same documents, better query.
#
# --------------------------------------------------------------------------
# Test 4 (try this!): Cross-document question
# --------------------------------------------------------------------------
#   python grader.py "What caused the outage and what changes did we make?"
#
#   Expected: Should pull from incident-q3-2025.md (cause) AND
#   architecture-decisions.md (response decisions). Tests whether the
#   retriever finds chunks across multiple files.
#
# --------------------------------------------------------------------------
# Test 5 (try this!): Strict threshold
# --------------------------------------------------------------------------
#   python grader.py --threshold 5 "How much do we spend on infrastructure?"
#
#   Expected: With threshold=5, only perfect matches pass. The grader might
#   score 4 on first try (budget info is in meeting-notes.md but spread
#   across sections), triggering a reformulation to find more precise chunks.
#
# --------------------------------------------------------------------------
# KEY OBSERVATIONS
# --------------------------------------------------------------------------
#
# 1. Grader call is cheap: ~800 tokens per grading (~$0.0001 with DeepSeek)
#
# 2. Reformulation helps MOST when:
#    - User's wording doesn't match document wording (vocabulary mismatch)
#    - Query is too vague, matching wrong topics by accident
#
# 3. Reformulation does NOT help when:
#    - Information genuinely doesn't exist in the documents
#    - Chunking split the relevant content badly (fix in ingest.py instead)
#
# 4. The grader grades against the ORIGINAL query, not the reformulated one.
#    This prevents drift: reformulated searches should still serve the
#    original question.
#
# 5. best_chunks tracking matters: attempt 2 might score worse than attempt 1
#    if the suggestion was bad. We keep the best results across all attempts.
