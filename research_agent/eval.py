"""
Phase 4: Automated Evaluation Pipeline
========================================
Runs a gold dataset of 20 questions through the research agent,
uses LLM-as-a-judge to score each answer, and produces a report.

WHY EVALS?
----------
When you change a prompt, model, or retrieval strategy, you need to
know if the agent got BETTER or just DIFFERENT. Manual testing doesn't
scale. This script gives you a single number: "17/20 passed."

HOW IT WORKS:
-------------
1. Load 20 questions from eval_dataset.json (with expected answers)
2. Run each question through the research agent (graph.py)
3. Use a second LLM call ("judge") to score each answer on:
   - Correctness: Does it contain the key facts?
   - Faithfulness: Is it grounded in the docs (no hallucination)?
   - Completeness: Does it cover the full expected answer?
4. Print a summary report + save results for comparison

For UNANSWERABLE questions (no answer in docs), the scoring rubric
changes: the agent SHOULD say "not found" — hallucinating an answer
is the worst outcome.

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    python eval.py                  # Run all 20 questions
    python eval.py --id 5           # Run just question #5
    python eval.py --category hard  # Run only "hard" questions
    python eval.py --save           # Save results to eval_results/
"""

import os
import json
import argparse
import time
from datetime import datetime

import litellm

# Import the graph builder and retriever from our research agent
from graph import build_graph, _get_retriever


# ============================================================================
# Part 1: RUNNER — Execute the agent on a question
# ============================================================================

def run_question(graph, question: str, model: str, thread_id: str) -> dict:
    """
    Run a single question through the research agent graph.

    Returns a dict with the agent's answer and metadata.
    Same logic as graph.py's main(), but called as a function.
    """
    initial_state = {
        "query": question,
        "model": model,
        "top_k": 5,
        "max_retries": 3,
        "grade_threshold": 3,
        "human_mode": False,
        "current_query": question,
        "chunks": [],
        "best_chunks": [],
        "best_score": 0,
        "grade": {},
        "queries_tried": [],
        "attempt": 1,
        "answer": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    start_time = time.time()
    final_state = graph.invoke(initial_state, config)
    elapsed = time.time() - start_time

    return {
        "answer": final_state.get("answer", ""),
        "best_score": final_state.get("best_score", 0),
        "attempts": final_state.get("attempt", 1),
        "queries_tried": final_state.get("queries_tried", []),
        "elapsed_seconds": round(elapsed, 1),
    }


# ============================================================================
# Part 2: JUDGE — LLM-as-a-Judge scoring
# ============================================================================

def judge_answerable(question: str, expected: str, actual: str,
                     expected_source: str, model: str) -> dict:
    """
    Judge an answer for a question that HAS an answer in the docs.

    Scores on 3 dimensions (1-5):
    - correctness: Does it contain the key facts?
    - faithfulness: Is it grounded in docs (no hallucination)?
    - completeness: Does it cover the full expected answer?
    """
    system_message = (
        "You are an evaluation judge. Score the agent's answer by comparing\n"
        "it to the expected answer.\n"
        "\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "correctness": <1-5>,\n'
        '  "faithfulness": <1-5>,\n'
        '  "completeness": <1-5>,\n'
        '  "reasoning": "<one sentence explaining your scores>"\n'
        "}\n"
        "\n"
        "Scoring guide:\n"
        "\n"
        "CORRECTNESS (does it contain the key facts?):\n"
        "  5 = All key facts from expected answer are present\n"
        "  4 = Most key facts present, minor omissions\n"
        "  3 = Some key facts present, noticeable gaps\n"
        "  2 = Few key facts, mostly wrong or irrelevant\n"
        "  1 = Completely wrong or no answer given\n"
        "\n"
        "FAITHFULNESS (is it grounded in docs, no hallucination?):\n"
        "  5 = Every claim appears in the source docs\n"
        "  4 = Mostly grounded, one minor unsupported detail\n"
        "  3 = Some grounded, some speculative\n"
        "  2 = Significant hallucination\n"
        "  1 = Entirely made up\n"
        "\n"
        "COMPLETENESS (does it cover the full expected answer?):\n"
        "  5 = Covers all aspects of the expected answer\n"
        "  4 = Covers most, one minor gap\n"
        "  3 = Covers about half\n"
        "  2 = Only a small fragment\n"
        "  1 = Essentially empty or off-topic\n"
        "\n"
        "Be fair: the agent's answer doesn't need to be word-for-word.\n"
        "Different phrasing is fine as long as the key facts match."
    )

    user_message = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Expected source: {expected_source}\n\n"
        f"Agent's actual answer:\n{actual}\n\n"
        f"Score the agent's answer."
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"correctness": 0, "faithfulness": 0, "completeness": 0,
                  "reasoning": f"Judge returned invalid JSON: {raw[:100]}"}

    scores.setdefault("correctness", 0)
    scores.setdefault("faithfulness", 0)
    scores.setdefault("completeness", 0)
    scores.setdefault("reasoning", "")

    return scores


def judge_unanswerable(question: str, actual: str, model: str) -> dict:
    """
    Judge an answer for a question that has NO answer in the docs.

    The agent SHOULD say "not found" or similar. Hallucinating an answer
    is the worst possible outcome.
    """
    system_message = (
        "You are an evaluation judge. The question below has NO answer in the\n"
        "source documents. The agent SHOULD say it couldn't find the answer.\n"
        "\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "correctness": <1-5>,\n'
        '  "faithfulness": <1-5>,\n'
        '  "completeness": <1-5>,\n'
        '  "reasoning": "<one sentence explaining your scores>"\n'
        "}\n"
        "\n"
        "Scoring guide (ALL three dimensions use the same rubric here):\n"
        "  5 = Agent clearly says it couldn't find the answer / no relevant info\n"
        "  4 = Agent hedges but mostly acknowledges it can't answer\n"
        "  3 = Agent gives vaguely related info but doesn't directly answer\n"
        "  2 = Agent attempts to answer with unrelated information\n"
        "  1 = Agent confidently hallucinated a specific answer\n"
    )

    user_message = (
        f"Question (UNANSWERABLE — not in the docs): {question}\n\n"
        f"Agent's actual answer:\n{actual}\n\n"
        f"Score how well the agent handled this unanswerable question."
    )

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"correctness": 0, "faithfulness": 0, "completeness": 0,
                  "reasoning": f"Judge returned invalid JSON: {raw[:100]}"}

    scores.setdefault("correctness", 0)
    scores.setdefault("faithfulness", 0)
    scores.setdefault("completeness", 0)
    scores.setdefault("reasoning", "")

    return scores


# ============================================================================
# Part 3: EVAL PIPELINE — Run all questions, score, report
# ============================================================================

PASS_THRESHOLD = 4   # All three scores must be >= this to "pass"


def passes(scores: dict) -> bool:
    """A question passes if ALL three scores are >= PASS_THRESHOLD."""
    return (scores.get("correctness", 0) >= PASS_THRESHOLD and
            scores.get("faithfulness", 0) >= PASS_THRESHOLD and
            scores.get("completeness", 0) >= PASS_THRESHOLD)


def run_eval(dataset: list[dict], model: str,
             filter_id: int = None, filter_category: str = None) -> list[dict]:
    """
    Run the full eval pipeline.

    1. Build the graph once
    2. For each question: run agent → judge answer → record scores
    3. Return list of results
    """
    # Build graph once (reuses embedding model across all questions)
    print("⏳ Building graph and loading retriever...")
    graph = build_graph()
    _get_retriever()  # Pre-load embedding model

    # Filter dataset if needed
    questions = dataset
    if filter_id is not None:
        questions = [q for q in dataset if q["id"] == filter_id]
    if filter_category is not None:
        questions = [q for q in dataset if q["category"] == filter_category]

    if not questions:
        print("❌ No questions match the filter.")
        return []

    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        category = q["category"]
        answerable = q["answerable"]

        print(f"\n{'━' * 60}")
        print(f"  [{i}/{total}] Q{qid} ({category}): {question[:50]}...")
        print(f"{'━' * 60}")

        # --- Run the agent ---
        # Each question gets a unique thread to avoid state leakage
        thread_id = f"eval-q{qid}-{int(time.time())}"
        agent_result = run_question(graph, question, model, thread_id)

        print(f"\n  Agent answer: {agent_result['answer'][:100]}...")
        print(f"  Retriever score: {agent_result['best_score']}/5, "
              f"attempts: {agent_result['attempts']}, "
              f"time: {agent_result['elapsed_seconds']}s")

        # --- Judge the answer ---
        print(f"  Judging...")
        if answerable:
            scores = judge_answerable(
                question, q["expected_answer"], agent_result["answer"],
                q["expected_source"], model,
            )
        else:
            scores = judge_unanswerable(question, agent_result["answer"], model)

        passed = passes(scores)
        icon = "✅" if passed else "❌"
        print(f"  {icon} correctness={scores['correctness']}, "
              f"faithfulness={scores['faithfulness']}, "
              f"completeness={scores['completeness']}")
        print(f"  Judge: {scores['reasoning']}")

        results.append({
            "id": qid,
            "question": question,
            "category": category,
            "answerable": answerable,
            "expected_answer": q.get("expected_answer", ""),
            "expected_source": q.get("expected_source", ""),
            "agent_answer": agent_result["answer"],
            "agent_best_score": agent_result["best_score"],
            "agent_attempts": agent_result["attempts"],
            "agent_time": agent_result["elapsed_seconds"],
            "queries_tried": agent_result["queries_tried"],
            "scores": scores,
            "passed": passed,
        })

    return results


# ============================================================================
# Part 4: REPORT — Print summary and details
# ============================================================================

def print_report(results: list[dict], model: str):
    """Print a summary report of eval results."""
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    # --- By category ---
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    # --- Averages ---
    avg_correct = sum(r["scores"]["correctness"] for r in results) / total if total else 0
    avg_faithful = sum(r["scores"]["faithfulness"] for r in results) / total if total else 0
    avg_complete = sum(r["scores"]["completeness"] for r in results) / total if total else 0
    avg_time = sum(r["agent_time"] for r in results) / total if total else 0

    print(f"\n{'═' * 60}")
    print(f"  EVAL REPORT — Research Agent")
    print(f"  Model: {model}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Pass threshold: all scores >= {PASS_THRESHOLD}")
    print(f"{'═' * 60}")

    print(f"\n  OVERALL: {passed_count}/{total} passed "
          f"({100 * passed_count // total if total else 0}%)")

    print(f"\n  By category:")
    for cat_order in ["easy", "medium", "hard", "unanswerable"]:
        if cat_order in categories:
            c = categories[cat_order]
            pct = 100 * c["passed"] // c["total"] if c["total"] else 0
            print(f"    {cat_order:14s} {c['passed']}/{c['total']}  ({pct}%)")

    print(f"\n  Average scores:")
    print(f"    Correctness:  {avg_correct:.1f}/5")
    print(f"    Faithfulness: {avg_faithful:.1f}/5")
    print(f"    Completeness: {avg_complete:.1f}/5")
    print(f"    Avg time:     {avg_time:.1f}s per question")

    # --- Failed questions ---
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  Failed questions:")
        for r in failed:
            s = r["scores"]
            print(f"\n    #{r['id']} [{r['category']}] \"{r['question'][:60]}\"")
            print(f"        Scores: correctness={s['correctness']}, "
                  f"faithfulness={s['faithfulness']}, "
                  f"completeness={s['completeness']}")
            print(f"        Judge: {s['reasoning']}")
            if r["answerable"]:
                print(f"        Expected: {r['expected_answer'][:80]}...")
            print(f"        Got: {r['agent_answer'][:80]}...")
    else:
        print(f"\n  🎉 All questions passed!")

    print(f"\n{'═' * 60}")


def save_results(results: list[dict], model: str):
    """Save results to eval_results/ for later comparison."""
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"eval_results/run_{timestamp}.json"

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    output = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "total": total,
            "passed": passed_count,
            "pass_rate": f"{100 * passed_count // total if total else 0}%",
            "pass_threshold": PASS_THRESHOLD,
        },
        "results": results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to {filename}")
    return filename


# ============================================================================
# Part 5: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Eval pipeline for the research agent"
    )
    parser.add_argument("--id", type=int, default=None,
                        help="Run only this question ID")
    parser.add_argument("--category", default=None,
                        choices=["easy", "medium", "hard", "unanswerable"],
                        help="Run only this category")
    parser.add_argument("--model", default=None, help="LiteLLM model string")
    parser.add_argument("--save", action="store_true",
                        help="Save results to eval_results/")
    parser.add_argument("--dataset", default="eval_dataset.json",
                        help="Path to eval dataset JSON")
    args = parser.parse_args()

    model = args.model or os.environ.get("AGENT_MODEL", "deepseek/deepseek-chat")

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)

    print(f"📊 Loaded {len(dataset)} questions from {args.dataset}")

    # Run eval
    results = run_eval(dataset, model,
                       filter_id=args.id, filter_category=args.category)

    if not results:
        return

    # Print report
    print_report(results, model)

    # Save if requested
    if args.save:
        save_results(results, model)


if __name__ == "__main__":
    main()
