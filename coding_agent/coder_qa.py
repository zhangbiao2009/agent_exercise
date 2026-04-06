"""
Phase 3: Coder + QA Multi-Agent Orchestration
===============================================
Two agents collaborate in a loop:
  - CODER writes Python code for a given task
  - QA reviews, generates tests, runs the code, and reports back

They iterate until QA passes the code or max attempts are reached.

HOW IT WORKS:
-------------
This is the same LangGraph pattern from Phase 2 (research_agent/graph.py),
but with different actors:

  Phase 2:  retrieve → grade → reformulate (loop)
  Phase 3:  coder   → qa    → (feedback loop back to coder)

The graph:

    ┌──────────┐
    │  START    │
    └────┬─────┘
         ▼
  ┌─────────────┐
  │   coder      │  LLM writes/fixes Python code
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │     qa       │  Execute code + LLM reviews it
  └──────┬──────┘
         ▼
     passed?         Conditional edge
    ┌────┴────┐
   YES       NO
    │         │
    ▼         ▼
┌────────┐    │ (loops back to coder with feedback)
│  END   │ ◄──┘ (or if max attempts hit)
└────────┘

MULTI-AGENT PATTERN:
--------------------
Each "agent" is just a node with a different system prompt:
  - Coder node → system prompt says "you are a Python developer"
  - QA node    → system prompt says "you are a strict QA engineer"

They share state (AgentState) like a shared whiteboard:
  Coder writes to state["code"]
  QA reads state["code"], writes to state["review"] and state["status"]
  Coder reads state["review"] in the next round to fix bugs

This is the "Manager/Loop" orchestration pattern — agents go back
and forth until the work meets quality standards.

Usage:
    export DEEPSEEK_API_KEY="sk-..."

    python coder_qa.py "Write a function fizzbuzz(n) that returns a list"
    python coder_qa.py "Write a function to merge two sorted lists"
    python coder_qa.py --max-attempts 5 "Write a binary search function"
"""

import os
import re
import json
import argparse
import subprocess
import textwrap
from typing import TypedDict

import litellm
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# Part 1: STATE — The shared whiteboard between Coder and QA
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state between the Coder and QA agents.

    Think of this as a desk that both agents sit at:
    - Coder writes code and puts it on the desk
    - QA picks up the code, runs it, writes feedback, puts it back
    - Coder picks up the feedback, fixes the code, repeat
    """
    # --- Inputs (set once) ---
    task: str               # What to build: "Write a function that..."
    model: str              # LLM model string
    max_attempts: int       # Max coder→qa rounds
    strict_mode: bool       # If True, QA enforces extra rules (security, perf, etc.)

    # --- Working state (updated each round) ---
    code: str               # Current Python code (written by Coder)
    test_code: str          # Test cases (written by QA on first round)
    execution_result: str   # stdout/stderr from running the code + tests
    review: str             # QA's feedback (issues found, suggestions)
    status: str             # "needs_work" | "passed" | "error"
    attempt: int            # Current round number
    history: list[dict]     # Past rounds: [{attempt, code, review}, ...]


# ============================================================================
# Part 2: HELPER — Extract code from LLM response
# ============================================================================

def extract_code(response_text: str) -> str:
    """
    Extract Python code from an LLM response.

    LLMs often wrap code in markdown fences like:
        ```python
        def foo():
            return 42
        ```

    This function strips those fences. If there are no fences,
    it returns the raw text (the LLM followed instructions).
    """
    # Look for ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        # Return the longest match (sometimes LLMs output multiple blocks)
        return max(matches, key=len).strip()

    # Look for generic ``` ... ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Handle UNCLOSED fences (LLM hit max_tokens before closing ```)
    # Look for ```python\n... with no closing ```
    pattern = r"```python\s*\n(.*)"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r"```\s*\n(.*)"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences found — return as-is
    return response_text.strip()


def run_code(code: str, test_code: str, workspace: str = "workspace") -> str:
    """
    Save code to a file and execute it. Returns stdout + stderr.

    This is the KEY DIFFERENCE from Phase 2: instead of asking an LLM
    "is this code correct?", we actually RUN it and get real feedback.

    Steps:
    1. Write the code to workspace/solution.py
    2. Write test code to workspace/test_solution.py (imports solution)
    3. Run test_solution.py with a timeout
    4. Return the output (success or error messages)
    """
    # Resolve to absolute path so subprocess.run always finds the files
    workspace = os.path.abspath(workspace)
    os.makedirs(workspace, exist_ok=True)

    solution_path = os.path.join(workspace, "solution.py")
    test_path = os.path.join(workspace, "test_solution.py")

    # Write the solution file
    with open(solution_path, "w") as f:
        f.write(code)

    # Write the test file (imports from solution.py)
    with open(test_path, "w") as f:
        f.write(test_code)

    # Run the tests
    try:
        result = subprocess.run(
            ["python", test_path],
            capture_output=True,
            text=True,
            timeout=10,           # Kill after 10 seconds (catches infinite loops)
            cwd=workspace,        # Run from workspace/ so "import solution" works
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode == 0:
            output += "EXIT CODE: 0 (success)"
        else:
            output += f"EXIT CODE: {result.returncode} (failure)"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out after 10 seconds (possible infinite loop)"


# ============================================================================
# Part 3: CODER NODE — The developer agent
# ============================================================================

def coder_node(state: AgentState) -> dict:
    """
    NODE: The Coder agent writes or fixes Python code.

    Round 1: Writes code from scratch based on the task.
    Round 2+: Reads QA feedback and fixes the bugs.

    The Coder sees:
    - The task description
    - Its previous code (if revising)
    - QA's review with specific issues
    - Full history of past attempts
    """
    task = state["task"]
    attempt = state["attempt"]
    model = state["model"]
    review = state.get("review", "")
    code = state.get("code", "")
    history = state.get("history", [])

    print(f"\n{'━' * 60}")
    print(f"  🧑‍💻 CODER (attempt {attempt}/{state['max_attempts']})")
    print(f"{'━' * 60}")

    # --- System prompt: defines the Coder's personality and rules ---
    system_message = (
        "You are a senior Python developer. Write clean, correct Python code.\n"
        "\n"
        "Rules:\n"
        "1. Output ONLY Python code inside a ```python``` block\n"
        "2. No explanations before or after the code\n"
        "3. Include docstrings and type hints\n"
        "4. Handle edge cases (empty inputs, None, negative numbers, etc.)\n"
        "5. The code must be a complete, runnable Python module\n"
        "6. If fixing bugs, change ONLY what's needed — don't rewrite everything"
    )

    # --- Build the user message based on which round ---
    if attempt == 1:
        # First round: write from scratch
        user_message = f"Write Python code for this task:\n\n{task}"
        print(f"📝 Writing code from scratch...")
    else:
        # Revision round: fix based on QA feedback
        # Include history so Coder can see what it tried before
        history_text = ""
        for h in history:
            history_text += f"\n--- Attempt {h['attempt']} ---\n"
            history_text += f"Code:\n```python\n{h['code']}\n```\n"
            history_text += f"QA Review: {h['review']}\n"

        user_message = (
            f"Task: {task}\n\n"
            f"Your current code:\n```python\n{code}\n```\n\n"
            f"QA Review (MUST FIX):\n{review}\n\n"
            f"Previous attempts:{history_text}\n\n"
            f"Fix the issues. Output ONLY the corrected code in a ```python``` block."
        )
        print(f"🔧 Fixing code based on QA feedback...")
        print(f"   QA said: {review[:100]}...")

    # --- Call the LLM ---
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,     # Low temperature for more consistent code
        max_tokens=2048,
    )

    raw_response = response.choices[0].message.content
    new_code = extract_code(raw_response)

    usage = response.usage
    print(f"   Coder tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, total: {usage.total_tokens}")

    # Show a preview of the code
    lines = new_code.split("\n")
    preview_lines = lines[:8]
    print(f"\n   Code preview ({len(lines)} lines):")
    for line in preview_lines:
        print(f"   │ {line}")
    if len(lines) > 8:
        print(f"   │ ... ({len(lines) - 8} more lines)")

    return {"code": new_code}


# ============================================================================
# Part 4: QA NODE — The tester/reviewer agent
# ============================================================================

def qa_node(state: AgentState) -> dict:
    """
    NODE: The QA agent tests and reviews the code.

    This node does THREE things:

    1. Generate test cases (first round only)
       - Asks the LLM to write pytest-style assertions
       - Saved to workspace/test_solution.py

    2. Execute the code + tests (REAL feedback)
       - Runs the code with subprocess
       - Captures stdout, stderr, exit code, timeouts

    3. LLM review (LOGICAL feedback)
       - Even if code runs, it might be wrong
       - LLM checks correctness, edge cases, code quality

    This combination is powerful:
    - Execution catches: syntax errors, crashes, infinite loops, import errors
    - LLM review catches: logic bugs, missing edge cases, wrong return values
    """
    code = state["code"]
    task = state["task"]
    model = state["model"]
    attempt = state["attempt"]
    test_code = state.get("test_code", "")

    print(f"\n{'━' * 60}")
    print(f"  🔍 QA (reviewing attempt {attempt})")
    print(f"{'━' * 60}")

    # --- Step 1: Generate test cases (first round only) ---
    if not test_code:
        strict = state.get("strict_mode", False)
        print(f"📋 Generating test cases{' (STRICT mode)' if strict else ''}...")

        if strict:
            # STRICT MODE: adversarial tests that are hard to pass first try
            test_prompt = (
                f"You are a HOSTILE QA engineer. Write ADVERSARIAL test code.\n\n"
                f"Task: {task}\n\n"
                f"The solution is in solution.py. Write a test file that:\n"
                f"1. Imports from solution (e.g., 'from solution import function_name')\n"
                f"2. Uses simple assert statements (NOT pytest or unittest)\n"
                f"3. Prints 'ALL TESTS PASSED' at the end if no assertions fail\n"
                f"4. Has at least 10 test cases including:\n"
                f"   - Normal cases\n"
                f"   - Empty inputs ([], '', 0, None)\n"
                f"   - Boundary values (negative numbers, very large numbers, MAX_INT)\n"
                f"   - Type edge cases (float instead of int, bool, string of number)\n"
                f"   - Duplicate values\n"
                f"   - Single-element inputs\n"
                f"   - Already-sorted / reverse-sorted inputs if applicable\n"
                f"   - Performance test: call with a large input (n=10000) and check speed\n"
                f"5. Wrap each test in try/except to show which test failed and why\n\n"
                f"Output ONLY the test code in a ```python``` block."
            )
        else:
            test_prompt = (
                f"You are a QA engineer. Write test code for this task:\n\n"
                f"Task: {task}\n\n"
                f"The solution is in solution.py. Write a test file that:\n"
                f"1. Imports from solution (e.g., 'from solution import function_name')\n"
                f"2. Uses simple assert statements (NOT pytest or unittest)\n"
                f"3. Tests normal cases, edge cases (empty, None, etc.), and boundary cases\n"
                f"4. Prints 'ALL TESTS PASSED' at the end if no assertions fail\n"
                f"5. Has at least 5 test cases\n\n"
                f"Output ONLY the test code in a ```python``` block."
            )

        test_response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You write Python test code. Output ONLY code in a ```python``` block."},
                {"role": "user", "content": test_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        test_code = extract_code(test_response.choices[0].message.content)

        usage = test_response.usage
        print(f"   Test gen tokens — prompt: {usage.prompt_tokens}, "
              f"completion: {usage.completion_tokens}, total: {usage.total_tokens}")

        # Show test preview
        test_lines = test_code.split("\n")
        print(f"\n   Test preview ({len(test_lines)} lines):")
        for line in test_lines[:6]:
            print(f"   │ {line}")
        if len(test_lines) > 6:
            print(f"   │ ... ({len(test_lines) - 6} more lines)")

    # --- Step 2: Execute code + tests ---
    print(f"\n🏃 Running code + tests...")
    execution_result = run_code(code, test_code)

    # Show execution output
    for line in execution_result.split("\n"):
        print(f"   │ {line}")

    # --- Step 3: LLM review ---
    strict = state.get("strict_mode", False)
    print(f"\n📊 LLM reviewing code{' (STRICT mode)' if strict else ''}...")

    if strict:
        # STRICT MODE: enforce extra rules beyond basic correctness
        review_system = (
            "You are a SENIOR QA engineer doing a thorough code review.\n"
            "\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "status": "passed" | "needs_work",\n'
            '  "issues": ["issue 1", "issue 2"],\n'
            '  "summary": "one sentence overall assessment"\n'
            "}\n"
            "\n"
            "You MUST check ALL of the following. Fail if ANY rule is violated:\n"
            "\n"
            "1. CORRECTNESS: Does the code solve the task? Do all tests pass?\n"
            "2. INPUT VALIDATION: Does it validate types at the boundary?\n"
            "   (e.g., raise TypeError/ValueError for wrong input types)\n"
            "3. EDGE CASES: Does it handle empty, None, zero, negative, huge inputs?\n"
            "4. NO BARE EXCEPT: No 'except:' or 'except Exception:' that swallows errors silently\n"
            "5. EFFICIENCY: No obvious O(n²) when O(n) or O(n log n) is possible.\n"
            "   If the task involves sorting/searching, the algorithm must be reasonable.\n"
            "6. DOCSTRING: Must have a docstring with Args, Returns, and at least one Example\n"
            "7. TYPE HINTS: Function signature must have type hints\n"
            "\n"
            "Be SPECIFIC: say exactly what's wrong and how to fix it.\n"
            "Only give 'passed' if ALL rules are satisfied AND all tests pass."
        )
    else:
        review_system = (
            "You are a strict QA engineer. Review the code and test results.\n"
            "\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "status": "passed" | "needs_work",\n'
            '  "issues": ["issue 1", "issue 2"],\n'
            '  "summary": "one sentence overall assessment"\n'
            "}\n"
            "\n"
            "Rules:\n"
            '- "passed" means: code is correct, handles edge cases, tests pass\n'
            '- "needs_work" means: there are bugs, missing edge cases, or test failures\n'
            "- Be SPECIFIC about issues — say exactly what's wrong and how to fix it\n"
            "- If tests passed and code looks correct, give it \"passed\"\n"
            "- Don't nitpick style if the code is functionally correct"
        )

    review_user = (
        f"Task: {task}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Test code:\n```python\n{test_code}\n```\n\n"
        f"Execution result:\n{execution_result}\n\n"
        f"Review the code. Is it correct and complete?"
    )

    review_response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": review_system},
            {"role": "user", "content": review_user},
        ],
        temperature=0,
        max_tokens=500,
    )

    raw_review = review_response.choices[0].message.content.strip()

    usage = review_response.usage
    print(f"   Review tokens — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, total: {usage.total_tokens}")

    # Parse JSON response
    if raw_review.startswith("```"):
        raw_review = raw_review.strip("`").strip()
        if raw_review.startswith("json"):
            raw_review = raw_review[4:].strip()

    try:
        review_data = json.loads(raw_review)
    except json.JSONDecodeError:
        print(f"   ⚠️  QA returned invalid JSON, assuming needs_work")
        review_data = {
            "status": "needs_work",
            "issues": ["Could not parse QA response"],
            "summary": raw_review[:200],
        }

    status = review_data.get("status", "needs_work")
    issues = review_data.get("issues", [])
    summary = review_data.get("summary", "")

    # Format the review as a readable string for the Coder
    review_text = summary
    if issues:
        review_text += "\nIssues:\n" + "\n".join(f"  - {issue}" for issue in issues)

    if status == "passed":
        print(f"\n   ✅ PASSED: {summary}")
    else:
        print(f"\n   ❌ NEEDS WORK: {summary}")
        for issue in issues:
            print(f"      - {issue}")

    # --- Update history ---
    new_history = state.get("history", []) + [{
        "attempt": attempt,
        "code": code,
        "review": review_text,
    }]

    return {
        "test_code": test_code,
        "execution_result": execution_result,
        "review": review_text,
        "status": status,
        "history": new_history,
        "attempt": attempt + 1,      # Increment for next round
    }


# ============================================================================
# Part 5: CONDITIONAL EDGE — Should we loop or stop?
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: After QA reviews, decide what to do.

    Returns:
        "done"   → code passed review or max attempts reached → END
        "revise" → QA found issues, loop back to Coder
    """
    status = state["status"]
    attempt = state["attempt"]      # Already incremented by QA
    max_attempts = state["max_attempts"]

    if status == "passed":
        print(f"\n✅ QA approved the code!")
        return "done"
    elif attempt > max_attempts:
        print(f"\n❌ Max attempts reached ({max_attempts}). Stopping with best effort.")
        return "done"
    else:
        print(f"\n🔄 QA rejected — sending back to Coder (round {attempt}/{max_attempts})")
        return "revise"


# ============================================================================
# Part 6: BUILD THE GRAPH
# ============================================================================

def build_graph():
    """
    Wire the Coder and QA nodes into a LangGraph state machine.

    Same pattern as research_agent/graph.py:
    - Nodes: coder, qa
    - Edges: START→coder→qa→(conditional)→coder or END
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("coder", coder_node)
    builder.add_node("qa", qa_node)

    # Wire edges
    builder.add_edge(START, "coder")          # Start with the Coder
    builder.add_edge("coder", "qa")           # Coder → QA always

    # QA → conditional: either done or back to Coder
    builder.add_conditional_edges(
        "qa",
        should_continue,
        {
            "done": END,          # Passed or max attempts → stop
            "revise": "coder",    # Failed → back to Coder for fixes
        }
    )

    # Compile with checkpointer
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# ============================================================================
# Part 7: CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coder + QA multi-agent system"
    )
    parser.add_argument("task", nargs="?", default=None, help="The coding task")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Max coder→qa rounds (default: 3)")
    parser.add_argument("--model", default=None, help="LiteLLM model string")
    parser.add_argument("--thread", default="default",
                        help="Thread ID for checkpointing")
    parser.add_argument("--strict", action="store_true",
                        help="Strict QA: adversarial tests + enforces input validation, "
                             "efficiency, docstrings, type hints")
    args = parser.parse_args()

    if args.task is None:
        print("Usage:")
        print('  python coder_qa.py "Write a function fizzbuzz(n)"')
        print('  python coder_qa.py --strict "Write a function fizzbuzz(n)"')
        print('  python coder_qa.py --max-attempts 5 "Write a binary search function"')
        return

    model = args.model or os.environ.get("AGENT_MODEL", "deepseek/deepseek-chat")

    graph = build_graph()

    initial_state = {
        "task": args.task,
        "model": model,
        "max_attempts": args.max_attempts,
        "strict_mode": args.strict,
        "code": "",
        "test_code": "",
        "execution_result": "",
        "review": "",
        "status": "needs_work",
        "attempt": 1,
        "history": [],
    }

    config = {"configurable": {"thread_id": args.thread}}

    print(f"\n{'=' * 60}")
    print(f"  Coder + QA Multi-Agent System")
    print(f"  Model: {model}")
    print(f"  Max attempts: {args.max_attempts}")
    print(f"  Strict QA: {'ON' if args.strict else 'OFF'}")
    print(f"{'=' * 60}")
    print(f"\n📋 Task: {args.task}")

    # Run the graph
    final_state = graph.invoke(initial_state, config)

    # --- Print final results ---
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"\n  Status: {'✅ PASSED' if final_state['status'] == 'passed' else '❌ BEST EFFORT'}")
    print(f"  Attempts used: {final_state['attempt'] - 1}/{args.max_attempts}")
    print(f"\n  Final code (saved to workspace/solution.py):")
    print(f"{'─' * 60}")
    for line in final_state["code"].split("\n"):
        print(f"  {line}")
    print(f"{'─' * 60}")

    if final_state["status"] != "passed":
        print(f"\n  Last QA review: {final_state['review']}")


if __name__ == "__main__":
    main()


# ============================================================================
# TEST CASES & RESULTS
# ============================================================================
#
# --- Normal mode (no --strict) ---
#
# Test 1: FizzBuzz (needs one fix)
#   $ python coder_qa.py "Write a function fizzbuzz(n)..."
#   Round 1: Coder raised ValueError for n=0, QA tests expected [] → FAIL
#   Round 2: Coder fixed (returns [] for n<1). All 7 tests passed
#   Result: ✅ Passed in 2/3 attempts
#
# Test 2: Merge sorted lists (first-try pass)
#   $ python coder_qa.py "Write a function merge_sorted(list1, list2)..."
#   Round 1: Two-pointer merge, handled None/empty. All tests passed
#   Result: ✅ Passed in 1/3 attempts
#
# Test 3: Flatten nested list (first-try pass)
#   $ python coder_qa.py "Write a function flatten(nested)..."
#   Round 1: Recursive flatten. All tests passed
#   Result: ✅ Passed in 1/3 attempts
#
# --- Strict mode (--strict) ---
#
# Test 4: FizzBuzz strict (full 3-round loop)
#   $ python coder_qa.py --strict "Write a function fizzbuzz(n)..."
#   Round 1: 12 adversarial tests generated. 5 failed (ValueError vs empty
#            list for n=0, float 5.0, string '5'). QA rejected.
#   Round 2: Coder converted float/string to int. All 12 tests passed.
#            QA still rejected: "type hint says int but accepts float/string"
#   Round 3: Coder reverted to strict int-only. Tests 6,7 failed again
#            (contradiction between tests and QA review).
#   Result: ❌ Best effort after 3/3 attempts
#   Note: This shows realistic spec ambiguity — QA tests and QA review
#         disagree on whether to accept float/string inputs!
#
# Test 5: Merge sorted strict (full 3-round loop)
#   $ python coder_qa.py --strict "Write a function merge_sorted(list1, list2)..."
#   All 3 rounds used. QA demanded input type validation, then changed
#   its mind about how strict it should be. Similar spec ambiguity.
#   Result: ❌ Best effort after 3/3 attempts
#
# BUGS FIXED DURING DEVELOPMENT:
# - workspace path double-nesting: run_code used relative path, subprocess
#   cwd also relative → workspace/workspace/. Fixed with os.path.abspath().
# - Unclosed markdown fences: LLM hit max_tokens mid-code-block, leaving
#   ```python with no closing ```. extract_code regex failed silently.
#   Fixed by adding fallback regex for unclosed fences.
# - max_tokens too low for strict test generation: bumped from 1024 to 2048.
#
# KEY OBSERVATIONS:
# - Normal mode: LLMs write correct code first-try for simple tasks,
#   making the loop easy to miss. The loop mainly helps with edge case
#   disagreements (ValueError vs empty list).
# - Strict mode: Reliably triggers multi-round loops by demanding input
#   validation, type hints, docstrings, efficiency, and adversarial tests.
# - Interesting tension: QA-generated tests and QA review can DISAGREE
#   (tests expect float→int conversion, review says "don't accept floats").
#   This mirrors real-world spec ambiguity between test suites and reviewers.
# - Real execution is essential: catches crashes that LLM review misses.
# - The loop cap (max_attempts) prevents infinite disagreement loops.
