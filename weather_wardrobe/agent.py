"""
Weather & Wardrobe Agent
========================
A "Hello World" agent that demonstrates the Perception-Reasoning-Action loop.

  Perception : User says "What should I wear in Beijing today?"
  Reasoning  : LLM decides it needs weather data → plans a tool call
  Action     : Agent calls get_weather("Beijing"), then reasons about the
               result and recommends an outfit.

Usage:
    # Set your API key (pick ONE provider):
    export OPENAI_API_KEY="sk-..."       # for OpenAI
    # OR
    export ANTHROPIC_API_KEY="sk-..."    # for Anthropic
    # OR
    export DEEPSEEK_API_KEY="sk-..."     # for DeepSeek

    # Optionally choose a model (defaults to gpt-4o-mini):
    export AGENT_MODEL="deepseek/deepseek-chat"

    python agent.py
"""

import os
import sys

from smolagents import CodeAgent, LiteLLMModel
from tools import get_weather


def build_agent():
    """Create the Weather & Wardrobe agent."""

    # --- Choose the LLM backend ---
    # smolagents uses litellm under the hood, so any model string works.
    # Default to a cheap, fast model. Override with env var if you like.
    #
    # Supported providers (set the matching env var):
    #   OPENAI_API_KEY      → model: "gpt-4o-mini", "gpt-4o"
    #   ANTHROPIC_API_KEY   → model: "anthropic/claude-sonnet-4-20250514"
    #   DEEPSEEK_API_KEY    → model: "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"
    #
    # LiteLLM picks up the API key from the environment automatically.
    model_id = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    print(f"Using model: {model_id}")

    model = LiteLLMModel(model_id=model_id)

    # --- Instructions: give the agent its "personality" ---
    instructions = (
        "You are a helpful wardrobe assistant. "
        "When the user asks what to wear, ALWAYS use the get_weather tool first "
        "to check the current weather for their city. "
        "Then recommend a practical, stylish outfit based on the conditions. "
        "Be specific: mention clothing items, layers, and accessories like "
        "umbrellas or sunglasses when appropriate."
    )

    # --- Assemble the agent ---
    agent = CodeAgent(
        tools=[get_weather],
        model=model,
        instructions=instructions,
    )
    return agent


def main():
    agent = build_agent()

    # Default query — change the city to yours!
    default_query = "What should I wear in Beijing today?"
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_query

    print(f"\n🗣️  You: {query}\n")
    print("=" * 60)

    result = agent.run(query)

    print("=" * 60)
    print(f"\n🤖 Agent: {result}\n")


if __name__ == "__main__":
    main()
