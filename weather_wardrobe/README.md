# Weather & Wardrobe Agent 🌦️👔

A "Hello World" AI agent that checks the weather and recommends an outfit.
Built to teach the **Perception → Reasoning → Action** loop.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your LLM API key (pick one)
export OPENAI_API_KEY="sk-..."        # OpenAI
# or
export ANTHROPIC_API_KEY="sk-ant-..." # Anthropic

# 3. Run it!
python agent.py

# Or with a custom query:
python agent.py "What should I wear in Tokyo today?"
```

## How It Works — The PRA Loop

```
┌──────────────────────────────────────────────────────┐
│  USER: "What should I wear in Beijing today?"        │
└──────────────────┬───────────────────────────────────┘
                   ▼
         ┌─────────────────┐
         │  1. PERCEPTION   │  Agent receives the user query
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  2. REASONING    │  LLM thinks: "I need weather data.
         │                  │  I'll call get_weather('Beijing')."
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  3. ACTION       │  Agent executes the tool call,
         │                  │  gets back: "15°C, Partly cloudy"
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  2. REASONING    │  LLM thinks: "15°C and cloudy →
         │   (again!)       │  light jacket, jeans, bring umbrella"
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  RESPONSE        │  "Wear a light jacket over a
         │                  │   long-sleeve shirt..."
         └─────────────────┘
```

Notice the loop: **Reason → Act → Observe → Reason again**. The agent may loop multiple times if needed.

## Project Structure

```
agent_exercise/
├── agent.py          # The agent (entry point)
├── tools.py          # Weather tool (get_weather)
├── requirements.txt  # Python dependencies
└── README.md         # You are here
```

## What You'll Learn by Building This

### The "Magic" ✨
- The LLM **decides on its own** to call the weather API — you didn't hard-code that
- It interprets raw weather data and gives **creative, contextual** outfit advice
- It handles different cities, conditions, and seasons naturally

### The "Mess" 🐛
Watch for these common agent failure modes:
1. **Tool-skipping**: Agent "hallucinates" weather instead of calling the API
2. **Bad arguments**: Sends malformed city names to the tool
3. **Over-tooling**: Calls the weather API multiple times for no reason
4. **Ignoring results**: Gets weather data but gives generic advice anyway

These failures are *the point* — they teach you **why** prompt engineering, tool descriptions, and guardrails matter.

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `AGENT_MODEL` | `gpt-4o-mini` | LiteLLM model string (e.g. `claude-sonnet-4-20250514`, `gpt-4o`) |

## Next Steps

After playing with this, try:
1. **Add more tools** — a calendar tool so it considers your schedule
2. **Add memory** — let the agent remember your style preferences
3. **Add guardrails** — validate tool outputs before the agent sees them
4. **Try different models** — compare how GPT-4o vs Claude handle the same query
