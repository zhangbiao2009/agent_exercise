# Meeting Notes: Engineering All-Hands

## January 15, 2026

### Budget Review

Total engineering infrastructure spend for 2025: $847,000
- Cloud compute (AWS): $520,000
- Third-party SaaS tools: $187,000
- Kafka managed service: $92,000
- Monitoring (Datadog): $48,000

For 2026, finance approved a 15% increase, bringing the budget to approximately $974,000. The extra budget is allocated to:
- Kubernetes cluster expansion for the new microservices
- Hiring two additional SRE engineers
- Proof of concept for edge computing (CDN-based serverless functions)

### Hiring Update

Currently 42 engineers across 6 teams. Plan to hire 8 more in Q1-Q2 2026:
- 2 SRE / Platform engineers
- 3 Backend engineers for the payment service rewrite
- 2 Frontend engineers for the new dashboard
- 1 ML engineer for the recommendation system

### Tech Debt Discussion

The team raised concerns about:
1. The reporting engine still runs on Python 2.7 — needs migration urgently
2. Frontend test coverage dropped to 34% — below our 60% target
3. The CI pipeline takes 45 minutes — goal is under 15 minutes

Sarah proposed a "Tech Debt Sprint" — dedicate one full sprint per quarter to nothing but tech debt. Vote passed 8-2.

## February 28, 2026

### Q1 Progress Check

Auth service extraction: ✅ Complete. Running in production since Feb 10.
Payment service extraction: 🟡 In progress. 60% complete. Blocked on PCI compliance review.
Reporting engine: 🔴 Not started. Deprioritized in favor of payment service.

### On-Call Burnout

Three engineers reported burnout from on-call rotations. Currently 1-week rotations with a 6-person pool. Proposed changes:
- Expand pool to 10 people (include senior devs from each team)
- Shorten rotation to 3 days
- Add "compensation day off" after each rotation

Decision: Approved. Starts March 2026.

### AI Strategy

CTO presented a plan to integrate LLMs into the product:
- Phase 1: Customer support chatbot using RAG over our knowledge base
- Phase 2: AI-assisted code review for internal PRs
- Phase 3: Recommendation engine powered by embeddings

Budget allocated: $150,000 for AI/ML experiments in 2026. Using DeepSeek and Claude models to keep costs down.
