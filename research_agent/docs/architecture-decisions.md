# Architecture Decisions Log

## 2025-09-15: Breaking Up the Monolith

After the Q3 outage that took down the entire platform for 4 hours, we decided to decompose our main application into smaller, independently deployable services. Each team will own a "bounded context" based on domain-driven design principles.

The migration plan:
1. Extract the authentication module first (lowest risk, clearest boundary)
2. Move the payment processing next (highest business value)
3. Leave the reporting engine for last (most coupled to everything)

Timeline: estimated 6 months, starting October 2025.

## 2025-10-02: Choosing a Message Broker

We evaluated three options for inter-service communication:
- **RabbitMQ**: Mature, good for task queues, but limited throughput at scale
- **Apache Kafka**: High throughput, event sourcing friendly, but operationally complex
- **NATS**: Lightweight, fast, but smaller ecosystem

Decision: Go with Kafka. The event sourcing capability is critical for our audit trail requirements. The ops complexity is manageable since we already run Kubernetes.

## 2025-11-20: API Gateway Selection

Evaluated Kong, Envoy, and AWS API Gateway. Selected Envoy because:
- Native gRPC support (we're migrating internal APIs to gRPC)
- Better observability with built-in Prometheus metrics
- Service mesh integration via Istio if we need it later

Cost estimate: $0 for the proxy itself (open source), ~$200/month additional compute.

## 2026-01-10: Database Strategy

Each service gets its own database (database-per-service pattern). This means:
- Auth service: PostgreSQL (relational data, strong consistency)
- Payment service: PostgreSQL with row-level security
- Reporting: ClickHouse (columnar, optimized for analytics queries)
- User profiles: MongoDB (flexible schema for varying user metadata)

We explicitly rejected a shared database because it would defeat the purpose of service independence.
