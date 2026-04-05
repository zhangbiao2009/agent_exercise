# Personal Learning Notes — Distributed Systems

## CAP Theorem (revisited)

You can only have 2 of 3: Consistency, Availability, Partition tolerance. Since network partitions are unavoidable, the real choice is between CP and AP.

- Our payment service is CP — we can't afford inconsistent balances. We accept brief unavailability during network issues.
- Our user profiles are AP — it's okay to show slightly stale profile info. Availability matters more.

## Event Sourcing Notes

Instead of storing current state, store every change as an event. Current state = replay all events.

Pros:
- Complete audit trail (critical for payments)
- Can rebuild any historical state
- Natural fit for event-driven architecture

Cons:
- Event store grows forever (need compaction strategy)
- Schema evolution is painful — what happens when event format changes?
- Debugging is harder: "why is this state wrong?" requires replaying events

We use this for the payment service. Every transaction is an event: PaymentInitiated, PaymentAuthorized, PaymentCaptured, PaymentRefunded.

## CQRS Pattern

Command Query Responsibility Segregation — separate the write model from the read model.

Why we use it with the reporting service:
- Writes go to PostgreSQL (normalized, consistent)
- Reads come from ClickHouse (denormalized, fast for analytics)
- An ETL pipeline syncs data every 5 minutes

The 5-minute delay is acceptable for reporting. It would NOT be acceptable for the payment service.

## Circuit Breaker Pattern

When service A calls service B and B is down, A should "open the circuit" and fail fast instead of waiting for timeouts. We use Envoy's built-in circuit breaker:
- Threshold: 5 consecutive failures
- Open duration: 30 seconds
- Half-open: allow 1 request through to test recovery

This saved us during the February 2026 incident when the auth service had a memory leak. The API gateway circuit-breaked instead of cascading the failure.

## Saga Pattern for Distributed Transactions

Since we have database-per-service, traditional ACID transactions don't work across services. Instead, we use the Saga pattern:

Example: Place Order saga
1. Order Service: Create order (pending)
2. Payment Service: Reserve funds
3. Inventory Service: Reserve items
4. If all succeed → confirm all
5. If any fails → compensating transactions (refund, unreserve)

We use choreography (event-based) rather than orchestration. Each service listens for events and reacts. Simpler to implement but harder to debug.
