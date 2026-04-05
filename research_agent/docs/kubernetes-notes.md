# Kubernetes & Deployment Notes

## Cluster Setup

We run two Kubernetes clusters on AWS EKS:
- **prod-us-east-1**: Production workloads, 12 nodes (m6i.xlarge), autoscaling 8-20
- **staging-us-east-1**: Staging + dev, 4 nodes (m6i.large), fixed size

All services deploy via ArgoCD with GitOps. The deployment manifest lives in the `infra-manifests` repo.

## Resource Limits Learned the Hard Way

After several OOMKilled incidents, we established these baseline resource configs:

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|------------|-----------|----------------|-------------|
| Auth | 250m | 500m | 256Mi | 512Mi |
| Payment | 500m | 1000m | 512Mi | 1Gi |
| API Gateway | 100m | 250m | 128Mi | 256Mi |

Rule of thumb: set requests to average usage, limits to 2x requests.

## Secrets Management

Switched from Kubernetes Secrets (base64 encoded, not encrypted) to HashiCorp Vault in November 2025. Each service gets a unique Vault role with access only to its own secrets.

Rotation policy:
- Database credentials: rotated every 30 days automatically
- API keys: rotated every 90 days
- TLS certificates: managed by cert-manager, auto-renewed 30 days before expiry

## Monitoring Stack

- **Prometheus**: Metrics collection (scrape interval: 15s)
- **Grafana**: Dashboards and alerting
- **Loki**: Log aggregation (replaced ELK stack in Q4 2025 — 60% cost reduction)
- **Jaeger**: Distributed tracing for inter-service calls

Alert routing: PagerDuty for P1/P2, Slack #alerts for P3/P4.

## Lessons from the Kafka Migration

When we moved Kafka to Kubernetes (Strimzi operator), we hit several issues:
1. Broker rebalancing caused 30-second message delays during pod restarts
2. Persistent volume claims needed manual cleanup after broker decommission
3. ZooKeeper was a constant headache — switched to KRaft mode in December 2025

Current Kafka config: 5 brokers, replication factor 3, 200 partitions total across all topics.
