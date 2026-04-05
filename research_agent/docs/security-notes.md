# Security & Compliance Notes

## PCI DSS Compliance (Payment Service)

Since we process credit card payments, PCI DSS Level 2 applies. Key requirements:
- Cardholder data must be encrypted at rest (AES-256) and in transit (TLS 1.3)
- No raw card numbers stored — we tokenize via Stripe
- Quarterly vulnerability scans by approved scanning vendor (ASV)
- Annual penetration test (last done: November 2025, next: November 2026)
- Access to payment systems requires MFA + VPN + IP allowlist

Current status: compliant as of January 2026 audit. The service extraction is complicating things because we need to re-scope the PCI boundary for the new microservice.

## Zero Trust Network

Implemented in phases:
1. ✅ Service-to-service mTLS via Istio (done Q4 2025)
2. ✅ Network policies — deny all by default, explicit allow rules (done Q4 2025)  
3. 🟡 Identity-aware proxy for internal tools (in progress)
4. 🔴 Full BeyondCorp model for employee access (planned Q3 2026)

## Secret Incidents

### October 2025: AWS Key Leaked in Git
A junior developer committed an AWS access key to a public repo. Detected by GitHub secret scanning within 3 minutes. Key rotated within 15 minutes. No unauthorized access found in CloudTrail logs.

Response actions:
- Mandatory pre-commit hooks for secret detection (using gitleaks)
- All developers completed security awareness training
- Reduced IAM key lifetime to 12 hours (using STS temporary credentials)

### January 2026: Dependency Vulnerability
A critical CVE in a transitive dependency (log4j-style issue in a Python logging library). Detected by Dependabot. Patched within 24 hours. No exploitation detected.

We now run Snyk in CI/CD to block deploys with critical vulnerabilities.

## Authentication Architecture

- External users: OAuth 2.0 + OIDC via Auth0
- Internal services: mTLS certificates (managed by Vault)
- Admin panel: SSO via Okta + hardware security keys required
- API keys: scoped per-client, rate-limited, rotated every 90 days
