# Q3 2025 Incident Post-Mortem

**Date**: September 8, 2025
**Duration**: 4 hours 12 minutes (09:15 UTC - 13:27 UTC)
**Severity**: P1 — Complete platform outage

## Summary

A database migration script ran against production instead of staging due to a misconfigured environment variable. This corrupted the users table, which cascaded to every service since they all shared a single PostgreSQL instance.

## Timeline

- 09:15 — Deploy pipeline triggers migration script
- 09:18 — First alerts fire: 500 errors on /api/login
- 09:22 — All API endpoints returning errors
- 09:30 — Incident declared, war room opened
- 10:45 — Root cause identified: migration ran ALTER TABLE on production users table
- 12:00 — Database restored from backup (3-hour-old snapshot)
- 13:27 — All services verified healthy, incident closed

## Root Cause

The CI/CD pipeline used `$DB_HOST` which defaulted to production when the staging override wasn't set. The migration script assumed the `users` table had a column `email_v2` that only existed in staging.

## Lessons Learned

1. **Shared database is a single point of failure** — every service went down because they all depended on one database. This directly led to the decomposition decision (see architecture log).
2. **Environment variables need validation** — we now require explicit environment confirmation before destructive operations.
3. **Backup frequency was insufficient** — 3-hour-old backup meant we lost 3 hours of user signups. Changed to continuous WAL archiving.

## Action Items

- [ ] Implement database-per-service (Owner: Platform team, Due: Q1 2026)
- [ ] Add environment confirmation gate to CI/CD (Owner: DevOps, Due: Oct 2025)
- [x] Switch to continuous backup with point-in-time recovery (Done: Sep 20, 2025)
