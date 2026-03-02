# Claude API Data Security & Compliance Guide

**Version**: 1.0
**Date**: 2026-03-01
**Audience**: Engineering, Legal, Sales, Customer Success

---

## Overview

This document covers Anthropic's data security policies, compliance certifications, and privacy protections across Claude plan types. Use this as a reference when evaluating Claude for regulated industries or when responding to customer security questionnaires.

All compliance documentation is available at Anthropic's Trust Portal: **https://trust.anthropic.com**

---

## Security Certifications

Anthropic maintains the following certifications across all commercial plans:

| Certification | Status | Notes |
|---|---|---|
| **SOC 2 Type I** | Completed | Independent audit of Claude infrastructure |
| **SOC 2 Type II** | Completed | SOC 3 summary publicly available; full report available under NDA for Enterprise customers |
| **ISO 27001:2022** | Certified | Information Security Management Systems |
| **ISO/IEC 42001:2023** | Certified | AI Management Systems standard |
| **HIPAA** | BAA Available | Business Associate Agreements available on API, Team, and Enterprise plans |

---

## Data Security Controls (All Plans)

### Encryption

| Layer | Standard |
|---|---|
| **In Transit** | TLS 1.2+ enforced for all network requests |
| **At Rest** | AES-256 encryption for stored logs, outputs, and files |

### Employee Access

- Anthropic employees **cannot access conversations by default**
- Access requires explicit user consent (e.g., submitted feedback)
- Exception: Trust & Safety policy enforcement reviews
- Mandatory annual security and privacy training for all staff
- Least privilege principle enforced across all internal systems

### International Data Transfers

- European Commission adequacy decisions for EEA/UK data transfers
- Standard Contractual Clauses (SCCs) with international partners
- GDPR compliant

---

## Plan-by-Plan Breakdown

### API Plan (Pay-as-You-Go)

**Data Retention:**
- Inputs and outputs deleted within **30 days**
- API logs deleted after **7 days**

**Training:**
- Data is **never used for model training** — prohibited by default

**Compliance Add-ons:**
- **Zero Data Retention (ZDR)**: Available — data processed in real-time, immediately discarded with no logging or storage
- **HIPAA BAA**: Available — requires sales engagement, custom pricing

**Best For:** Development, API integrations, compliance-critical workloads with ZDR

---

### Pro / Max Plan (Consumer)

**Data Retention:**
- Deleted conversations removed from backend within **30 days**
- Policy violation content retained up to **2 years**
- Trust & safety scores retained up to **7 years**

**Training:**
- As of September 28, 2025: training use is **opt-in only** (changed from opt-out)
- Incognito chats: **never used for training**, regardless of settings

**Compliance Add-ons:**
- ZDR: **Not available**
- HIPAA BAA: **Not available**

**Best For:** Individual users, personal productivity

---

### Team Plan

**Data Retention:**
- Inputs/outputs automatically deleted after **30 days**
- Saved conversations retained until user deletes them (deleted within 30 days of request)

**Training:**
- **Prohibited by default** under Commercial Terms of Service
- Only allowed if customer explicitly joins the Development Partner Program

**Admin Controls:**
- Admins can disable feedback collection to prevent team data from being submitted
- Basic organization management

**Compliance Add-ons:**
- HIPAA BAA: **Available** — requires sales engagement
- ZDR: **Not available**

**Best For:** Small teams, internal tooling, workplaces without heavy compliance requirements

---

### Enterprise Plan

**Data Retention:**
- **Configurable** — minimum 30-day floor, extendable per org needs
- Admins set retention preferences via Organization Settings > Data and Privacy
- Audit log retention: **180 days**, exportable

**Training:**
- **Strictly prohibited** — no opt-in options, no exceptions

**Security & Identity:**
- **SSO**: SAML 2.0 and OIDC-based authentication
- **SCIM**: Automated user provisioning and directory sync with IdP groups
- **Role-Based Access Control (RBAC)**: Admin-configurable access levels, group-based role mapping
- **Spending Controls**: Organization-level and per-user spending caps
- **Audit Logs**: 180-day retention, exportable by Organization Owners via Admin Settings > Data and Privacy

**Compliance Add-ons:**
- **ZDR**: Available — custom contract, pricing negotiated with sales
- **HIPAA BAA**: Available — custom contract, pricing negotiated with sales

**Best For:** Healthcare, finance, legal, government, and other regulated industries

---

## Plan Comparison Matrix

| Feature | API | Pro/Max | Team | Enterprise |
|---------|:---:|:-------:|:----:|:----------:|
| Training prohibited | ✓ | Opt-in | ✓ | ✓ |
| 30-day data deletion | ✓ | ✓ | ✓ | ✓ (configurable) |
| Zero Data Retention (ZDR) | ✓ | ✗ | ✗ | ✓ |
| HIPAA BAA | ✓ | ✗ | ✓ | ✓ |
| SOC 2 Certified | ✓ | ✓ | ✓ | ✓ |
| ISO 27001 | ✓ | ✓ | ✓ | ✓ |
| SSO / SCIM | ✗ | ✗ | Limited | ✓ Full |
| Audit Logs (180-day export) | ✗ | ✗ | ✗ | ✓ |
| Custom retention periods | ✗ | ✗ | ✗ | ✓ |
| Spending controls | ✗ | ✗ | ✗ | ✓ |

---

## Zero Data Retention (ZDR)

ZDR is an optional add-on for **API** and **Enterprise** plan customers with strict compliance or data isolation requirements.

### How It Works

When ZDR is active:
- Prompts and responses are processed in real-time
- **No logging, no storage** — data is immediately discarded after the response
- Exceptions apply only for legal compliance and misuse prevention

### Eligible Endpoints

| Endpoint | ZDR Eligible |
|---|---|
| Messages API (`/v1/messages`) | ✓ Yes |
| Token Counting API | ✓ Yes |
| Web Search / Web Fetch tools | ✓ Yes |
| Memory tool, Context Management, Fast Mode | ✓ Yes |
| Batch API | ✗ No |
| Code Execution tool | ✗ No |
| Files API | ✗ No |
| Server-side tool search | ✗ No |

### Pricing

ZDR pricing is **not publicly listed** — it is negotiated per customer through Anthropic's enterprise sales team.

---

## HIPAA Business Associate Agreement (BAA)

A BAA is required before processing Protected Health Information (PHI) with Claude. Anthropic offers BAAs for qualifying customers.

### Eligibility

| Plan | BAA Available |
|---|---|
| API | ✓ Yes |
| Pro / Max | ✗ No |
| Team | ✓ Yes |
| Enterprise | ✓ Yes |

### What the BAA Covers

- Use of Claude API for processing PHI
- Anthropic's obligations as a Business Associate under HIPAA
- Breach notification procedures
- Data handling and safeguard requirements

### Pricing

BAA pricing is **not publicly listed** — terms and costs are negotiated with Anthropic's sales team.

> **Note:** A BAA alone does not guarantee full HIPAA compliance. Your organization remains responsible for its own compliance controls, configurations, and workforce training.

---

## Use Case Recommendations

| Industry / Use Case | Recommended Plan | Key Requirements |
|---|---|---|
| Individual / Developer | Pro / Max or API | Standard retention, opt-in training |
| Internal team tooling | Team | Training prohibited, basic admin controls |
| API integration (standard) | API | 30-day retention, training prohibited |
| API integration (sensitive) | API + ZDR | Real-time processing, no logging |
| Healthcare (HIPAA) | Enterprise + BAA | BAA, ZDR recommended, audit logs |
| Finance / Legal | Enterprise | Audit logs, SSO, custom retention, ZDR |
| Government / Air-gap | Self-hosted / On-prem | See `docs/SPARK_DEPLOYMENT.md` |

---

## Engaging Anthropic Sales for Compliance Add-ons

Both HIPAA BAA and ZDR require direct engagement with Anthropic's enterprise sales team:

1. Visit **https://claude.ai/contact-sales**
2. Describe your deployment requirements and compliance needs
3. Anthropic will assess eligibility and provide custom terms and pricing

**Questions to ask sales:**
- Is HIPAA BAA bundled into Enterprise pricing or a separate line item?
- Is ZDR bundled or priced per API call / per month?
- Are there minimum contract terms for ZDR or BAA?
- Can BAA and ZDR be combined on the same contract?

---

## Key Policy Change: September 28, 2025

Consumer plan (Pro/Max) data training policy changed from **opt-out** to **opt-in**:

- Users must now explicitly enable data sharing for model training
- API, Team, and Enterprise customers are **unaffected** — training remains prohibited
- Incognito mode remains excluded from training regardless of settings

---

## Related Documentation

- `docs/ARCHITECTURE.md` — System architecture and data flow
- `docs/CLAUDE_API_VS_OLLAMA.md` — Hosted API vs. local inference trade-offs
- `docs/SPARK_DEPLOYMENT.md` — Air-gap / on-prem deployment (no data leaves your network)
- Anthropic Trust Portal: https://trust.anthropic.com
- Anthropic Privacy Center: https://privacy.claude.com
- ZDR Docs: https://platform.claude.com/docs/en/build-with-claude/zero-data-retention
