# Marketing Requirements Document — AI Ready RAG

| Field | Value |
|-------|-------|
| **Status** | DRAFT — For Review |
| **Version** | v1.0 |
| **Authors** | Jason, Paul |
| **Date** | 2026-02-11 |
| **Review Date** | 2026-02-13 (Thursday meeting with Paul & Ryan) |
| **Confidentiality** | Vital Enterprises — Internal |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2026-02-11 | Jason | Initial draft — assembled from transcripts, codebase, and market research |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision & Strategy](#2-product-vision--strategy)
3. [Market Opportunity](#3-market-opportunity)
4. [Target Customer](#4-target-customer)
5. [Customer Personas](#5-customer-personas)
6. [Problem Statement](#6-problem-statement)
7. [Product Requirements — MVP](#7-product-requirements--mvp)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Quality Standards & Benchmarks](#9-quality-standards--benchmarks)
10. [Competitive Landscape](#10-competitive-landscape)
11. [Pricing & Packaging](#11-pricing--packaging)
12. [Go-to-Market Strategy](#12-go-to-market-strategy)
13. [Timeline & Milestones](#13-timeline--milestones)
14. [Success Metrics](#14-success-metrics)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [Open Items & Decisions Required](#16-open-items--decisions-required)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

AI Ready RAG is an **air-gapped enterprise knowledge assistant** that enables organizations to search, query, and get cited answers from their internal documents — without any data leaving their infrastructure.

The product runs entirely on-premises on the **NVIDIA DGX Spark** ($3,999 hardware appliance) or as a **dedicated hosted instance** managed by Vital Enterprises. All components — document processing, vector search, LLM inference, and the web application — run locally with zero internet dependency at runtime.

**Core value proposition:** Deploy an air-gapped knowledge assistant with predictable SMB pricing and hands-on support.

**Key differentiators:**
1. **Air-gapped security** — No data leaves the customer's network. Full runtime independence from the internet.
2. **Lower total cost** — Predictable $850/month pricing vs. enterprise platforms starting at $2,000–$8,600+/month.
3. **Customer service** — Hands-on rollout, operator onboarding, and ongoing support (not self-service DIY).

**Business targets:**
- **March 16, 2026** — Internal testing with 2–3 Vital Enterprises organizations
- **April 20, 2026** — Demo customers enabled
- **May 25, 2026** — Product launch (GA)
- **September 16, 2026** — 6 paid pilots closed

---

## 2. Product Vision & Strategy

### Vision

Every mid-market and regulated enterprise has critical operational knowledge trapped in documents that employees can't find quickly. AI Ready RAG makes that knowledge instantly accessible through natural-language questions with cited answers — all running inside the customer's own walls.

### Strategic Position

AI Ready RAG occupies a unique position at the intersection of:

1. **Air-gapped deployment** — True offline runtime, not "private cloud" with internet callbacks
2. **SMB affordability** — Under $1,000/month, competing against $2,000–$8,600+/month enterprise platforms
3. **Turnkey experience** — Pre-configured appliance, not a framework requiring engineering assembly

### Product Branding

> **[PLACEHOLDER — Paul/Jason]**: Final product name decision needed. Current working name is "AI Ready RAG." Options discussed in meeting include variations around "Enterprise Knowledge Assistant" and appliance-style branding.

### Dual Deployment Model

| Model | Description | Target Customer |
|-------|-------------|-----------------|
| **On-Premise (DGX Spark)** | Customer purchases NVIDIA DGX Spark ($3,999). AI Ready RAG ships pre-installed or is deployed on-site. All data stays on customer hardware. Zero network dependency. | Organizations with strict physical air-gap requirements, regulated industries |
| **Hosted (Dedicated Instance)** | Vital Enterprises hosts a dedicated, single-tenant instance per customer. Customer data is fully isolated — not exposed to the public internet and never commingled with other customers. The only network path is a private, encrypted connection between the customer and their dedicated instance. | Organizations wanting managed infrastructure without sacrificing data isolation |

Both deployment models deliver the same data isolation guarantee: **customer data is never exposed to the public internet and never shared across tenants.** The on-premise model places the hardware inside the customer's own walls. The hosted model places it on Vital Enterprises-managed infrastructure with the same single-tenant isolation — the difference is who manages the hardware, not how the data is protected.

---

## 3. Market Opportunity

### Market Context

Enterprise organizations generate vast amounts of internal documentation — SOPs, policies, training manuals, compliance guides, process documentation — that employees struggle to search and reference efficiently. Current solutions are either:
- **Cloud-based AI platforms** that require sending data to third-party infrastructure (unacceptable for security-sensitive organizations)
- **Open-source RAG frameworks** that require significant engineering expertise to deploy, maintain, and optimize
- **Legacy enterprise search** platforms priced for large enterprises ($25,000–$103,700+/year)

### Market Sizing

> **[PLACEHOLDER — Paul/Jason]**: Quantified TAM/SAM/SOM needed.
>
> **Suggested approach:**
> - **TAM**: US mid-market enterprises (100–1,000 employees) + regulated industries with document-heavy workflows
> - **SAM**: Subset with security/compliance constraints that prevent cloud AI adoption, West Coast initially
> - **SOM**: Achievable market — 6 paid pilots in first 6 months, expanding to [X] customers by end of Year 1
>
> **Data points available:**
> - There are approximately 200,000 mid-market businesses in the US (100–999 employees)
> - Priority verticals: Legal, Manufacturing, Financial services, Agriculture, Landscaping
> - Geographic focus: West Coast first (Portland, OR hub)
> - ICP filter: 20–300 employees, document-heavy, security-sensitive

### Market Timing

The window for establishing early-mover positioning is **March 16 – September 16, 2026**:
- NVIDIA DGX Spark brings local GPU inference to an affordable price point ($3,999)
- Open-source LLMs (Qwen, LLaMA, DeepSeek) are now production-quality for 8B–32B parameter range
- Enterprise AI adoption is accelerating, but air-gapped solutions remain scarce
- Competitors are primarily cloud-first or require engineering teams to assemble

---

## 4. Target Customer

### Ideal Customer Profile (ICP)

| Attribute | Specification |
|-----------|---------------|
| **Company size** | 20–300 employees |
| **Industry verticals** | Legal, Manufacturing, Financial services, Agriculture, Landscaping |
| **Geography** | West Coast first (Portland, OR hub), then US expansion |
| **Workflow type** | Document-heavy operations (SOPs, policies, training, compliance) |
| **Security posture** | Cloud hesitation or air-gap requirement from policy or risk posture |
| **Buyer profile** | Operations leader (not CTO or ML team) |
| **Budget authority** | Can approve $850/month without enterprise procurement cycle |
| **Engineering capacity** | Limited — cannot maintain DIY RAG stack |

### Target Segments

| Segment | Size | Characteristics |
|---------|------|-----------------|
| **Mid-market enterprises** | 100–1,000 employees | Multiple departments, established processes, document accumulation |
| **Regulated industries** | Any size | Compliance requirements, audit trails, data sovereignty mandates |

### Qualification Questions (Qualify Fast)

1. Is on-prem or private VPC a hard requirement?
2. Is internet-independent runtime required by policy or risk posture?
3. Does your team have bandwidth to own a DIY RAG stack?
4. Is predictable monthly cost required for approval?
5. Is reducing search/onboarding time an active objective this quarter?
6. Who owns budget and technical sign-off?

### Red Flags (Disqualify Fast)

1. "We only buy from hyperscalers" with no exception path
2. "We already have a funded internal RAG engineering squad"
3. "No measurable business problem to solve in next 90 days"
4. "Security constraints are optional and cloud-only is preferred"

---

## 5. Customer Personas

### Primary Persona: Operations Leader

| Attribute | Description |
|-----------|-------------|
| **Title** | VP Operations, Director of Operations, Operations Manager |
| **Reports to** | CEO or COO |
| **Team size** | 10–100 direct/indirect reports |
| **Key responsibilities** | Process consistency, onboarding, compliance, workforce productivity |
| **Pain points** | Time lost searching for docs, slow onboarding, inconsistent process execution, knowledge loss from turnover |
| **Success metrics** | Reduced search time, faster onboarding, fewer process errors |
| **Technology comfort** | Uses business apps daily, not technical enough to deploy/maintain RAG stack |
| **Budget** | Controls departmental OpEx, can approve $850/month |

### Secondary Persona: IT/Security Gatekeeper

| Attribute | Description |
|-----------|-------------|
| **Title** | IT Director, Security Officer, IT Manager |
| **Role in sale** | Evaluator / Blocker |
| **Concerns** | Data sovereignty, network security, compliance, integration with AD/SSO |
| **Success metrics** | No data egress, audit trail, SSO integration, minimal operational burden |
| **What they need** | Security architecture brief, deployment checklist, firewall rules, AD integration plan |

### Tertiary Persona: End User (Knowledge Worker)

| Attribute | Description |
|-----------|-------------|
| **Title** | Analyst, Coordinator, Specialist, Technician |
| **Interaction** | Daily user of the knowledge assistant |
| **Needs** | Fast answers with citations, accessible from existing workflow (web app or embedded widget) |
| **Success metrics** | Answers found in seconds instead of minutes, trusted citations |

> **[PLACEHOLDER — Paul/Jason]**: Customer interview quotes and validated persona details. Current personas are synthesized from market research and meeting discussions. Need real customer feedback from internal testing (March 16+) to validate.

---

## 6. Problem Statement

### Customer Problems

| Problem | Impact | Current Workaround |
|---------|--------|--------------------|
| **Document search is slow** | Employees spend 20–30% of time searching for information across scattered systems | Manual searching through file shares, SharePoint, email archives |
| **Onboarding takes too long** | New hires take weeks to become productive; institutional knowledge lives in people's heads | Shadow documentation, tribal knowledge, senior staff as bottleneck |
| **Process execution is inconsistent** | Different employees follow different versions of procedures | Outdated printed SOPs, informal knowledge transfer |
| **Knowledge lost on turnover** | When experienced staff leave, their knowledge leaves too | Exit interviews (rarely captures operational detail) |
| **Cloud AI is not acceptable** | Security/compliance prevents sending internal documents to cloud services | No AI-assisted search; manual processes continue |

### Why Now

1. **Hardware cost inflection** — DGX Spark at $3,999 makes local GPU inference economically viable for SMB
2. **Model quality inflection** — 8B parameter open-source models now rival cloud-only models from 12 months ago
3. **Regulatory pressure** — Increasing data sovereignty requirements in legal, financial, and government sectors
4. **Competitive pressure** — Cloud-native competitors are adding "private deployment" messaging, but true air-gap remains rare

---

## 7. Product Requirements — MVP

### 7.1 Core Features

| Feature | Priority | Description | Status |
|---------|----------|-------------|--------|
| **Document Ingestion** | P0 | Upload, process, chunk, and embed documents | Implemented |
| **Natural Language Query** | P0 | Ask questions in plain English, get cited answers | Implemented |
| **Access Control** | P0 | Pre-retrieval tag-based filtering; users only see authorized documents | Implemented |
| **Chat Sessions** | P0 | Multi-turn conversations with history | Implemented |
| **User Management** | P0 | Admin creates/manages users, assigns document access tags | Implemented |
| **AD/SSO Integration** | P0 | Hybrid auth — SSO (Entra ID/OIDC) when network available, local JWT fallback | **Planned** |
| **Embeddable Chat Widget** | P0 | `<script>` tag or iframe for embedding in customer intranets | **Planned** |
| **Usage Reports** | P0 | Admin-visible usage analytics (queries, users, sessions, documents) | **Planned** |
| **Backup / Restore** | P0 | Full system backup and restore capability (SQLite, Qdrant, uploads) | **Planned** |
| **Confidence Scoring** | P1 | CITE (answer with sources) vs. ROUTE (escalate to human) | Partially implemented |
| **RAG Quality Evaluation** | P1 | RAGAS-based batch and live evaluation | **In Development** — Spec complete (v1.3), implementation in progress |
| **Audit Logging** | P1 | 3-level configurable audit trail (essential, comprehensive, full_debug) | Implemented |

### 7.2 Supported Document Formats

| Format | Priority | Processing |
|--------|----------|------------|
| **PDF** | P0 | Docling with OCR, table extraction, semantic chunking. Tiered ingestion planned. |
| **DOCX** | P0 | Docling processing |
| **XLSX** | P0 | Tiered ingestion (Type A: tabular → structured DB, Type B: document-formatted → text serialization, Type C: hybrid → split and route). See ingestkit-excel SPEC v2.0. |
| **PPTX** | P0 | Docling processing |
| **TXT** | P0 | Direct chunking |
| **MD** | P0 | Markdown processing |
| **HTML** | P1 | HTML processing |
| **CSV** | P1 | Structured data processing |

### 7.3 Tiered Document Ingestion (XLSX)

The XLSX ingestion pipeline uses a three-tier classification system:

| Tier | Trigger | Volume | Method |
|------|---------|--------|--------|
| **Tier 1** | Rule-based heuristics (header detection, numeric density, blank-row analysis) | ~85% of files | Fast, deterministic classification |
| **Tier 2** | Lightweight LLM classification (sheet structure → LLM prompt) | ~12% of files | For ambiguous cases where rules are insufficient |
| **Tier 3** | Reasoning model (full sheet content → deep analysis) | ~3% of files | For complex hybrid documents requiring judgment |

**Classification outcomes:**

| Type | Description | Processing Path |
|------|-------------|-----------------|
| **Type A** (Tabular) | Clean relational data (databases, inventories, financial tables) | → Structured DB ingestion, SQL-queryable |
| **Type B** (Document-formatted) | Forms, reports, narratives using spreadsheet as layout | → Text serialization, RAG-ready chunks |
| **Type C** (Hybrid) | Mixed — some sheets tabular, some document-like | → Split by sheet, route each independently |

PDF tiered ingestion will follow a similar pattern (details pending specification).

### 7.4 User Interface

| Component | Description | Priority |
|-----------|-------------|----------|
| **Standalone React Web App** | Full-featured web application served by FastAPI | P0 — Implemented |
| **Embeddable Chat Widget** | Lightweight `<script>` tag or `<iframe>` for customer intranets and portals | P0 — Planned |
| **Admin Dashboard** | User management, document management, usage reports, system health | P0 — Partially implemented |

Both UI components ship at launch.

### 7.5 Authentication & Authorization

| Capability | Description | Priority |
|------------|-------------|----------|
| **Local JWT Auth** | Email/password login with HS256 JWT tokens | Implemented |
| **AD SSO (Entra ID)** | OIDC-based SSO via Microsoft Entra ID | P0 — Planned |
| **Hybrid Auth Mode** | SSO when network allows, local JWT when air-gapped | P0 — Planned |
| **Role-Based Access** | Admin and User roles | Implemented |
| **Tag-Based Document Access** | Users assigned document tags; retrieval filtered pre-search | Implemented |

**AD SSO Effort Estimate:** 6–8 engineering days, MODERATE difficulty. Requires:
- OIDC/OAuth2 client configuration
- Dual-mode auth middleware (local + SSO)
- Group-to-tag mapping
- Long-lived local token for air-gap fallback
- Admin UI for SSO configuration

### 7.6 Multi-Language Support

| Language | Status | Priority |
|----------|--------|----------|
| **English** | Supported | P0 — MVP minimum |
| **Spanish** | Under evaluation | Decision pending — need to evaluate model capability and effort |
| **Other languages** | Future | Post-MVP based on customer demand |

> **[PLACEHOLDER — Paul/Jason]**: Decision needed on Spanish language support timeline. Need to evaluate: (1) Ollama model multi-language capability with current models (qwen3:8b supports multilingual), (2) embedding model language support (nomic-embed-text is primarily English), (3) OCR language packs, (4) effort estimate.

### 7.7 Usage Reports

| Report | Description | Priority |
|--------|-------------|----------|
| **Query Volume** | Queries per day/week/month, by user | P0 |
| **Active Users** | DAU/WAU/MAU, session duration | P0 |
| **Document Usage** | Most-queried documents, unused documents | P0 |
| **Response Quality** | RAGAS scores over time (when evaluation is enabled) | P1 |
| **System Health** | Component status, storage utilization, model performance | P0 |

### 7.8 Backup & Restore

| Component | Backup Method | Restore Method |
|-----------|---------------|----------------|
| **SQLite Database** | File copy (with WAL checkpoint) | File restore + restart |
| **Qdrant Vectors** | Qdrant snapshot API | Snapshot restore via API |
| **Uploaded Documents** | Directory tar/archive | Extract to uploads directory |
| **Configuration** | `.env` file backup | File restore |

Admin-initiated backup/restore via API and UI. Automated scheduled backups as a P1 follow-on.

---

## 8. Deployment Architecture

### 8.1 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI 0.115.x (Python 3.12+) | REST API, auth middleware, access control |
| **Frontend** | React (Vite + TypeScript) | Web application and admin dashboard |
| **Database** | SQLite (WAL mode) | Users, sessions, chat history, tags, audit logs |
| **Vector Store** | Qdrant 1.13.x | Document embeddings with tag-based filtering |
| **LLM Inference** | Ollama + qwen3:8b | Local chat model |
| **Embeddings** | Ollama + nomic-embed-text | Document and query embeddings |
| **Document Processing** | Docling 2.68.0 + Tesseract OCR | PDF/DOCX parsing, OCR, table extraction |
| **Reverse Proxy** | NGINX | HTTPS termination, security headers |
| **Containerization** | Docker | Qdrant and Ollama containers |

### 8.2 On-Premise Architecture (DGX Spark)

```
┌─────────────────────────────────────────────────────────┐
│ NVIDIA DGX Spark ($3,999)                               │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────┐        │
│  │ NGINX (443)     │───→│ FastAPI + React       │        │
│  │ HTTPS / Headers │    │ (:8000)               │        │
│  └─────────────────┘    │  ├─ JWT Auth + SSO    │        │
│                         │  ├─ Access Control     │        │
│                         │  ├─ Audit Logging      │        │
│                         │  └─ RAG Pipeline       │        │
│                         └──────────┬─────────────┘        │
│                                    │                      │
│  ┌──────────────┐   ┌─────────────┴───────────────┐      │
│  │ SQLite (WAL) │   │                             │      │
│  │ Users, Chat  │   │                             │      │
│  │ Tags, Audit  │   ▼                 ▼           │      │
│  └──────────────┘  ┌──────────┐  ┌──────────┐    │      │
│                    │ Qdrant   │  │ Ollama   │    │      │
│                    │ (:6333)  │  │ (:11434) │    │      │
│                    │ Vectors  │  │ GPU LLM  │    │      │
│                    └──────────┘  └──────────┘    │      │
│                                                         │
│  🔒 No internet required at runtime                     │
│  🔒 Firewall: only port 443 exposed                    │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Hosted Architecture (Dedicated Instance)

```
┌──────────────────────────────────────────────────┐
│ Vital Enterprises Infrastructure                  │
│                                                   │
│  Customer A Instance    Customer B Instance        │
│  ┌──────────────────┐  ┌──────────────────┐       │
│  │ Full Stack        │  │ Full Stack        │      │
│  │ (isolated)        │  │ (isolated)        │      │
│  │ - FastAPI+React   │  │ - FastAPI+React   │      │
│  │ - SQLite          │  │ - SQLite          │      │
│  │ - Qdrant          │  │ - Qdrant          │      │
│  │ - Ollama          │  │ - Ollama          │      │
│  └──────────────────┘  └──────────────────┘       │
│                                                   │
│  NOT multi-tenant. Each customer gets a           │
│  dedicated, isolated deployment.                  │
└──────────────────────────────────────────────────┘
```

> **[PLACEHOLDER — Paul/Jason]**: Hosted infrastructure details needed — cloud provider, VM sizing, cost model per customer instance, SLA targets.

### 8.4 Hardware Requirements (On-Premise)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA DGX Spark | DGX Spark with A100 |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 100GB SSD | 500GB NVMe |
| **Network** | 1Gbps (for initial model download) | 10Gbps |

Runtime operates fully offline after initial setup.

---

## 9. Quality Standards & Benchmarks

### 9.1 RAG Quality Metrics (RAGAS Framework)

AI Ready RAG uses the **RAGAS** evaluation framework — the industry standard for measuring RAG quality. Four metrics are tracked:

| Metric | What It Measures | Requires Ground Truth |
|--------|------------------|-----------------------|
| **Faithfulness** | Does the answer accurately represent the retrieved context without hallucination? | No |
| **Answer Relevancy** | How relevant is the answer to the user's question? | No |
| **Context Precision** | Are the most relevant documents retrieved first? | Yes |
| **Context Recall** | Does the retrieved context contain enough information to answer? | Yes |

### 9.2 Tiered Quality Targets

Quality ramps over 90 days from demo customers (April 20 → July 19, 2026):

| Metric | Minimum Viable (Launch) | Production-Ready (90 Days) |
|--------|------------------------|---------------------------|
| **Faithfulness** | > 0.70 | > 0.85 |
| **Answer Relevancy** | > 0.60 | > 0.80 |
| **Context Precision** | > 0.50 | > 0.70 |
| **Context Recall** | > 0.70 | > 0.80 |

### 9.3 Industry Benchmark Context

These targets are informed by published industry benchmarks:

| Source | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------|-------------|-----------------|-------------------|----------------|
| **RAGAS Leaderboard (2025)** | 0.75–0.95 | 0.70–0.90 | 0.60–0.85 | 0.70–0.90 |
| **TruLens Production Baselines** | >0.80 (recommended) | >0.75 (recommended) | — | — |
| **DeepEval Thresholds** | >0.70 (minimum) | >0.70 (minimum) | >0.60 (minimum) | >0.70 (minimum) |
| **Enterprise RAG Surveys (2025)** | 0.80 median | 0.75 median | 0.65 median | 0.75 median |

**Our "Minimum Viable" targets meet or exceed industry minimums.** The "Production-Ready" targets align with enterprise medians, positioning AI Ready RAG as a quality-competitive solution.

### 9.4 Quality Assurance Approach

1. **Batch Evaluation** — Admin triggers evaluation runs against curated Q&A datasets. Per-sample scores stored with config snapshots for reproducibility.
2. **Live Monitoring** — Configurable sampling rate on production traffic. Faithfulness + Answer Relevancy scored without ground truth.
3. **Confidence Scoring** — Each response classified as CITE (sufficient confidence to answer with citations) or ROUTE (insufficient confidence, escalate to human).
4. **Hallucination Checking** — Enabled by default in Spark production profile.

---

## 10. Competitive Landscape

### 10.1 Competitive Summary

15 competitors analyzed across direct platforms, self-hostable infrastructure, open-source stacks, and adjacent cloud threats. Full analysis in `docs/competitor-analysis-2026-02-10.md`.

**Competitive positioning:** AI Ready RAG competes at the intersection of air-gapped security, SMB affordability (<$1k/month), and turnkey experience — a space where no current competitor fully operates.

### 10.2 Competitive Quadrant

```
                    High Air-Gap Capability
                           │
     AI Ready RAG ★        │    Sinequa, Mindbreeze
     (affordable +         │    (enterprise price)
      air-gapped)          │
                           │
 SMB Affordable ───────────┼─────────── Enterprise Priced
                           │
     OSS Stacks            │    AWS Kendra, Azure AI Search
     (DIY assembly         │    Elastic, IBM watsonx
      required)            │    SearchBlox
                           │
                    Low Air-Gap Capability
```

### 10.3 Top 5 Competitive Threats

Ranked by scored battle matrix (weighted: SMB Affordability 40%, Deployment Fit 30%, Air-Gap 20%, Ops Simplicity 10%):

| Rank | Competitor | Score | Threat Level | Our Counter |
|------|-----------|-------|-------------|-------------|
| 1 | **Qdrant + DIY Stack** | 4.8 | High | "You get vector search. You still need ingestion, access control, auth, UI, evaluation, and support. That's months of engineering." |
| 2 | **OpenSearch + DIY Stack** | 4.7 | High | "Mature search, but building RAG on it requires embedding pipelines, LLM integration, access control, and ongoing maintenance." |
| 3 | **Haystack (deepset OSS)** | 4.7 | Medium-High | "Excellent framework — for teams with Python engineers to build and maintain it. We deliver the same outcome without the build." |
| 4 | **LangChain + DIY Stack** | 4.7 | Medium-High | "LangChain connects pieces. You still need to choose, deploy, secure, and maintain every piece." |
| 5 | **LlamaIndex + DIY Stack** | 4.7 | Medium-High | "Great for prototypes. Production requires access control, audit, evaluation, backup, and ops support that frameworks don't include." |

**Core battlecard message:** "You can build many RAG stacks. The question is how fast you can deploy, govern, and operate it."

### 10.4 Competitive Pricing Comparison

| Competitor | Monthly Cost | Air-Gap | Turnkey | Notes |
|------------|-------------|---------|---------|-------|
| **AI Ready RAG** | **$850/mo** | **Yes** | **Yes** | Includes setup + support |
| OpenSearch (OSS) | $0 + infra + eng | Yes | No | Requires significant engineering |
| Qdrant (OSS) | $0 + infra + eng | Yes | No | Vector-only, needs full stack |
| Weaviate | $45–$400+/mo | Self-hosted only | Partial | Infrastructure only |
| LlamaIndex Cloud | $50–$500+/mo | No (cloud) | Partial | Cloud-only managed |
| SearchBlox | $2,083+/mo | Yes | Yes | 2.5x our price |
| Mindbreeze InSpire | $8,641+/mo | Yes | Yes | 10x our price |
| AWS Kendra | $230+/mo + usage | No | Yes | Cloud-only, scales unpredictably |

Full pricing research in `docs/competitor-pricing-snapshot-2026-02-10.md`.

### 10.5 Where We Win / Where We Don't

**We win when:**
- Strict offline/security requirements (air-gap is non-negotiable)
- SMB budget pressure (predictable $850/month)
- Operations-led buyers without ML platform teams
- Document-heavy operational workflows
- High-touch support expectations

**We don't win when (or need strong proof):**
- Cloud-first standardization without air-gap requirement
- Brand-first enterprise procurement (incumbents preferred)
- Feature-max buyers seeking ecosystem lock-in
- Highly custom internal AI teams with engineering capacity

Full analysis in `docs/where-we-win-where-we-dont-2026-02-10.md`.

---

## 11. Pricing & Packaging

### 11.1 Pilot Pricing Model

| Component | Price |
|-----------|-------|
| **Monthly subscription** | $850/month |
| **Pilot duration** | 6 months |
| **Pilot total** | $5,100 |
| **Minimum margin target** | 35% |

Pilot package includes:
- Setup and integration support
- Initial knowledge base ingestion
- Operator onboarding
- Monthly optimization review

### 11.2 Hardware (On-Premise Only)

| Item | Cost | Paid By |
|------|------|---------|
| NVIDIA DGX Spark | $3,999 | Customer (one-time purchase) |

> **[PLACEHOLDER — Paul/Jason]**: Hardware bundling decision needed. Options discussed:
> 1. Customer purchases Spark directly ($3,999) + subscription ($850/mo)
> 2. Bundled pricing ($10,000 upfront or ~$1,000/mo for 12 months)
> 3. Lease-to-own model
>
> The meeting transcript discussed "$4K box plus $10K" and "$1K/month" models but no final decision was made.

### 11.3 Post-Pilot Pricing

> **[PLACEHOLDER — Paul/Jason]**: Post-pilot conversion pricing to be defined later. Include examples:
> - **General subscription**: Annual contract with monthly billing (e.g., $750/mo annual vs. $850/mo monthly)
> - **Tiered support**: Basic (email, 48hr SLA), Standard (email + phone, 24hr SLA), Premium (dedicated support, 4hr SLA)
> - **Volume discounts**: Multi-department or multi-site deployments
> - **Usage-based add-ons**: Additional storage, additional users, priority processing

### 11.4 Industry Pricing Models (Reference)

For context, common pricing models in the enterprise knowledge/RAG space:

| Model | Examples | Pros | Cons |
|-------|----------|------|------|
| **Flat monthly/annual** | SearchBlox, Mindbreeze | Predictable for buyer | Hard to capture upsell |
| **Per-seat** | LangSmith ($39/seat/mo) | Scales with adoption | Complex for SMB |
| **Usage-based** | AWS Kendra ($0.32/hr base) | Low entry | Unpredictable costs |
| **Tiered** | LlamaIndex ($0/$50/$500/mo) | Multiple entry points | Feature gating complexity |
| **Hybrid** | Our approach ($850/mo flat) | Simple + predictable | May leave money on table at scale |

---

## 12. Go-to-Market Strategy

### 12.1 GTM Summary

| Element | Detail |
|---------|--------|
| **Launch market** | West Coast, Portland OR hub |
| **Primary channel** | Direct outbound (email + LinkedIn + warm intros) |
| **Secondary channels** | Local networking/events, referral partners (MSPs/IT consultants) |
| **Primary buyer** | Operations leader |
| **Sales motion** | Mixed (direct + partner) |
| **Budget** | <$10,000 total through September 16, 2026 |
| **Max CAC** | $1,500 per paid pilot |

### 12.2 Messaging Framework

**Primary message:** "Deploy an air-gapped knowledge assistant with predictable SMB pricing and hands-on support."

**Three value pillars:**
1. **Security** — Offline/on-prem or private VPC deployment
2. **Cost** — Predictable monthly pricing for SMB reality
3. **Service** — Hands-on rollout and operator enablement

**Pain-to-value statements:**
- Reduce time spent searching internal docs
- Improve onboarding speed for new staff
- Standardize execution of repeat processes
- Reduce knowledge loss from turnover

### 12.3 Channel Mix (Budget-Constrained)

| Channel | Budget | Activities |
|---------|--------|------------|
| **Outbound tooling + data** | $2,000 | Email tools, contact data |
| **Events/networking** | $3,000 | Portland local events, meetups |
| **Partner enablement** | $1,500 | MSP/IT consultant collateral |
| **Demo assets + tech prep** | $2,000 | Demo environments, one-pagers |
| **Contingency** | $1,500 | |
| **Total** | **<$10,000** | |

### 12.4 Sales Assets Required

1. One-page pilot brief
2. Security/deployment architecture brief (air-gapped)
3. ROI worksheet for operations leaders
4. Competitive comparison one-pager (top 5 alternatives)
5. Objection handling script

Full outbound messaging pack in `docs/outbound-messaging-pack-2026-02-10.md`.
Full GTM action plan in `docs/pilot-gtm-action-plan-2026-02-10.md`.

### 12.5 Priority Verticals

| Priority | Vertical | Why |
|----------|----------|-----|
| 1 | **Legal** | Document-heavy, strict confidentiality, high search time |
| 2 | **Manufacturing** | SOPs, safety procedures, compliance documentation |
| 3 | **Financial services** | Regulatory compliance, audit requirements, data sovereignty |
| 4 | **Agriculture** | Process documentation, safety protocols, seasonal onboarding |
| 5 | **Landscaping** | Operational SOPs, training consistency, turnover management |

---

## 13. Timeline & Milestones

### 13.1 Launch Timeline

```
2026
────────────────────────────────────────────────────────────────
Feb 11          Mar 16              Apr 20          May 25
  │               │                   │               │
  ▼               ▼                   ▼               ▼
 MRD          INTERNAL            DEMO            PRODUCT
 Draft        TESTING             CUSTOMERS       LAUNCH (GA)
              2-3 VE orgs         External         General
                                  pilots           availability
                                    │
                                    │  90-day quality ramp
                                    ├──────────────────────────►
                                    │                      Jul 19
                                    │               Production-Ready
                                    │               quality targets met
                                    │
                                    └──────────────────────────────────►
                                                                Sep 16
                                                        6 paid pilots
                                                         $5,100 MRR
```

### 13.2 Milestone Details

| Date | Milestone | Definition of Done |
|------|-----------|-------------------|
| **Feb 13** | MRD reviewed with Paul & Ryan | Decisions captured, open items assigned |
| **Mar 16** | Internal testing begins | 2–3 Vital Enterprises organizations using the system |
| **Apr 20** | Demo customers enabled | External customers can access hosted or on-prem deployments |
| **May 25** | Product launch (GA) | Public-facing product, sales motion active |
| **Jul 19** | Production-ready quality | RAGAS metrics meet Production-Ready thresholds |
| **Sep 16** | 6 paid pilots closed | $5,100 MRR, CAC ≤ $1,500 per pilot |

### 13.3 MVP Feature Readiness

| Feature | Target Date | Status |
|---------|-------------|--------|
| Core RAG (ingest, query, cite) | Ready | Implemented |
| Access control + audit | Ready | Implemented |
| React web app | Ready | Implemented |
| AD/SSO integration | Mar 16 (internal) | Planned — 6-8 days |
| Embeddable chat widget | Apr 20 (demo) | Planned |
| Usage reports | Apr 20 (demo) | Planned |
| Backup/restore | Apr 20 (demo) | Planned |
| RAGAS evaluation | Apr 20 (demo) | Spec complete |
| Tiered XLSX ingestion | Apr 20 (demo) | Spec complete |

> **[PLACEHOLDER — Paul/Jason]**: Engineering resource allocation and sprint plan needed. How many engineers? Full-time or part-time? Who owns each feature area?

---

## 14. Success Metrics

### 14.1 Business Metrics

| Metric | Target | Timeline |
|--------|--------|----------|
| **Paid pilots** | 6 | By September 16, 2026 |
| **MRR at cohort** | $5,100 | 6 × $850/month |
| **Top-of-funnel** | 24 qualified opportunities | Mar 16 – Sep 16 |
| **Proposals sent** | 12 | Mar 16 – Sep 16 |
| **Close rate** | 50% (proposal → pilot) | Ongoing |
| **CAC per pilot** | ≤ $1,500 | Ongoing |
| **Total spend** | <$10,000 | Through Sep 16 |

### 14.2 Product Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Faithfulness (RAGAS)** | >0.70 → >0.85 | Batch evaluation + live monitoring |
| **Answer Relevancy** | >0.60 → >0.80 | Batch evaluation + live monitoring |
| **Context Precision** | >0.50 → >0.70 | Batch evaluation |
| **Context Recall** | >0.70 → >0.80 | Batch evaluation |
| **System uptime** | >99.5% | Health check monitoring |
| **Response time (P95)** | <10 seconds | Application metrics |

### 14.3 Pilot Success Metrics (Per Customer)

What to prove in every pilot:
1. **Time-to-first-usable-answer** — How quickly does the system deliver value after deployment?
2. **Reduction in document search time** — Baseline vs. post-deployment
3. **Onboarding acceleration** — New hire time-to-productivity
4. **Access-control correctness** — Under real user roles and tags
5. **Operator adoption and support load** — Daily active usage and support ticket volume

### 14.4 Weekly Leading Indicators

| Indicator | Target |
|-----------|--------|
| New outbound contacts | 30+ |
| Discovery calls | 3+ |
| Demos | 2+ |
| Proposals | 1+ |

### 14.5 Monthly Conversion Checkpoints

| Stage | Target Conversion |
|-------|-------------------|
| Discovery → Demo | ≥ 50% |
| Demo → Proposal | ≥ 50% |
| Proposal → Paid pilot | ≥ 50% |

---

## 15. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **No early proof points** | High | High | Enforce KPI instrumentation from day 1 of each pilot. Structure pilots as measurable outcomes projects. |
| **DIY alternatives viewed as "good enough"** | High | Medium | Sell time-to-value and operational support, not only technical capability. Lead with total cost of ownership. |
| **Long security reviews at customer** | Medium | High | Pre-package security architecture brief and deployment checklist. Air-gap story is a strength here. |
| **Budget friction in SMB** | Medium | Medium | Clear fixed monthly pricing ($850/mo). Phased rollout option. No surprise costs. |
| **DGX Spark supply constraints** | Medium | High | Hosted deployment model as alternative. Limited initial Spark inventory acknowledged. |
| **Model quality insufficient for domain** | Medium | Medium | 90-day quality ramp with RAGAS evaluation. Model flexibility (swap models without code changes). |
| **AD/SSO integration delays** | Low-Medium | High | SSO is MVP requirement for enterprise sales. Prioritize in sprint planning. |
| **Multi-language demand before readiness** | Low | Medium | English-first MVP. Evaluate Spanish as fast-follow. qwen3:8b has multilingual capability. |
| **Single points of failure (key personnel)** | Medium | High | Document all operational procedures. Automate deployment. |

> **[PLACEHOLDER — Paul/Jason]**: Additional risks from business perspective (funding, partnership, regulatory).

---

## 16. Open Items & Decisions Required

### For Thursday Meeting (Feb 13)

| # | Decision | Owner | Options | Notes |
|---|----------|-------|---------|-------|
| 1 | **Product name** | Paul/Jason | "AI Ready RAG" vs. alternatives | Branding for market positioning |
| 2 | **Hardware bundling model** | Paul/Jason | (a) Customer buys Spark separately, (b) Bundle pricing, (c) Lease-to-own | Affects pricing model and sales motion |
| 3 | **Hosted infrastructure** | Paul/Jason | Cloud provider, VM sizing, cost model per customer | Needed for hosted deployment offering |
| 4 | **Post-pilot pricing** | Paul/Jason | Subscription tiers, support tiers, volume discounts | Can define later but need direction |
| 5 | **Spanish language** | Paul/Jason | (a) Include in MVP, (b) Fast-follow, (c) Evaluate only | qwen3:8b supports multilingual, but embedding model is English-primary |
| 6 | **Engineering resources** | Paul/Jason | Team size, allocation, sprint plan | How many engineers? Full-time? |

### Placeholders to Complete Before Finalization

| # | Item | Owner | Status |
|---|------|-------|--------|
| 1 | TAM/SAM/SOM market sizing | Paul/Jason | Need quantified data |
| 2 | Customer interview quotes / validation | Paul/Jason | Pending internal testing (Mar 16+) |
| 3 | Financial model (revenue projections, cost structure, break-even) | Paul/Jason | Not yet started |
| 4 | Hosted infrastructure cost model | Paul/Jason | Not yet defined |
| 5 | Engineering sprint plan (Mar 16 → May 25) | Paul/Jason | Not yet defined |
| 6 | Post-pilot conversion pricing | Paul/Jason | Define later |
| 7 | Product name finalization | Paul/Jason | Options discussed, no decision |
| 8 | Partnership strategy (MSP/IT consultants) | Paul/Jason | Mentioned in GTM, details TBD |

---

## 17. Appendices

### Appendix A: Research Documents

All supporting research is in the `docs/` directory:

| Document | Description |
|----------|-------------|
| `competitor-analysis-2026-02-10.md` | 15-competitor ranked analysis |
| `competitor-battle-matrix-2026-02-10.md` | Scored 0–5 competitive matrix |
| `competitor-pricing-snapshot-2026-02-10.md` | Public pricing evidence for all 15 competitors |
| `where-we-win-where-we-dont-2026-02-10.md` | Win/loss positioning guide |
| `top-5-battlecards-2026-02-10.md` | Top 5 competitor battlecards |
| `outbound-messaging-pack-2026-02-10.md` | Email sequences, LinkedIn, discovery script, demo flow, objection handling |
| `pilot-gtm-action-plan-2026-02-10.md` | 6-month GTM execution plan |
| `snowflake-pricing-estimate-2026-02-10.md` | Snowflake Cortex Search pricing comparison |
| `competitor-ui-demo-library-2026-02-10.md` | Competitor demo and UI reference links |
| `github-open-source-rag-pattern-benchmark-2026-02-11.md` | OSS RAG pattern benchmark |
| `SPARK_DEPLOYMENT.md` | Full DGX Spark deployment guide |

### Appendix B: Technical Specifications

| Spec | Version | Status | Location |
|------|---------|--------|----------|
| RAG Evaluation Framework | v1.3 | DRAFT | `specs/RAG_EVALUATION_FRAMEWORK_v1.md` |
| Excel Tiered Ingestion | v2.0 | DRAFT | `ingestkit/packages/ingestkit-excel/SPEC.md` |

### Appendix C: Meeting Transcripts

| Transcript | Date | Participants | Topics |
|-----------|------|--------------|--------|
| First Linus Session | 2026-02-09 | Paul, Jason (with Linus AI) | Enterprise knowledge base concept, two markets, competitive landscape, pricing, SSO, roadmap |

### Appendix D: Industry RAG Quality Benchmarks (Detail)

**Sources surveyed:**
- RAGAS official documentation and leaderboard
- TruLens production baselines
- DeepEval evaluation thresholds
- LangChain evaluation guides
- Enterprise RAG deployment surveys (2025)

**Faithfulness benchmarks:**
- Industry minimum: 0.70 (DeepEval)
- Industry recommended: 0.80 (TruLens)
- Industry top-quartile: 0.90+ (RAGAS leaderboard)

**Answer Relevancy benchmarks:**
- Industry minimum: 0.70 (DeepEval)
- Industry recommended: 0.75 (TruLens)
- Industry top-quartile: 0.85+ (RAGAS leaderboard)

**Context Precision benchmarks:**
- Industry minimum: 0.60 (DeepEval)
- Industry median: 0.65 (enterprise surveys)
- Industry top-quartile: 0.80+ (RAGAS leaderboard)

**Context Recall benchmarks:**
- Industry minimum: 0.70 (DeepEval)
- Industry median: 0.75 (enterprise surveys)
- Industry top-quartile: 0.85+ (RAGAS leaderboard)

---

*This document is a DRAFT prepared for the February 13, 2026 review meeting. All placeholders marked with `[PLACEHOLDER — Paul/Jason]` require input before finalization.*
