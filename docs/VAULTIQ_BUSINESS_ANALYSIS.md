# VaultIQ — Executive Business Analysis

### Intelligent Insurance Document Management & Automation Platform

**Prepared for**: VTM Group Executive Team
**Date**: February 20, 2026
**Version**: 1.1 — DRAFT
**Classification**: Confidential

---

## Executive Summary

VaultIQ is a purpose-built AI platform that transforms how insurance agencies manage documents and access knowledge across their book of business. By combining advanced document intelligence with structured data extraction and natural-language querying, VaultIQ gives every agent in an agency instant access to every fact across every property they manage.

**The opportunity**: 39,000 independent P&C agencies in the US manage $510 billion in commercial lines premium. These agencies collectively employ 686,000 agents who spend 30-40% of their time on document lookup, renewal preparation, and compliance verification — tasks that VaultIQ automates. Despite $228 billion in annual insurance IT spending, no incumbent technology vendor provides AI-powered document intelligence for independent agencies. The existing AMS market (dominated by Applied Systems and Vertafore) tracks transactions but cannot read, understand, or analyze the content of insurance documents.

**Why now**: Three converging factors have created this market:
1. Large language models (Claude, GPT) can now reliably extract structured data from complex insurance documents — a capability that didn't exist 18 months ago
2. Insurance agency AI adoption is at 6% — the market is pre-early-adopter, creating a first-mover window
3. Agency consolidation and staffing pressure demand operational leverage that current tools cannot provide

**The business model**: Cloud-hosted SaaS with a **single-tenant architecture** — each agency gets its own dedicated instance (application + database), ensuring complete data isolation and per-customer customization. Core platform handles 80%+ of functionality across all commercial lines verticals. Niche-specific modules (community associations, construction, transportation, healthcare) add 10-20% of specialized capability per vertical.

**Conservative 5-year projection**: $12M ARR by Year 5 from 400 agency customers at an average of $2,500/month, with 78% gross margins and positive unit economics by Month 8 per customer.

---

## Table of Contents

1. [Market Opportunity](#market-opportunity)
2. [The Untapped Market: Why Now](#the-untapped-market)
3. [Competitive Landscape](#competitive-landscape)
4. [Product Architecture & Vertical Scalability](#product-architecture)
5. [Customer ROI Analysis](#customer-roi-analysis)
6. [Business Model & Unit Economics](#business-model)
7. [Go-to-Market Strategy](#go-to-market-strategy)
8. [Risk Analysis](#risk-analysis)

**Appendices**:
- [A: TAM/SAM/SOM Methodology](#appendix-a)
- [B: Detailed Competitor Profiles](#appendix-b)
- [C: Customer ROI Model — Worked Examples](#appendix-c)
- [D: 5-Year Financial Model](#appendix-d)
- [E: AI Technology Landscape](#appendix-e)
- [F: Market Research Sources](#appendix-f)

---

<a id="market-opportunity"></a>
## 1. Market Opportunity

### Total Addressable Market (TAM)

The TAM represents all US independent P&C agencies that manage commercial lines business and could benefit from AI-powered document intelligence.

| Metric | Value | Source |
|---|---|---|
| Independent P&C agencies in the US | 39,000 | Insurance Business Mag / Future One 2024 |
| Agencies with commercial lines focus | ~27,300 (70%) | Industry estimate — 70% write commercial |
| Average agency technology spend | 5% of premium revenue | Gartner / SimpleSolve 2024 |
| Average agency revenue | $300K-$500K | Reagan/IIABA Best Practices |
| Total agency IT spending (addressable) | ~$5.5B annually | 27,300 agencies × $400K avg rev × 5% |

**TAM: $5.5 billion** in annual agency technology spending across commercial lines agencies.

For VaultIQ specifically (document intelligence + automation SaaS):

| Metric | Value |
|---|---|
| Commercial lines agencies (addressable) | 27,300 |
| VaultIQ annual price point (avg) | $30,000/year ($2,500/mo) |
| **Product-specific TAM** | **$819 million/year** |

### Serviceable Addressable Market (SAM)

The SAM narrows to agencies where VaultIQ delivers the highest ROI: mid-size agencies with 50+ commercial accounts and document-heavy niches.

| Segment | Agency Count | Avg Deal Size | SAM |
|---|---|---|---|
| Community association specialists | 3,500 | $36,000/yr | $126M |
| Construction insurance agencies | 5,000 | $30,000/yr | $150M |
| Real estate / property management | 4,000 | $24,000/yr | $96M |
| General commercial mid-size (100+ accts) | 6,000 | $30,000/yr | $180M |
| Transportation specialists | 2,500 | $24,000/yr | $60M |
| Healthcare / professional liability | 2,000 | $24,000/yr | $48M |
| **Total SAM** | **23,000** | | **$660 million/year** |

### Serviceable Obtainable Market (SOM)

Year 1-3 realistic capture with focused go-to-market:

| Year | Target Segment | Agencies | Revenue |
|---|---|---|---|
| Year 1 | Community associations (beachhead) | 50 | $1.5M |
| Year 2 | + Construction, real estate | 150 | $4.5M |
| Year 3 | + General commercial, transportation | 300 | $8.5M |

**SOM by Year 3: $8.5 million ARR** (1.3% of SAM — conservative)

### Market Sizing Rationale

The $660M SAM is conservative for three reasons:

1. **Only includes mid-size+ agencies**: Excludes small agencies (<50 commercial accounts) that may still benefit but have lower willingness to pay
2. **Single product price point**: Does not include upsell to premium features (certificate automation, client portal, carrier analytics)
3. **US only**: No international markets, Lloyd's/London market, or Canadian expansion included

See [Appendix A](#appendix-a) for detailed methodology.

---

<a id="the-untapped-market"></a>
## 2. The Untapped Market: Why This Opportunity Exists Now

### 2.1 The Gap Nobody Has Filled

The insurance technology landscape has a structural blind spot. Every incumbent serves one side of the data divide:

```
TRANSACTION SYSTEMS                    DOCUMENT CONTENT
(What was bound)                       (What's in the policy)
─────────────────                      ────────────────────
Applied Epic                           ???
Vertafore AMS360                       ???
HawkSoft                               ???
AgencyZoom                             ???
InsuredMine                            ???

"Policy #CPP0294618                    "Policy #CPP0294618 has a
 was bound 12/01/2025                   $1M per-occurrence GL limit
 Premium: $8,234"                       with a $5,000 deductible and
                                        excludes EIFS construction"
```

**Every AMS in the market tracks the transaction. None of them understand the document.** This is the gap VaultIQ fills.

The AMS knows a policy was bound. VaultIQ knows what's in it — the limits, the exclusions, the endorsements, the coverage gaps, the compliance requirements. These are fundamentally different capabilities, and they complement rather than compete with each other.

### 2.2 Why AI Enables This Now (Not Before)

This market didn't exist before 2024. Here's why:

#### Before LLMs (Pre-2023)

Traditional OCR and NLP could extract text from documents but couldn't *understand* insurance language:

- "Each Occurrence: $1,000,000" → OCR could capture the text
- But it couldn't determine: Is this GL? Property? D&O? Is this the limit or the deductible? Is this the current policy or an expired one?
- Rule-based extraction required hand-crafted parsers for every document format, every carrier, every form version
- **Chisel AI tried this approach** (500+ insurance-specific entity rules) and failed ($1.8M revenue at shutdown)

#### After LLMs (2024-Present)

Large language models fundamentally changed what's possible:

| Capability | Before LLMs | With LLMs (Claude/GPT) |
|---|---|---|
| Read an ACORD form | OCR captures text fields | Understands field relationships and meaning |
| Read a policy dec page | Extracts text strings | Identifies coverage type, limits, conditions |
| Read a loss run | Captures numbers | Understands claim status, trends, severity |
| Read a CC&R | Extracts paragraphs | Identifies insurance requirements, compares to coverage |
| Handle format variation | Breaks with new layouts | Adapts to any carrier's format |
| Cross-document analysis | Not possible | "Compare limits across all properties" |

**The key insight**: LLMs don't just extract text — they *understand context*. This is the difference between "the number $1,000,000 appears on page 3" and "the per-occurrence GL limit is $1,000,000, issued by CondoLogic under policy #CPP0294618, effective 12/01/2025."

#### The Cost Curve Made It Viable

| Year | Cost to Process 1 Document (AI) | Viable for Agencies? |
|---|---|---|
| 2022 | $5-10 (GPT-3, high token costs) | No — too expensive |
| 2023 | $1-2 (GPT-4, early pricing) | Marginal |
| 2024 | $0.10-0.30 (Claude Sonnet, prompt caching) | Yes — approaching free |
| 2025-26 | $0.04-0.06 (Claude with caching) | **Yes — negligible cost** |

At $0.04 per document, processing a 50-property book (2,500 documents) costs $100 total. This is a rounding error against the time savings.

### 2.3 Why Incumbents Won't Build This

**Applied Systems** (dominant AMS) won't build VaultIQ because:
- Their business model is transaction management, not document intelligence
- They acquired Indio for form *filling* (input), not document *reading* (output)
- Their architecture is built around structured data entry by agents, not AI extraction from documents
- Adding deep document AI would require rebuilding their data model from scratch
- They're focused on carrier connectivity and real-time quoting — different problem

**Vertafore** won't build it because:
- Same transaction-first architecture as Applied
- Fragmented product portfolio from 14+ acquisitions already strains integration capacity
- Their AI investment is going toward workflow automation, not document comprehension
- Acquisition of AgencyZoom shows CRM/sales focus, not knowledge management focus

**Carrier-focused AI companies** (Roots.ai, AgentFlow) won't serve agencies because:
- Built for carrier operations (claims processing, underwriting) at enterprise scale
- Different data model — carriers see policies they issue, not the multi-carrier book an agency manages
- Pricing and deployment complexity designed for $1B+ carriers, not $500K revenue agencies

**Generic AI tools** (ChatGPT, Copilot) won't solve this because:
- No insurance domain knowledge built in
- No structured data extraction into queryable databases
- No ACORD form understanding
- No multi-document, multi-property knowledge management
- No compliance checking against CC&Rs or bylaws
- No access control (every agent sees everything)

### 2.4 Market Timing: The 6% Adoption Window

According to the Agent for the Future / IIABA study (2024):

- **6%** of agency principals have implemented an AI solution
- **20%** are starting to learn about AI
- **27%** are interested but it's not a priority
- **36%** say they'll likely use AI within 5 years

This is pre-early-adopter territory. The market is:
- **Aware** that AI is important (83% have heard of it)
- **Uncertain** about what to buy (no clear category leader)
- **Waiting** for a purpose-built solution (not willing to cobble together generic tools)

**First-mover advantage is available.** The window is 18-24 months before incumbents react or a well-funded competitor emerges.

---

<a id="competitive-landscape"></a>
## 3. Competitive Landscape

### 3.1 Category Map

```
                    AGENCY-FOCUSED
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Transaction      Knowledge       Sales/CRM
    Management       Intelligence    Automation
         │               │               │
    Applied Epic     VaultIQ ★       InsuredMine
    Vertafore        (no incumbent)  AgencyZoom
    HawkSoft                        Zywave
         │                               │
         │         CARRIER-FOCUSED        │
         │               │               │
         │          Roots.ai             │
         │          AgentFlow            │
         │          Allianz+Claude       │
         │               │               │
         └───────────────┼───────────────┘
                         │
                   NOT AGENCY-FOCUSED
```

VaultIQ creates a **new category**: Insurance Knowledge Intelligence for independent agencies. No incumbent occupies this space.

### 3.2 Competitive Comparison Matrix

| Capability | VaultIQ | Applied Epic | Vertafore | HawkSoft | Roots.ai |
|---|---|---|---|---|---|
| **Reads & understands policy documents** | Yes | No | No | No | Yes (carriers only) |
| **Extracts structured coverage data** | Yes | Manual entry | Manual entry | Manual entry | Yes (carriers only) |
| **Natural language Q&A across book** | Yes | No | No | No | No |
| **Compliance gap detection** | Yes | No | No | No | No |
| **Renewal prep automation** | Yes | Partial | Partial | No | No |
| **Multi-carrier program comparison** | Yes | No | No | No | No |
| **ACORD form understanding** | Yes | Form filling (Indio) | No | Pre-fill only | No |
| **Loss run analysis & trending** | Yes | No | No | No | Partial |
| **For independent agencies** | Yes | Yes | Yes | Yes | No |
| **Dedicated instance per customer** | Yes | No (shared) | No (shared) | No (shared) | No |
| **Air-gap deployment option** | Yes | No | No | No | No |
| **SaaS / cloud hosted** | Yes | Yes | Yes | Yes | Yes |
| **Setup time** | Same day | Weeks-months | Weeks-months | 1-2 weeks | Months |

### 3.3 Competitive Moat

VaultIQ's defensibility comes from four sources:

1. **Domain-specific AI models**: Insurance document extraction prompts, entity taxonomies, and compliance rules built over hundreds of real-world documents. This is accumulated knowledge that takes months to replicate.

2. **Structured insurance database**: The more documents processed, the richer the structured data. Historical premium trends, coverage change history, and loss patterns across a portfolio create compounding value that increases switching costs. Unified PostgreSQL + pgvector architecture enables cross-referencing that competitors using separate systems cannot match.

3. **Vertical module library**: Each niche (community associations, construction, transportation) has specific document types, compliance rules, and workflows. Building these takes domain expertise plus engineering. A generic AI tool can't replicate this without the same investment.

4. **Single-tenant customization**: Each customer's dedicated instance can be configured to their specific workflow, compliance rules, and branding. This creates deep integration with each agency's operations that multi-tenant competitors cannot offer without significant architecture changes.

### 3.4 Closest Competitive Threat

**AgentiveAIQ** — an emerging RAG-based AI agent platform for insurance agencies. Uses a dual knowledge base (RAG + knowledge graph) for policy document Q&A. Limited public information available, but this is the most directly competitive product identified.

**Risk level**: Moderate. They have the right idea but appear to be early-stage with no visible structured data extraction, compliance checking, or automation capabilities beyond basic Q&A.

**Response**: Move fast on go-to-market. Our technical architecture (Claude enrichment + structured SQL + query routing) is more sophisticated than generic RAG. Lock in beachhead customers before competitors gain traction.

See [Appendix B](#appendix-b) for detailed competitor profiles.

---

<a id="product-architecture"></a>
## 4. Product Architecture & Vertical Scalability

### 4.1 Core Platform (80%+ Shared Across All Verticals)

The VaultIQ core platform handles capabilities universal to every commercial lines insurance agency:

| Core Capability | Description | % of Agent Time Saved |
|---|---|---|
| **Document Intelligence** | Upload any insurance document → AI reads, classifies, and extracts structured data | Foundational |
| **Structured Insurance Database** | Policies, coverages, claims, certificates in queryable SQL tables | Foundational |
| **Natural Language Q&A** | Ask any question in plain English, get cited answers | 90% of lookup time |
| **Deterministic Query Router** | SQL first for facts (instant), RAG fallback for analysis (cited) | Transparent to user |
| **Coverage Dashboard** | At-a-glance view of any account's program | 5 min/account saved |
| **Loss History Consolidation** | 5-year view across all lines, all carriers | 20-30 min saved |
| **Renewal Prep** | One-click coverage schedule, premium history, expiring details | 2-3 hrs/property saved |
| **Access Control** | Tag-based visibility — agents see only their assigned properties | Security requirement |
| **Audit Trail** | Every query, every answer, every source cited | Compliance requirement |

**Universal document types handled by core**:
- Policies and declarations pages (all lines)
- ACORD certificates (25, 24, 27, 28)
- Loss runs (all carriers)
- Endorsements and amendments
- Applications and submissions
- Quotes and proposals
- Bind orders and binders

### 4.2 Vertical Modules (10-20% Niche-Specific)

Each vertical adds specific document types, entity extraction rules, compliance checks, and workflows:

#### Community Associations (Launch Vertical)

| Vertical Addition | What It Does |
|---|---|
| CC&R/Bylaw compliance checking | Cross-reference coverage against governing document requirements |
| Reserve study analysis | Extract funding levels, recommend coverage adjustments |
| Unit owner letter generation | Auto-generate HO6 requirement letters from current coverage data |
| Board presentation package | Coverage summary + compliance check + renewal recommendation |
| Multi-policy program view | Property + GL + D&O + Crime + WC + Umbrella in one dashboard |

**Unique document types**: CC&Rs, bylaws, reserve studies, unit owner letters, board meeting minutes, appraisals

**Build effort**: Already built (Marshall Wells Lofts is the test case)

#### Construction Insurance

| Vertical Addition | What It Does |
|---|---|
| Subcontractor certificate tracking | Track COIs for every sub on every project |
| Certificate compliance verification | Verify sub coverage meets contract requirements (limits, AI, WOS) |
| Wrap-up / OCIP management | Track owner-controlled or contractor-controlled insurance programs |
| Builder's risk tracking | Monitor single-project policies with varying terms |
| Project-based coverage view | See all insurance tied to a specific construction project |

**Unique document types**: Subcontractor agreements, wrap-up policies, builder's risk, OCIP/CCIP schedules

**Build effort**: 2-3 weeks on top of core

#### Real Estate / Property Management

| Vertical Addition | What It Does |
|---|---|
| Tenant certificate tracking | Track tenant COIs across portfolio of managed properties |
| Lease insurance requirement extraction | Pull insurance requirements from lease agreements |
| Landlord/tenant coverage comparison | Compare tenant's coverage against lease requirements |
| Building schedule management | Track SOVs and building values across portfolio |

**Unique document types**: Lease abstracts, tenant certificates, building schedules, SOVs

**Build effort**: 2 weeks on top of core

#### Transportation

| Vertical Addition | What It Does |
|---|---|
| MCS-90 / DOT filing tracking | Track regulatory filings and compliance |
| Fleet schedule management | Driver lists, vehicle schedules, coverage by unit |
| Cargo coverage tracking | Shipper-specific cargo limits and requirements |
| Driver MVR analysis | Extract and track motor vehicle record data |

**Unique document types**: MCS-90 endorsements, fleet schedules, cargo certificates, IFTA filings, MVRs

**Build effort**: 2-3 weeks on top of core

#### Healthcare / Professional Liability

| Vertical Addition | What It Does |
|---|---|
| Claims-made vs. occurrence tracking | Track retroactive dates, tail coverage needs |
| Credentialing document management | Organize credentialing and privileging documents |
| Consent-to-settle tracking | Flag policies with consent-to-settle provisions |
| Professional liability limit trending | Track limit adequacy over time vs. revenue growth |

**Unique document types**: Credentialing documents, tail coverage endorsements, consent-to-settle letters

**Build effort**: 2 weeks on top of core

### 4.3 Scalability Summary

```
VaultIQ Core Platform (80%+)
├── Document Intelligence Engine (Claude enrichment)
├── Structured Insurance Database (SQL)
├── Natural Language Q&A (query router)
├── Coverage Dashboard
├── Renewal Automation
├── Certificate Management
├── Access Control & Audit
│
├── Community Associations Module (+15%)  ← LAUNCH
├── Construction Module (+15%)            ← Year 1
├── Real Estate Module (+12%)             ← Year 1
├── Transportation Module (+15%)          ← Year 2
├── Healthcare Module (+12%)              ← Year 2
└── General Commercial (+10%)             ← Year 1
```

**Key architectural principles**:

**Single-tenant isolation**: Each customer gets their own dedicated instance — separate application, separate PostgreSQL database (with pgvector for unified structured data + vector storage), separate configuration. No data commingling between customers. This enables per-customer customization and simplifies compliance.

**Unified data layer**: Structured insurance data (policies, coverages, claims) and document vectors live in the same PostgreSQL database using pgvector. This enables SQL JOINs between insurance tables and vector search results — e.g., finding relevant document sections while simultaneously pulling structured coverage data. Eliminates the need for a separate vector database service.

**Vertical modules are configuration layers**, not separate codebases. The same enrichment pipeline, SQL schema, query router, and UI serve every vertical. Modules add:
- Document type classifiers (which extraction rules to apply)
- Entity extraction prompts (what to look for in niche-specific documents)
- Compliance rule sets (what requirements to check against)
- Dashboard templates (what data to surface)

This means a 4-person engineering team can support 5+ verticals simultaneously.

---

<a id="customer-roi-analysis"></a>
## 5. Customer ROI Analysis

### 5.1 Time Savings Model

Based on observed workflows with a mid-size community association agency (50 properties, 3 agents):

| Task | Frequency | Time Today | Time w/ VaultIQ | Hours Saved/Year |
|---|---|---|---|---|
| Answer coverage questions | 10/agent/day | 7 min each | 30 sec each | **1,625 hrs** |
| Renewal preparation | 50 properties/yr | 2.5 hrs each | 15 min each | **188 hrs** |
| Loss run consolidation | 50 properties/yr | 25 min each | 1 min each | **20 hrs** |
| Compliance checking | 50 properties/yr | 45 min each | 2 min each | **36 hrs** |
| Certificate issuance | 300/yr | 20 min each | 3 min each | **85 hrs** |
| Staff onboarding per account | 10 accounts/yr | 3 hrs each | 15 min each | **28 hrs** |
| Policy checking (rarely done today) | 50 policies/yr | 0 (skipped) | 5 min each | **N/A (new capability)** |
| **TOTAL HOURS SAVED** | | | | **~1,982 hrs/year** |

### 5.2 Dollar Value of Time Saved

| Metric | Value |
|---|---|
| Hours saved per year | 1,982 |
| Fully-loaded cost per agent hour | $45/hr ($93K FTE ÷ 2,080 hrs) |
| **Annual labor cost savings** | **$89,190** |
| VaultIQ annual cost (est.) | $30,000 |
| **Net annual savings** | **$59,190** |
| **ROI** | **197%** |
| **Payback period** | **4 months** |

### 5.3 Revenue Uplift (Capacity Gains)

The 1,982 hours saved represents **0.95 FTE**. This capacity can be redirected to revenue-generating activity:

| Scenario | Calculation | Revenue Impact |
|---|---|---|
| Service more properties (same team) | 0.95 FTE × 25 properties/agent capacity | +24 properties serviceable |
| At average premium per property | 24 × $15,000 avg premium | +$360,000 in managed premium |
| At 15% average commission | $360,000 × 15% | **+$54,000 in commission revenue** |

### 5.4 Risk Reduction (E&O Avoidance)

| Risk Area | Without VaultIQ | With VaultIQ | Value |
|---|---|---|---|
| Policy checking errors | Rarely performed | Automated, every policy | Avoids 1-2 E&O claims/decade |
| Compliance gaps missed | Manual cross-reference | Automated detection | Avoids coverage disputes |
| Expired coverage gaps | Calendar-dependent | System-monitored | Avoids lapse exposure |
| Average E&O claim cost | $35,000-$150,000 | Reduced exposure | **$35K+ avoided per incident** |

### 5.5 ROI Summary by Agency Size

| Agency Size | Properties | Agents | Hours Saved | Annual Savings | VaultIQ Cost | ROI |
|---|---|---|---|---|---|---|
| Small (1 agent) | 25 | 1 | 660 hrs | $29,700 | $18,000/yr | 65% |
| Mid-size (3 agents) | 75 | 3 | 1,982 hrs | $89,190 | $30,000/yr | 197% |
| Large (10 agents) | 250 | 10 | 6,600 hrs | $297,000 | $60,000/yr | 395% |
| Enterprise (25 agents) | 500+ | 25 | 16,500 hrs | $742,500 | $120,000/yr | 519% |

**ROI scales with agency size.** Larger agencies save more per dollar spent because the core platform cost doesn't scale linearly with users.

See [Appendix C](#appendix-c) for worked examples by vertical.

---

<a id="business-model"></a>
## 6. Business Model & Unit Economics

### 6.1 Pricing Strategy

**Model**: Monthly SaaS subscription based on portfolio size, not seat count. No per-user fees — this removes adoption friction and encourages agency-wide deployment.

| Tier | Properties/Accounts | Monthly Price | Annual Price | Target Customer |
|---|---|---|---|---|
| Starter | Up to 25 | $1,500 | $18,000 | Solo agent, small book |
| Professional | Up to 100 | $2,500 | $30,000 | Mid-size agency |
| Enterprise | Up to 500 | $5,000 | $60,000 | Large agency/brokerage |
| Custom | 500+ | Custom | $100K+ | Regional brokerages |

**Pricing rationale**:
- Anchored to customer ROI (Starter tier pays back in 6 months at 25 properties)
- Portfolio-based pricing aligns VaultIQ's value with agency growth
- No per-seat fee means principals don't hesitate to give access to every team member
- Competitive with AMS add-on costs ($85-100/user/month × 5 users = $500/month for HawkSoft alone)

### 6.2 Revenue Streams

| Stream | % of Revenue | Description |
|---|---|---|
| **Core SaaS subscription** | 75% | Monthly platform access |
| **Vertical modules** | 15% | Niche-specific add-ons (construction cert tracking, etc.) |
| **Professional services** | 10% | Onboarding, custom integration, training |

### 6.3 Unit Economics

**Per-customer economics (Professional tier)**:

Each customer gets a dedicated instance (Docker Compose stack with FastAPI + PostgreSQL/pgvector). No shared databases, no multi-tenant complexity.

| Metric | Value |
|---|---|
| Monthly revenue | $2,500 |
| Annual revenue | $30,000 |
| **Cost of Goods Sold (COGS)** | |
| Cloud infrastructure (VM + PostgreSQL/pgvector, dedicated instance) | $120/mo |
| Claude API — enrichment (one-time, amortized) | $25/mo |
| Claude API — queries (~100 queries/day) | $50/mo |
| Support allocation | $100/mo |
| **Total COGS** | **$295/mo** |
| **Gross Margin** | **$2,205/mo (88%)** |
| | |
| **Customer Acquisition Cost (CAC)** | |
| Sales cycle (estimated 60 days) | $3,000 |
| Marketing attribution | $1,500 |
| Onboarding/implementation | $2,000 |
| **Total CAC** | **$6,500** |
| | |
| **LTV (36-month avg retention)** | **$79,380** (88% margin × $30K × 3 yrs) |
| **LTV:CAC Ratio** | **12.2:1** |
| **Payback Period** | **2.6 months** |

**Single-tenant cost advantage**: By using PostgreSQL + pgvector as a unified data store (no separate Qdrant vector database), infrastructure cost per customer is lower than a multi-service architecture. The single-tenant model costs more in aggregate than multi-tenant at small scale, but provides complete data isolation, per-customer customization, and simpler compliance — critical for insurance agencies handling sensitive client data.

### 6.4 Margin Analysis

Each customer runs on a dedicated instance with a unified PostgreSQL + pgvector database (no separate vector database service needed).

| Component | Monthly Cost | Notes |
|---|---|---|
| Cloud compute (per customer) | $50-75 | Docker Compose on AWS/GCP, dedicated VM |
| PostgreSQL + pgvector (per customer) | $30-45 | Managed PostgreSQL with pgvector extension |
| Claude API — enrichment | $25 (amortized) | One-time per document, ~$0.04/doc |
| Claude API — queries | $30-80 | Usage-dependent, Haiku for simple queries |
| Storage (documents) | $5-15 | S3/equivalent for uploaded files |
| **Total infrastructure per customer** | **$140-240/mo** | |
| **Gross margin range** | **84-94%** | Depends on query volume |

**Key insight**: The Claude API cost for queries is the primary variable cost. Heavy users (~250 queries/day) cost ~$80/month in API fees. Light users (~20 queries/day) cost ~$10/month. The Haiku/Sonnet model routing strategy keeps costs low — 70% of queries use Haiku at ~$0.001/query.

**Single-tenant scaling note**: Infrastructure costs scale linearly with customers (each gets a dedicated instance). At 50+ customers, container orchestration (Kubernetes or equivalent) reduces per-instance overhead through shared node pools while maintaining complete data isolation. At 200+ customers, the operational tooling investment (automated provisioning, monitoring, rolling updates) pays for itself.

### 6.5 Expansion Revenue

Once a customer is onboarded:

| Upsell Opportunity | Trigger | Revenue |
|---|---|---|
| Tier upgrade | Agency grows book, needs more properties | +$1,000-2,500/mo |
| Vertical module | Enters new niche (e.g., adds construction) | +$500-1,000/mo |
| Client portal add-on | Board/PM wants self-service access | +$500/mo |
| Certificate automation | High certificate volume | +$300-500/mo |
| Additional agencies | Franchise/network referral | New customer |

**Estimated net revenue retention**: 115-120% (expansion exceeds churn)

---

<a id="go-to-market-strategy"></a>
## 7. Go-to-Market Strategy

### 7.1 Beachhead: Community Association Insurance

**Why start here**:
- Highest document complexity (CC&Rs, bylaws, reserve studies add unique value)
- Board governance creates demand for compliance checking and reporting
- Concentrated market — 3,500 agencies, reachable through industry associations
- Sara Eanni at Associa is our first reference customer and design partner
- VaultIQ already tested against real Marshall Wells Lofts documents

**Target accounts (Year 1)**: 50 community association agencies

**Channel**:
- Community Associations Institute (CAI) annual conference
- IIABA (Big I) state association events
- Direct outreach to top 200 community association agencies
- Content marketing: "How AI changes community association insurance"
- Pilot program: one property free for 30 days

### 7.2 Expansion: Construction & Real Estate (Year 2)

**Why these next**:
- Construction has the highest certificate volume in insurance (massive VaultIQ ROI)
- Real estate/property management agencies often overlap with community associations
- Both niches use the same core ACORD forms
- Vertical module build time is 2-3 weeks each

### 7.3 Sales Motion

| Stage | Action | Timeline |
|---|---|---|
| **Awareness** | Content + conference presence + referrals | Ongoing |
| **Demo** | Live demo with customer's own documents (1 property) | 30 min |
| **Pilot** | Upload one property, use for 30 days free | 30 days |
| **Close** | Expand to full book, annual contract | Week 5-6 |
| **Onboard** | Upload remaining properties, train team | 2-4 weeks |
| **Expand** | Add verticals, client portal, certificates | Months 6-12 |

**The demo is the product.** Upload a customer's actual policy PDF during the demo. Watch the system extract carrier, limits, policy number, and coverage details in real time. Ask a question about the document. Get a cited answer. This is the moment that sells.

### 7.4 Pricing Psychology

- **No per-seat pricing**: Removes the "how many seats do I need?" friction. Principal buys it for the agency, everyone uses it.
- **Portfolio-based tiers**: Aligns with how agencies think about their business (number of accounts, not number of users)
- **30-day pilot with real data**: Low risk. Customer sees value on their own documents before committing.
- **Annual contract with monthly option**: 10% discount for annual commitment

---

<a id="risk-analysis"></a>
## 8. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Applied/Vertafore builds competing AI feature** | Medium (18-24 months) | High | Move fast. Lock in 100+ customers before incumbents react. Build switching costs through structured data accumulation. |
| **Claude API pricing increases** | Low | Medium | Architecture supports model swapping. Haiku handles 70% of queries. Can route to other providers (Gemini, GPT) if needed. At scale (15+ customers), can add shared GPU pool for simple queries. |
| **Claude API reliability** | Low | Medium | Graceful degradation: serve cached results (24-hour TTL) and SQL lookups when API is unavailable. Enrichment is one-time, not continuous. Structured queries work without any API dependency. |
| **Customer data sensitivity concerns** | Medium | Medium | Single-tenant architecture provides complete data isolation per customer. Enterprise tier (air-gapped) addresses strictest requirements. Standard tier uses encrypted, dedicated instances in US data centers. PII fields receive additional encryption. SOC 2 compliance on roadmap. |
| **Slow agency AI adoption** | Medium | Medium | 6% adoption today means early majority hasn't started. Product must be "upload and ask" simple. Pilot program reduces risk for buyers. |
| **Well-funded startup enters space** | Medium | High | First-mover advantage + vertical module library + customer data moat. AgentiveAIQ is the closest threat — monitor closely. |
| **Extraction accuracy concerns** | Low | High | Factual answers (limits, premiums) come from structured database, not AI interpretation. Every analytical answer cites source document. Confidence scoring flags uncertain results. Low-confidence answers route to human review queue — never presented as definitive. |
| **Single-tenant operational scale** | Medium | Medium | Linear infrastructure growth with customer count. Mitigated by automated provisioning, container orchestration (K8s at 50+ customers), and unified PostgreSQL + pgvector (fewer services per instance). DevOps investment starts Year 2. |

---

## Conclusion

VaultIQ addresses a $660M SAM in a market with zero incumbent competition, 6% AI adoption, and universal pain points driven by standardized forms and workflows. The single-tenant architecture provides complete data isolation and per-customer customization — critical differentiators for insurance agencies handling sensitive client data. The product architecture scales across verticals with 80%+ shared core and 2-3 week niche module builds. Customer ROI ranges from 65% (small agency) to 519% (enterprise) with 4-month average payback.

The window for first-mover advantage is 18-24 months. AI has created a new category — Insurance Knowledge Intelligence — that didn't exist before 2024. The incumbents are structurally misaligned to build it. The market is waiting for a purpose-built solution.

**Recommendation**: Invest in go-to-market for community association vertical immediately. Target 50 customers in Year 1, expand to 3 verticals by Year 2, and build toward $12M ARR by Year 5.

---
---

# Appendices

---

<a id="appendix-a"></a>
## Appendix A: TAM/SAM/SOM Methodology

### TAM Calculation

**Step 1: Universe of agencies**
- 442,833 total insurance brokerages & agencies in the US (IBISWorld 2025)
- 39,000 are independent P&C agencies (Insurance Business Mag / Future One 2024)
- ~70% have some commercial lines business = 27,300 commercial lines agencies

**Step 2: Willingness to pay**
- Average agency revenue: $300K-$500K (Reagan/IIABA Best Practices)
- Average IT spend: 5% of revenue = $15K-$25K (Gartner/SimpleSolve)
- VaultIQ average annual price: $30,000
- Addressable agencies (can afford VaultIQ): agencies at $500K+ revenue
- Estimated: 27,300 agencies × $30K = $819M

**Step 3: TAM context check**
- US insurance IT spending: $227.7B (Gartner 2025) — $819M is 0.36% of total
- InsurTech market: $20.3B (2025) — $819M is 4.0% of InsurTech
- Both benchmarks confirm TAM is conservative and plausible

### SAM Calculation

**Step 1: High-value segments**

| Segment | How Estimated | Agency Count |
|---|---|---|
| Community association | CAI membership + specialist directories | 3,500 |
| Construction | IIABA construction niche surveys | 5,000 |
| Real estate / property mgmt | NAR + PM agency overlap | 4,000 |
| General commercial mid-size | IIABA agencies with $500K+ revenue | 6,000 |
| Transportation | IIABA transportation specialist count | 2,500 |
| Healthcare / professional | IIABA healthcare niche | 2,000 |
| **Total** | | **23,000** |

**Step 2: SAM = segment count × weighted average deal size**
- Weighted avg deal size: $28,700/year
- SAM: 23,000 × $28,700 = $660M

### SOM Calculation

**Assumptions**:
- Year 1: 50 customers in beachhead (community associations) — 1.4% of segment
- Year 2: 150 customers across 3 segments — 0.9% of combined segments
- Year 3: 300 customers across 5 segments — 1.3% of SAM

These penetration rates are conservative — typical SaaS products achieve 2-5% of SAM within 3 years.

---

<a id="appendix-b"></a>
## Appendix B: Detailed Competitor Profiles

### Applied Systems (Applied Epic)

| Attribute | Detail |
|---|---|
| **Market position** | #1 AMS globally; 8 of top 10 US brokers; 69 of Insurance Journal Top 100 |
| **Funding** | $7.12B total capital (Hellman & Friedman PE) |
| **Users** | 160,000+ across 4 operating regions |
| **Key products** | Applied Epic (AMS), Indio (submissions), Applied CSR24 (client self-service) |
| **AI capabilities** | Limited; Indio handles form auto-fill but not document intelligence |
| **Pricing** | $1,000+ entry; enterprise contracts custom |
| **Weakness** | Cannot read or analyze policy documents. Transaction system only. High cost and implementation complexity. |

### Vertafore

| Attribute | Detail |
|---|---|
| **Market position** | #2 AMS; ~11.55% market share; 80,000 users on AMS360 |
| **Funding** | Bain Capital + Vista Equity Partners (acquired ~$2.7B in 2016) |
| **Key products** | AMS360 (small/mid AMS), Sagitta (enterprise), AgencyZoom (CRM) |
| **AI capabilities** | Emerging; not deeply embedded |
| **Weakness** | Fragmented product portfolio. Aging architecture. CRM/sales focus (AgencyZoom) rather than knowledge management. |

### HawkSoft

| Attribute | Detail |
|---|---|
| **Market position** | #3 AMS for small agencies |
| **Pricing** | $85-100/user/month; $250/month base |
| **Key strength** | Affordable, simple, agency-friendly (data portability, no lock-in) |
| **AI capabilities** | None |
| **Weakness** | No document intelligence. Limited commercial lines depth. |

### Roots.ai

| Attribute | Detail |
|---|---|
| **Market position** | Leading carrier-focused agentic AI platform |
| **Funding** | $44.3M total ($22.2M Series B, Sept 2024) |
| **Investors** | Erie Strategic Ventures, Liberty Mutual Strategic Ventures, CRV |
| **Target** | Insurance carriers and large operations departments |
| **Key strength** | 98%+ accuracy on insurance document extraction; purpose-built for insurance |
| **Weakness** | Not designed for independent agencies. Carrier data model (single-carrier view) vs. agency data model (multi-carrier book). Enterprise pricing and deployment. |

### AgentiveAIQ

| Attribute | Detail |
|---|---|
| **Market position** | Emerging; closest direct competitor |
| **Product** | RAG-powered AI agents for insurance agencies |
| **Approach** | Dual knowledge base: RAG module + Knowledge Graph |
| **Weakness (estimated)** | No visible structured data extraction into SQL. No compliance checking. No automation beyond Q&A. Early stage. |
| **Threat level** | **Moderate — monitor closely** |

### Chisel AI (Defunct — Cautionary Case)

| Attribute | Detail |
|---|---|
| **What they tried** | Purpose-built NLP for insurance documents (policies, quotes, endorsements) |
| **Why they failed** | Pre-LLM technology required 500+ hand-crafted entity rules. Couldn't adapt to format variations. Revenue peaked at $1.8M. |
| **Lesson for VaultIQ** | LLMs solve the technology problem Chisel couldn't. But technology alone isn't enough — go-to-market and revenue model must be right from day one. |

---

<a id="appendix-c"></a>
## Appendix C: Customer ROI Model — Worked Examples

### Example 1: Mid-Size Community Association Agency

**Profile**: 3 agents, 75 properties, $1.1M in managed premium

| Cost Category | Current Annual Cost | With VaultIQ | Savings |
|---|---|---|---|
| Agent time on document lookup | $73,125 (1,625 hrs × $45/hr) | $4,875 | $68,250 |
| Renewal prep labor | $8,460 (188 hrs × $45/hr) | $1,125 | $7,335 |
| Certificate issuance labor | $3,825 (85 hrs × $45/hr) | $675 | $3,150 |
| Compliance checking labor | $1,620 (36 hrs × $45/hr) | $150 | $1,470 |
| Other efficiency gains | $4,410 (98 hrs × $45/hr) | $720 | $3,690 |
| **Total labor savings** | | | **$83,895** |
| Capacity uplift (serve more properties) | | | +$54,000 revenue |
| E&O risk reduction (avoided claims) | | | $35,000+ per incident |
| **VaultIQ annual cost** | | | **($30,000)** |
| **Net annual benefit** | | | **$107,895+** |
| **ROI** | | | **360%** |

### Example 2: Large Construction Insurance Agency

**Profile**: 8 agents, 200 commercial accounts, 1,500 certificates/year

| Benefit | Annual Value |
|---|---|
| Agent time savings (5,280 hrs) | $237,600 |
| Certificate automation (1,500 × 17 min saved) | $19,125 |
| Subcontractor compliance automation | $22,500 |
| Quote comparison automation | $18,000 |
| Policy checking (new capability) | $13,500 |
| **Total savings** | **$310,725** |
| **VaultIQ annual cost** | **($60,000)** |
| **Net annual benefit** | **$250,725** |
| **ROI** | **418%** |

### Example 3: Solo Agent — Small Book

**Profile**: 1 agent, 20 properties, $300K in managed premium

| Benefit | Annual Value |
|---|---|
| Agent time savings (528 hrs) | $23,760 |
| Renewal prep savings | $2,250 |
| Certificate automation | $1,125 |
| **Total savings** | **$27,135** |
| **VaultIQ annual cost** | **($18,000)** |
| **Net annual benefit** | **$9,135** |
| **ROI** | **51%** |

Even the smallest viable customer sees positive ROI, though the payback is longer (8 months vs. 4 months for mid-size).

---

<a id="appendix-d"></a>
## Appendix D: 5-Year Financial Model (Conservative)

### Revenue Projections

| Metric | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| **New customers** | 50 | 100 | 150 | 100 | 75 |
| **Churned customers** (10%/yr) | 0 | 5 | 15 | 30 | 38 |
| **Total customers (end of year)** | 50 | 145 | 280 | 350 | 387 |
| **Avg revenue per customer/mo** | $2,200 | $2,400 | $2,500 | $2,600 | $2,700 |
| **Monthly recurring revenue** | $110K | $348K | $700K | $910K | $1.04M |
| **Annual recurring revenue** | **$1.32M** | **$4.18M** | **$8.40M** | **$10.92M** | **$12.53M** |
| **Growth rate** | — | 217% | 101% | 30% | 15% |

### Assumptions

- **Year 1**: 50 community association agencies at average $2,200/mo (mix of Starter and Professional)
- **Year 2**: Expand to construction + real estate verticals; price creep from upsells
- **Year 3**: Transportation + healthcare verticals; enterprise accounts
- **Year 4-5**: Growth moderates as market penetration increases; focus shifts to retention and expansion
- **Churn**: 10% annual (industry average for vertical SaaS is 8-12%)
- **Price escalation**: 8-10% annual from tier upgrades and module add-ons
- **No international expansion** included

### Cost Structure

| Cost Category | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| **Engineering** (4→8→12 people) | $600K | $1.1M | $1.6M | $1.8M | $2.0M |
| **Sales & Marketing** | $300K | $600K | $900K | $1.0M | $1.1M |
| **Cloud infrastructure** (single-tenant instances) | $50K | $150K | $300K | $400K | $450K |
| **Claude API costs** | $40K | $150K | $350K | $475K | $550K |
| **Customer success** | $100K | $250K | $400K | $500K | $600K |
| **DevOps / instance management** | $0K | $75K | $150K | $175K | $200K |
| **G&A** | $150K | $250K | $350K | $400K | $450K |
| **Total costs** | **$1.24M** | **$2.58M** | **$4.05M** | **$4.75M** | **$5.35M** |

Note: Single-tenant architecture adds a DevOps cost line (automated provisioning, monitoring, rolling updates across customer instances) starting Year 2. Infrastructure costs are slightly lower per customer due to unified PostgreSQL + pgvector (no separate vector database), offsetting the per-instance overhead.

### Profitability

| Metric | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| **ARR** | $1.32M | $4.18M | $8.40M | $10.92M | $12.53M |
| **Total costs** | $1.24M | $2.58M | $4.05M | $4.75M | $5.35M |
| **Operating income** | $80K | $1.60M | $4.35M | $6.17M | $7.18M |
| **Operating margin** | 6% | 38% | 52% | 57% | 57% |
| **Cumulative cash flow** | $80K | $1.68M | $6.03M | $12.20M | $19.38M |

### Key Financial Metrics (Year 5)

| Metric | Value |
|---|---|
| ARR | $12.53M |
| Customers | 387 |
| ARPA (monthly) | $2,700 |
| Gross margin | 78% |
| Operating margin | 57% |
| LTV:CAC | 12:1 |
| Net revenue retention | 115% |
| Cumulative cash flow | $19.38M |
| Architecture | Single-tenant (dedicated instance per customer) |
| Data store | PostgreSQL + pgvector (unified) |

### Sensitivity Analysis

| Scenario | Year 5 ARR | Year 5 Customers | Operating Margin |
|---|---|---|---|
| **Conservative (base case)** | $12.53M | 387 | 59% |
| **Pessimistic** (30% fewer customers, 15% churn) | $7.2M | 240 | 42% |
| **Optimistic** (50% more customers, 8% churn) | $19.8M | 520 | 65% |

---

<a id="appendix-e"></a>
## Appendix E: AI Technology Landscape

### Why Claude (Anthropic) as the AI Backbone

| Factor | Claude | GPT (OpenAI) | Gemini (Google) |
|---|---|---|---|
| **Document comprehension** | Excellent — 200K context window, strong at structured extraction | Good — 128K context | Good — 1M+ context |
| **Structured output (JSON)** | Reliable JSON mode | Reliable JSON mode | Improving |
| **Insurance domain performance** | Validated by Allianz partnership (Jan 2026) | No major insurance partnerships | No major insurance partnerships |
| **Prompt caching** | Yes — 30-40% input cost savings on batch processing | No equivalent | Implicit caching |
| **Cost (Sonnet tier)** | $3/$15 per 1M tokens (input/output) | $2.50/$10 (GPT-4o) | $1.25/$5 (Gemini 1.5 Pro) |
| **Cost (Haiku tier)** | $0.25/$1.25 per 1M tokens | $0.15/$0.60 (GPT-4o-mini) | $0.075/$0.30 (Gemini Flash) |
| **Enterprise trust** | Growing rapidly; Allianz, DHL, Pfizer | Established | Established |
| **Air-gap potential** | Not yet (API only) | Not yet | Not yet |

**Architecture decisions**:
- Use Claude as primary, but abstract the LLM layer so we can swap providers if pricing or quality changes. The enrichment prompts are model-agnostic by design.
- PostgreSQL + pgvector as unified data store — structured insurance data and document vectors in the same database, enabling SQL JOINs between them. Eliminates separate vector database dependency.
- Deterministic query routing — SQL lookup first for factual questions, RAG fallback for analytical queries. No LLM needed to choose the path.
- Single-tenant hosting — each customer gets a dedicated instance for complete data isolation and per-customer customization.
- At scale (15+ customers), option to add shared GPU pool for simple query formatting, reserving Claude API for complex analysis only.

### AI Adoption in Insurance — The Inflection Point

| Metric | 2024 | 2025 | 2026 (projected) |
|---|---|---|---|
| Independent agencies using AI | 6% | ~12% | ~20% |
| Carriers using AI | 34% fully adopted | ~50% | 91% (projected) |
| AI InsurTech funding (% of total InsurTech) | 45% | 74.8% (Q3 2025) | 80%+ |
| AI in insurance market size | $7.7B | $10.3B | $13B+ |

The carrier side is adopting rapidly. The agency side is 2-3 years behind. VaultIQ targets the agency gap.

---

<a id="appendix-f"></a>
## Appendix F: Market Research Sources

### Industry Data

| Source | Data Used | URL/Reference |
|---|---|---|
| IBISWorld | Total US insurance agencies (442,833) | ibisworld.com |
| Insurance Business Mag / Future One | Independent P&C agencies (39,000) | insurancebusinessmag.com |
| BLS | Licensed P&C agents (686,300); median salary ($60,370) | bls.gov/ooh/sales/insurance-sales-agents |
| III (Insurance Information Institute) | Commercial lines premium ($510B) | iii.org |
| Reagan/IIABA Best Practices | Agency revenue benchmarks | reaganconsulting.com |
| Gartner | Insurance IT spending ($227.7B in 2025) | gartner.com |
| CAI | Community associations (373,000) | caionline.org |
| Agent for the Future / IIABA | AI adoption at agencies (6%) | agentforthefuture.com |

### Competitive Intelligence

| Source | Data Used |
|---|---|
| Applied Systems press releases | Market position, Epic adoption, Indio integration |
| Enlyft | Vertafore market share (11.55%) |
| Crunchbase | Funding data for Bold Penguin, InsuredMine, Roots.ai, Chisel AI |
| GetLatka | Revenue data for Zywave ($87.1M), InsuredMine ($4.1M) |
| CB Insights | InsurTech funding trends (Q3 2025 report) |
| Allianz press release (Jan 2026) | Claude/Anthropic enterprise insurance validation |

### Technology & Market Trends

| Source | Data Used |
|---|---|
| EY GenAI in Insurance Survey | Carrier AI adoption rates |
| Wolters Kluwer 2025 Insurance Tech Trends | Cautious agency adoption |
| Sonant.ai 100+ AI Tools Guide | Emerging competitors landscape |
| PYMNTS.com | Agentic AI in insurance back office |
| Squirro State of RAG 2026 | RAG market maturation |

---

*This document is confidential and intended for VTM Group executive team use only. Market data is sourced from publicly available reports and industry surveys as of February 2026. Financial projections are estimates based on stated assumptions and should not be treated as guarantees.*
